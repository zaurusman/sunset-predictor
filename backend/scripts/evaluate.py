"""
Scoring evaluation harness
==========================

The measuring stick from docs/scoring-v2-plan.md, Phase 0. Every claim about a
scoring change should be reproduced with this rather than with a throwaway
script, so the numbers in commit messages can be checked.

Run from backend/:

    python scripts/evaluate.py                       # 3 default cities, 1 year
    python scripts/evaluate.py --days 730            # 2 years
    python scripts/evaluate.py --labels data/ratings.jsonl

WHAT IT MEASURES, AND WHAT IT DOES NOT
--------------------------------------
Without human labels this reports DISTRIBUTION properties only:

  - spread and percentiles (is the score degenerate?)
  - category shares (is "Epic" actually rare? is "Poor" ever used?)
  - per-component variance contribution (is any weight carrying no information?)

Those are necessary but NOT sufficient. A distribution can look healthy while
the ordering is wrong: demoting good evenings and demoting bad ones produce the
same histogram. Only --labels, which correlates scores against ratings collected
via POST /rate, measures whether the engine is actually right. Until enough
ratings exist, treat everything here as "less broken", not "more accurate".

It scores through the REAL WeatherService, so the light corridor, caching and
data-source switching are all exercised exactly as they are in production.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings
from app.services.astronomy_service import AstronomyService
from app.services.scoring_engine import ScoringEngine
from app.services.climatology_service import _rank_in_sorted
from app.services.weather_service import WeatherService
from app.utils.cache import TTLCache
from app.utils.math_utils import spearman

DEFAULT_LOCATIONS = [
    ("TelAviv", 32.08, 34.78),
    ("London", 51.51, -0.13),
    ("SanFrancisco", 37.77, -122.42),
]

COMPONENTS = ["cloud_quality", "atmosphere", "moisture"]
CATEGORY_ORDER = ["Poor", "Decent", "Good", "Great", "Epic"]

# Guardrails. A healthy daily-glance app cannot call 15 % of evenings "Epic",
# and a component holding weight while contributing no variance is dead weight.
MAX_EPIC_SHARE = 8.0        # per cent of days
MAX_SINGLE_BAND_SHARE = 65.0
MIN_P10_P50_GAP = 3.0       # points; below this the score is degenerate
# The same degeneracy check at the TOP of the raw scale. Percentile calibration
# hides compression there — the displayed bands stay perfectly shaped while the
# raw scores they rank are all crowded into a few points, so the ranking is
# decided by noise. Added after the clear-sky pathway pushed Tel Aviv's raw p50
# to 80 and its p90 to 87 with every guardrail still reporting green.
MIN_P50_P90_GAP = 8.0


async def collect(
    name: str, lat: float, lon: float, days: int, horizon_deg: float
) -> dict:
    """Score every day of the past *days* at one location, through the real stack."""
    settings = Settings()
    astro = AstronomyService()
    engine = ScoringEngine()

    async with httpx.AsyncClient(timeout=90.0) as http:
        weather = WeatherService(
            http_client=http,
            astro_service=astro,
            cache=TTLCache(ttl_seconds=86_400, persist_path=None),
            settings=settings,
        )

        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=days - 1)

        windows = await weather.get_historical_range_windows(lat, lon, start, end)
        corridor_map = await weather.get_corridor_samples_map(
            lat, lon, [d for d, _ in windows]
        )

        finals: list[float] = []
        per_component: dict[str, list[float]] = {c: [] for c in COMPONENTS}
        corridor_values: list[float] = []
        pathway_wins: dict[str, int] = {}
        pathway_best: dict[str, float] = {}

        for d, snaps in windows:
            samples = corridor_map.get(d, [])
            scored: list[tuple[str, float]] = []
            results = []
            for snap in snaps:
                r = engine.score(snap, horizon_deg, corridor_samples=samples)
                scored.append((snap.timestamp_label or "sunset", r.physics_score))
                results.append(r)
            if not scored:
                continue
            finals.append(engine.score_window(scored).final_score)

            # Report the components of the BEST window point, not the first one.
            # These two diverge now that the clear-sky pathway peaks several
            # degrees below the horizon: reporting the -15m snapshot showed a
            # mean cloud_quality of 25 against a mean final score of 75, which
            # made the component table look broken when it was just measuring
            # the wrong moment.
            best = max(results, key=lambda r: r.physics_score)
            # Which route to beauty carried the evening. If one pathway wins
            # nearly always, the model has collapsed back to a single recipe
            # however many functions it contains; if one never wins, it is
            # dead code pretending to be coverage.
            winner = engine.dominant_pathway(best.pathways)
            if winner:
                pathway_wins[winner] = pathway_wins.get(winner, 0) + 1
            for k, v in (best.pathways or {}).items():
                pathway_best[k] = max(pathway_best.get(k, 0.0), v)
            for c in COMPONENTS:
                per_component[c].append(getattr(best, c))
            if best.light_corridor is not None:
                corridor_values.append(best.light_corridor)

    # Calibrated view: rank each evening against the year we just scored, which
    # is exactly what ClimatologyService does once a location is warm. Ranking
    # in-sample flatters by ~1/n (0.3 % here) — negligible at 365 days.
    curve = sorted(finals)
    calibrated = [
        engine.percentile_to_display_score(_rank_in_sorted(curve, v)) for v in finals
    ]

    return {
        "name": name,
        "finals": finals,
        "calibrated": calibrated,
        "components": per_component,
        "corridor": corridor_values,
        "pathway_wins": pathway_wins,
        "pathway_best": pathway_best,
        "weights": engine._weights,
    }


def report(result: dict) -> list[str]:
    """Print one location's report; return a list of guardrail failures."""
    name, finals = result["name"], result["finals"]
    engine = ScoringEngine()
    failures: list[str] = []

    if len(finals) < 30:
        print(f"{name}: only {len(finals)} scored days — skipping report")
        return [f"{name}: too few days ({len(finals)})"]

    n = len(finals)
    ordered = sorted(finals)
    pct = lambda p: ordered[int(p / 100 * (n - 1))]
    p10, p50 = pct(10), pct(50)

    print(f"\n{'=' * 74}\n{name}  (n={n})\n{'=' * 74}")
    print(
        f"  mean={st.mean(finals):5.1f}  sd={st.stdev(finals):5.1f}  "
        f"p10={p10:5.1f}  p50={p50:5.1f}  p90={pct(90):5.1f}  p99={pct(99):5.1f}"
    )

    def bands(values: list[float]) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in values:
            c = engine.score_to_category(v)
            out[c] = out.get(c, 0) + 1
        return out

    raw_cats = bands(finals)
    print("  bands (raw):        "
          + "  ".join(f"{c}:{raw_cats.get(c, 0) * 100 / n:5.1f}%" for c in CATEGORY_ORDER))

    cal = result.get("calibrated") or []
    cats = raw_cats
    if cal:
        cats = bands(cal)
        ordered_cal = sorted(cal)
        pc = lambda q: ordered_cal[int(q / 100 * (n - 1))]
        print("  bands (calibrated): "
              + "  ".join(f"{c}:{cats.get(c, 0) * 100 / n:5.1f}%" for c in CATEGORY_ORDER))
        print(f"  calibrated: mean={st.mean(cal):5.1f}  sd={st.stdev(cal):5.1f}  "
              f"p10={pc(10):5.1f}  p50={pc(50):5.1f}  p90={pc(90):5.1f}")
        print(f"  go-outside (>= 70): {sum(1 for v in cal if v >= 70) * 100 / n:.1f}% of evenings")

    if result["corridor"]:
        cv = result["corridor"]
        print(f"  light corridor: mean={st.mean(cv):.2f}  min={min(cv):.2f}  "
              f"days below 0.6: {sum(1 for v in cv if v < 0.6) * 100 / len(cv):.0f}%")

    wins = result.get("pathway_wins") or {}
    if wins:
        print("\n  pathway            wins   best seen")
        total_wins = sum(wins.values()) or 1
        for k in sorted(
            set(wins) | set(result.get("pathway_best") or {}),
            key=lambda k: -wins.get(k, 0),
        ):
            share = wins.get(k, 0) * 100 / total_wins
            print(f"  {k:18s} {share:5.1f}%   {(result['pathway_best'] or {}).get(k, 0.0):6.1f}")
        top = max(wins.values()) * 100 / total_wins
        if top > 92.0:
            failures.append(
                f"{name}: one pathway wins {top:.0f}% of evenings — the model has "
                f"collapsed back to a single recipe"
            )

    # Per-component variance share — the diagnostic that found the dead weight.
    print("\n  component        mean     sd   weight   variance share   pinned@100")
    weights = result["weights"]
    sds = {c: (st.stdev(v) if len(v) > 1 else 0.0) for c, v in result["components"].items()}
    total = sum(weights.get(c, 0.0) * sds[c] for c in COMPONENTS) or 1.0
    for c in COMPONENTS:
        vals = result["components"][c]
        share = weights.get(c, 0.0) * sds[c] / total * 100
        pinned = sum(1 for x in vals if x >= 99.5) / len(vals) * 100
        print(f"  {c:14s} {st.mean(vals):6.1f} {sds[c]:6.1f}   {weights.get(c, 0):.2f}  "
              f"{share:12.1f}%   {pinned:8.1f}%")
        if weights.get(c, 0.0) >= 0.10 and share < 10.0:
            # NOTE: this share counts a component's DIRECT weighted
            # contribution only. atmosphere and moisture also feed the
            # clear-sky twilight pathway inside cloud_quality, so their real
            # influence is larger than the number here. Read a failure on
            # those two as "worth checking", not as proof of dead weight —
            # in a place with uniformly clean air (San Francisco: atmosphere
            # mean 92, sd 7) a low share can simply mean the signal is real
            # and near-constant.
            indirect = " (also feeds the clear-sky pathway — see note in code)" \
                if c in ("atmosphere", "moisture") else ""
            failures.append(
                f"{name}: '{c}' holds {weights[c]:.0%} of the weight but only "
                f"{share:.1f}% of the variance — dead weight{indirect}"
            )

    # Guardrails
    epic = cats.get("Epic", 0) * 100 / n
    if epic > MAX_EPIC_SHARE:
        failures.append(f"{name}: 'Epic' fires on {epic:.1f}% of days (max {MAX_EPIC_SHARE}%)")
    top_band, top_share = max(cats.items(), key=lambda kv: kv[1])
    if top_share * 100 / n > MAX_SINGLE_BAND_SHARE:
        failures.append(
            f"{name}: '{top_band}' covers {top_share * 100 / n:.1f}% of days "
            f"(max {MAX_SINGLE_BAND_SHARE}%) — the bands are miscalibrated"
        )
    if pct(90) - p50 < MIN_P50_P90_GAP:
        failures.append(
            f"{name}: raw p50 and p90 are {pct(90) - p50:.1f} points apart — the top "
            f"half of the scale is compressed, so ranking within it is noise"
        )
    if p50 - p10 < MIN_P10_P50_GAP:
        failures.append(
            f"{name}: p10 and p50 are {p50 - p10:.1f} points apart — the score is degenerate"
        )
    return failures


def report_labels(path: str) -> None:
    """Correlate stored human ratings against the scores recorded at capture time.

    This is the only part of this harness that measures accuracy rather than
    distribution shape.
    """
    p = Path(path)
    if not p.exists():
        print(f"\nNo label file at {path} — skipping accuracy check.")
        return

    human: list[float] = []
    model: list[float] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        r, s = rec.get("rating"), rec.get("predicted_score")
        if isinstance(r, int) and isinstance(s, (int, float)):
            human.append(float(r))
            model.append(float(s))

    print(f"\n{'=' * 74}\nACCURACY vs {len(human)} human rating(s)\n{'=' * 74}")
    if len(human) < 15:
        print(f"  Need at least 15 to say anything; have {len(human)}.")
        print("  Rate evenings via POST /rate — including the dull ones.")
        return

    rho = spearman(human, model)
    low = sum(1 for h in human if h <= 2)
    high = sum(1 for h in human if h >= 4)
    print(f"  Spearman rho = {rho:.3f}" if rho is not None else "  rho undefined")
    print(f"  {low} poor evenings, {high} good ones")
    if low == 0 or high == 0:
        print("  WARNING: one-sided labels. rho is not trustworthy without both ends.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--horizon-deg", type=float, default=2.0)
    ap.add_argument("--labels", type=str, default="data/ratings.jsonl")
    ap.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero if any distribution guardrail fails (for CI).",
    )
    args = ap.parse_args()

    failures: list[str] = []
    for name, lat, lon in DEFAULT_LOCATIONS:
        result = asyncio.run(collect(name, lat, lon, args.days, args.horizon_deg))
        failures.extend(report(result))

    report_labels(args.labels)

    print(f"\n{'=' * 74}")
    if failures:
        print("GUARDRAIL FAILURES")
        for f in failures:
            print(f"  - {f}")
    else:
        print("All distribution guardrails passed.")
    print(
        "\nNOTE: distribution health is not accuracy. Until the label count above\n"
        "is meaningful, this says the engine is less broken — not that it is right."
    )
    return 1 if (failures and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
