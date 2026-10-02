"""Turn stored rating records into model inputs and a training table.

Shared by scripts/evaluate.py (accuracy against labels) and scripts/ratings.py
(the training export), so "what the engine says about this label today" and
"the features of this label" are computed one way.

A record is replayable when it carries its raw inputs: window_snapshots, and —
from schema 2 on — corridor_samples. Older records lack the corridor; callers
that can reach Open-Meteo refetch it (for a past date the archive is
deterministic), offline callers leave the replayed score empty rather than
guess, because the corridor gates every cloud pathway.
"""
from __future__ import annotations

import json
import math
from datetime import date
from typing import Any, Iterable, Optional

from app.schemas.weather import WeatherSnapshot
from app.services.rating_store import band_of, dedupe_latest, label_0_100
from app.services.scoring_engine import ScoringEngine

# The four moments the engine samples, in time order.
MOMENTS = ("-15m", "sunset", "+15m", "+30m")

Corridor = list[tuple[float, float, float]]


def snapshots_of(record: dict[str, Any]) -> list[WeatherSnapshot]:
    """The record's window snapshots, or [] when absent or unreadable."""
    try:
        return [WeatherSnapshot(**s) for s in record.get("window_snapshots") or []]
    except Exception:
        return []


def stored_corridor(record: dict[str, Any]) -> Optional[Corridor]:
    """The corridor captured with the record, or None when it predates
    schema 2. An empty list is a real answer — the corridor fetch failed at
    capture time and production scored without it — so it is kept."""
    raw = record.get("corridor_samples")
    if raw is None:
        return None
    try:
        return [(float(d), float(low), float(mid)) for d, low, mid in raw]
    except (TypeError, ValueError):
        return None


def rescore(
    engine: ScoringEngine,
    record: dict[str, Any],
    corridor: Optional[Corridor] = None,
    horizon_deg: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Score the record's raw inputs with *engine* as it is now.

    *corridor* overrides the stored one (for records older than schema 2).
    Returns None when there is nothing to replay; otherwise
    ``{"by_moment": {label: score}, "final": window score, "comparable": the
    score this label should be compared against}`` — the observed moment's
    score when the label names one, the window aggregate otherwise.
    """
    snaps = snapshots_of(record)
    if not snaps:
        return None
    if corridor is None:
        corridor = stored_corridor(record)
    if corridor is None:
        return None
    if horizon_deg is None:
        horizon_deg = float(record.get("horizon_deg") or 2.0)

    scored: list[tuple[str, float]] = []
    for snap in snaps:
        result = engine.score(snap, horizon_deg, corridor_samples=corridor)
        scored.append((snap.timestamp_label or "sunset", result.physics_score))
    by_moment = dict(scored)
    final = engine.score_window(scored).final_score

    moment = record.get("observed_moment")
    comparable = by_moment[moment] if moment in by_moment else final
    return {"by_moment": by_moment, "final": final, "comparable": comparable}


# ---------------------------------------------------------------------------
# Training table
# ---------------------------------------------------------------------------

# Numeric WeatherSnapshot fields carried into the table, per moment. Taken
# from the model rather than hand-listed, so a field added to the snapshot
# reaches training without touching this file.
_SKIP_FIELDS = {"timestamp_label", "data_source"}


def _snapshot_features(snap: WeatherSnapshot) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for name, value in snap.model_dump().items():
        if name in _SKIP_FIELDS:
            continue
        if isinstance(value, bool):
            out[name] = float(value)
        elif isinstance(value, (int, float)):
            out[name] = float(value) if math.isfinite(value) else None
    return out


def _corridor_features(corridor: Optional[Corridor]) -> dict[str, Optional[float]]:
    if not corridor:
        return {"corridor_n": 0.0 if corridor is not None else None,
                "corridor_low_mean": None, "corridor_mid_mean": None,
                "corridor_low_max": None, "corridor_mid_max": None}
    lows = [c[1] for c in corridor]
    mids = [c[2] for c in corridor]
    return {
        "corridor_n": float(len(corridor)),
        "corridor_low_mean": sum(lows) / len(lows),
        "corridor_mid_mean": sum(mids) / len(mids),
        "corridor_low_max": max(lows),
        "corridor_mid_max": max(mids),
    }


def training_row(
    record: dict[str, Any], engine: Optional[ScoringEngine] = None
) -> Optional[dict[str, Any]]:
    """One flat row for one label, or None when the record has no label.

    Columns:
      label_*     the target — rating_0_100, and whether it was precise
      meta_*      where/when/which engine; not features, used for splits
      captured_*  what production said at the time (stale by design)
      engine_*    the CURRENT engine on the stored inputs — the baseline a
                  fine-tune learns a correction to; empty when not replayable
      <moment>__<field>  every numeric snapshot field at each of the four
                  sampled moments, e.g. "sunset__cloud_low"
      corridor_*  summary of the upstream corridor
      corridor_json  the raw samples, for models that want all of them
    """
    label = label_0_100(record)
    if label is None:
        return None

    target = None
    try:
        target = date.fromisoformat(str(record.get("target_date")))
    except ValueError:
        pass

    row: dict[str, Any] = {
        "label_rating_0_100": label,
        "label_band": band_of(label),
        "label_is_precise": bool(record.get("rating_is_precise")),
        "meta_target_date": str(record.get("target_date")),
        "meta_recorded_at": record.get("recorded_at"),
        "meta_latitude": record.get("latitude"),
        "meta_longitude": record.get("longitude"),
        "meta_location_name": record.get("location_name"),
        "meta_observed_moment": record.get("observed_moment"),
        "meta_schema_version": record.get("schema_version"),
        "meta_algorithm_version": record.get("algorithm_version"),
        "meta_has_raw_inputs": bool(record.get("window_snapshots")),
        "meta_inputs_backfilled": bool(record.get("inputs_backfilled")),
        "meta_context_error": record.get("context_error"),
        "month": float(target.month) if target else None,
        "month_sin": math.sin(target.month * 2 * math.pi / 12) if target else None,
        "month_cos": math.cos(target.month * 2 * math.pi / 12) if target else None,
        "horizon_deg": record.get("horizon_deg"),
        "captured_score": record.get("predicted_score"),
        "captured_score_at_observed_moment": record.get("predicted_score_at_observed_moment"),
        "captured_raw_physics_score": record.get("raw_physics_score"),
        "captured_climatology_percentile": record.get("climatology_percentile"),
        "captured_confidence": record.get("predicted_confidence"),
    }

    snaps = {s.timestamp_label or "sunset": s for s in snapshots_of(record)}
    for moment in MOMENTS:
        snap = snaps.get(moment)
        if snap is not None:
            for name, value in _snapshot_features(snap).items():
                row[f"{moment}__{name}"] = value

    corridor = stored_corridor(record)
    row.update(_corridor_features(corridor))
    row["corridor_json"] = json.dumps(corridor) if corridor is not None else None

    replay = rescore(engine, record) if engine else None
    row["engine_score"] = replay["comparable"] if replay else None
    row["engine_window_score"] = replay["final"] if replay else None
    for moment in MOMENTS:
        row[f"engine_score_{moment}"] = replay["by_moment"].get(moment) if replay else None
    return row


def training_rows(
    records: Iterable[dict[str, Any]], engine: Optional[ScoringEngine] = None
) -> list[dict[str, Any]]:
    """One row per (evening, place) — deduplicated exactly as GET
    /ratings/stats and the evaluator do."""
    rows = (training_row(rec, engine) for rec in dedupe_latest(list(records)))
    return [row for row in rows if row is not None]
