"""
Human sunset ratings: move them, and turn them into a training table
====================================================================

Run from backend/. SOURCE / DEST are a JSONL path or a Postgres URL
(production's is DATABASE_URL in backend/.env).

    # Production ratings -> a local JSONL snapshot (full records, raw inputs)
    python scripts/ratings.py export "$DATABASE_URL" data/ratings.jsonl

    # Local ratings (e.g. photo labels) -> production. Safe to re-run:
    # records already stored are skipped.
    python scripts/ratings.py import data/ratings.jsonl "$DATABASE_URL"

    # Training table: one row per (evening, place), label + features +
    # the current engine's score on the stored inputs
    python scripts/ratings.py dataset "$DATABASE_URL" data/ratings_dataset.parquet
    python scripts/ratings.py dataset data/ratings.jsonl data/ratings_dataset.csv --backfill

THE TRAINING TABLE
------------------
See app/services/rating_dataset.training_row for every column. The short
version: `label_rating_0_100` is the target, `<moment>__<field>` are the raw
weather features at each sampled moment, and `engine_score` is what the
physics engine says TODAY about the same inputs. Two ways to use it:

  - train from scratch on the weather features, or
  - fine-tune: learn `label_rating_0_100 - engine_score`, a correction on top
    of the physics, which needs far fewer labels than learning the sky.

Split by `meta_target_date`, never randomly: two ratings of the same evening
from different places share a sky, and a random split leaks it.

--backfill fetches inputs a record is missing from Open-Meteo: the window
snapshots of a rating stored while the weather was unavailable, and the
corridor of a rating older than schema 2. Backfilled rows are flagged
(meta_inputs_backfilled) — archive weather is a reanalysis, not the forecast
production scored, so keep them out of any "how good was the live forecast"
question.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rating_dataset import snapshots_of, stored_corridor, training_rows
from app.services.rating_store import (
    PostgresRatingStore,
    RatingStore,
    canonical_json,
    dedupe_latest,
    is_database_url,
    load_records,
)
from app.services.scoring_engine import ScoringEngine


async def export(source: str, dest: str) -> None:
    records = await load_records(source)
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(canonical_json(rec) + "\n")
    print(f"Wrote {len(records)} rating(s) to {out}")


async def import_(source: str, dest: str) -> None:
    records = await load_records(source)
    if is_database_url(dest):
        store = await PostgresRatingStore.connect(dest)
        try:
            added = await store.append_many(records)
            total = await store.total()
        finally:
            await store.close()
        print(f"Imported {added} new rating(s) of {len(records)}; {total} stored now.")
        return
    store = RatingStore(dest)
    for rec in records:
        await store.append(rec)
    print(f"Appended {len(records)} rating(s) to {dest}")


async def backfill(records: list[dict]) -> int:
    """Fill missing raw inputs in place from Open-Meteo. Returns how many
    records changed."""
    import httpx

    from app.core.config import Settings
    from app.services.astronomy_service import AstronomyService
    from app.services.weather_service import WeatherService
    from app.utils.cache import TTLCache

    astro = AstronomyService()
    changed = 0
    async with httpx.AsyncClient(timeout=90.0) as http:
        weather = WeatherService(
            http_client=http,
            astro_service=astro,
            cache=TTLCache(ttl_seconds=86_400, persist_path=None),
            settings=Settings(),
        )
        for rec in records:
            need_snaps = not snapshots_of(rec)
            need_corridor = stored_corridor(rec) is None
            if not (need_snaps or need_corridor):
                continue
            try:
                target = date.fromisoformat(str(rec.get("target_date")))
                lat, lon = float(rec["latitude"]), float(rec["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            sunset = astro.get_sunset_time(lat, lon, target)
            try:
                if need_snaps:
                    snaps = await weather.get_window_snapshots(lat, lon, target, sunset)
                    rec["window_snapshots"] = [s.model_dump(mode="json") for s in snaps]
                if need_corridor:
                    corridor = await weather.get_corridor_samples(lat, lon, target, sunset)
                    rec["corridor_samples"] = [list(c) for c in corridor]
            except Exception as exc:
                print(f"  could not backfill {target} ({lat:.3f}, {lon:.3f}): {exc}")
                continue
            rec["inputs_backfilled"] = True
            changed += 1
    return changed


async def dataset(source: str, dest: str, do_backfill: bool) -> None:
    import pandas as pd

    records = dedupe_latest(await load_records(source))
    if do_backfill:
        n = await backfill(records)
        print(f"Backfilled raw inputs for {n} rating(s)")

    rows = training_rows(records, ScoringEngine())
    df = pd.DataFrame(rows)
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)

    replayable = int(df["engine_score"].notna().sum()) if len(df) else 0
    print(f"Wrote {len(df)} row(s) x {len(df.columns)} column(s) to {out}")
    print(f"  {replayable} with a current-engine score; "
          f"{len(df) - replayable} missing raw inputs (try --backfill)")
    if len(df):
        bands = df["label_band"].value_counts().sort_index().to_dict()
        print(f"  label bands (1=nothing .. 5=exceptional): {json.dumps(bands)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export", help="ratings -> JSONL (full records)")
    p.add_argument("source")
    p.add_argument("dest")

    p = sub.add_parser("import", help="ratings -> a store, skipping ones already there")
    p.add_argument("source")
    p.add_argument("dest")

    p = sub.add_parser("dataset", help="ratings -> training table (.parquet or .csv)")
    p.add_argument("source")
    p.add_argument("dest")
    p.add_argument("--backfill", action="store_true",
                   help="fetch missing raw inputs from Open-Meteo first")

    args = ap.parse_args()
    if args.cmd == "export":
        asyncio.run(export(args.source, args.dest))
    elif args.cmd == "import":
        asyncio.run(import_(args.source, args.dest))
    else:
        asyncio.run(dataset(args.source, args.dest, args.backfill))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
