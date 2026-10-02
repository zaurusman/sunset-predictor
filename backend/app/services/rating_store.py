"""Where human sunset ratings (ML training labels) are kept.

Two stores honour the same contract (append / records / total):

  - PostgresRatingStore — production. Render's disk is EPHEMERAL, so a file
    there is wiped on every deploy and restart; ratings go to the same Neon
    database as push subscriptions and the durable weather cache.
  - RatingStore — newline-delimited JSON, one record per line. Tests, local
    runs without DATABASE_URL, and the interchange format the offline tools
    read (scripts/ratings.py export / import, scripts/evaluate.py --labels).

THE RECORD IS THE UNIT
----------------------
Both stores keep the whole record exactly as POST /rate built it. Postgres
stores it as JSONB next to a few typed columns copied out of it for querying;
the columns are a convenience, the JSON is the truth. Each record carries the
RAW inputs (window snapshots, corridor samples), not just the score, so a
future scoring change or a trained model can be replayed offline against the
same labels without refetching a year of weather history. See
scripts/ratings.py for turning records into a training table.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from app.core.logging import get_logger
from app.schemas.rating import COARSE_TO_0_100

logger = get_logger(__name__)

# Version of the record POST /rate writes. Bump it when a field is added or
# its meaning changes; readers branch on it, nothing is ever migrated.
#   1 - label, captured scores, window_snapshots
#   2 - adds corridor_samples, horizon_deg, raw_physics_score,
#       climatology_percentile, context_error
RATING_SCHEMA_VERSION = 2

# How close two ratings must be to count as the same place. ~5 km at this
# value, matching RatingStore.find(): a jittery GPS fix, or rating the same
# evening from home and then from the beach a few streets away, is ONE
# observation of that evening, not two.
DEFAULT_DEDUPE_TOLERANCE_DEG = 0.05


def label_0_100(rec: dict[str, Any]) -> Optional[float]:
    """The human label from one record, on the canonical 0-100 scale.

    Records written since the scale change carry `rating_0_100` directly.
    Older ones carry only the 1-5 tap value, which is converted to its band
    CENTRE — the honest reading of a five-way choice.

    Returns None when the record has no usable rating at all.
    """
    precise = rec.get("rating_0_100")
    if isinstance(precise, (int, float)):
        return float(precise)
    coarse = rec.get("rating")
    if isinstance(coarse, int) and coarse in COARSE_TO_0_100:
        return COARSE_TO_0_100[coarse]
    return None


def band_of(score_0_100: float) -> int:
    """Which of the five coarse bands a 0-100 label falls in (1-5).

    For histograms and the "are both ends represented?" check, where 100
    buckets over a few dozen ratings would be unreadable.
    """
    return max(1, min(5, int(score_0_100 // 20) + 1))


def dedupe_latest(
    records: Iterator[dict[str, Any]] | list[dict[str, Any]],
    tolerance_deg: float = DEFAULT_DEDUPE_TOLERANCE_DEG,
) -> list[dict[str, Any]]:
    """One record per (evening, place), last write wins.

    Lives at module level, and is the ONLY implementation, because there used
    to be two: this store deduped on coordinates rounded to 2 dp (~1.1 km)
    while find() matched within 0.05 deg (~5 km). Ratings 1-3 km apart were
    therefore "already rated tonight" for the purpose of showing the UI, but
    two separate observations for the purpose of measuring accuracy. That is
    how one evening ended up counted twice in the label set, once with a score
    from an engine several commits old.

    Clusters greedily rather than rounding to a grid: rounding puts two points
    1 km apart into different buckets whenever they straddle a boundary, which
    is the bug it looks like it is avoiding.
    """
    by_date: dict[str, list[tuple[float, float, dict[str, Any]]]] = {}

    for rec in records:
        day = str(rec.get("target_date"))
        try:
            lat = float(rec.get("latitude", 0.0))
            lon = float(rec.get("longitude", 0.0))
        except (TypeError, ValueError):
            continue

        clusters = by_date.setdefault(day, [])
        for i, (clat, clon, _) in enumerate(clusters):
            if abs(clat - lat) <= tolerance_deg and abs(clon - lon) <= tolerance_deg:
                # Later line wins; keep the first coordinates as the cluster
                # centre so a slow drift across many ratings cannot walk the
                # cluster arbitrarily far from where it started.
                clusters[i] = (clat, clon, rec)
                break
        else:
            clusters.append((lat, lon, rec))

    return [rec for clusters in by_date.values() for _, _, rec in clusters]


class RatingStore:
    """Append-only JSONL store. Safe for concurrent writes within one process."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def append(self, record: dict[str, Any]) -> int:
        """Append *record* and return the new total row count.

        The write is serialised through an asyncio lock and flushed to disk so
        a crash between ratings cannot lose an acknowledged row.
        """
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, default=str, ensure_ascii=False)
            # Open in append mode per write: the file stays consistent even if
            # the process dies, and the volume never justifies a held handle.
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            return self._count_unlocked()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield stored records, skipping any corrupt lines.

        A truncated final line (power loss mid-write) must not make the whole
        dataset unreadable, so parse failures are logged and skipped.
        """
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping corrupt rating record at %s:%d", self._path, lineno
                    )

    def count(self) -> int:
        return self._count_unlocked()

    def describe(self) -> str:
        return str(self._path)

    async def records(self, with_raw: bool = True) -> list[dict[str, Any]]:
        """Every record, oldest first (the store contract shared with
        PostgresRatingStore). *with_raw* is accepted for symmetry; reading a
        local file whole costs nothing worth saving."""
        return list(self.iter_records())

    async def total(self) -> int:
        return self._count_unlocked()

    def _count_unlocked(self) -> int:
        return sum(1 for _ in self.iter_records())

    def latest_per_evening(self) -> list[dict[str, Any]]:
        """One record per (date, location) — the last rating given.

        The store is append-only, so a user who taps "dull", reconsiders and
        taps "pleasant" leaves TWO records behind. find() already resolves that
        with last-write-wins, but anything reading the file in bulk — the stats
        endpoint, the accuracy check in scripts/evaluate.py — was counting both
        and treating a changed mind as two independent observations.

        With a handful of labels that is not a rounding error: it inflates the
        count that gates the correlation, and it double-weights exactly the
        evenings someone was uncertain about.

        See dedupe_latest() for why "same place" is a distance and not a
        rounded grid key.
        """
        return dedupe_latest(self.iter_records())

    def find(
        self, latitude: float, longitude: float, target_date: str, tolerance_deg: float = 0.05
    ) -> Optional[dict[str, Any]]:
        """Return the most recent rating for this date near this location, if any.

        Coordinates are matched within *tolerance_deg* (~5 km at the default)
        so a slightly jittery GPS fix still counts as "already rated tonight".
        """
        match: Optional[dict[str, Any]] = None
        for rec in self.iter_records():
            if rec.get("target_date") != target_date:
                continue
            if abs(rec.get("latitude", 999) - latitude) > tolerance_deg:
                continue
            if abs(rec.get("longitude", 999) - longitude) > tolerance_deg:
                continue
            match = rec  # keep scanning; last write wins
        return match


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------

# Keys holding the raw model inputs — the bulk of every record. Readers that
# only need labels and stored scores (GET /ratings/stats) skip them.
RAW_KEYS = ("window_snapshots", "corridor_samples")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sunset_ratings (
    id                bigserial PRIMARY KEY,
    record_hash       text NOT NULL UNIQUE,
    recorded_at       timestamptz NOT NULL DEFAULT now(),
    target_date       date,
    latitude          double precision,
    longitude         double precision,
    rating_0_100      real,
    rating_is_precise boolean,
    observed_moment   text,
    predicted_score   real,
    algorithm_version text,
    schema_version    integer,
    has_raw_inputs    boolean NOT NULL DEFAULT false,
    record            jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS sunset_ratings_target_date_idx ON sunset_ratings (target_date);
"""
# The typed columns are copies of fields inside `record`, for ad-hoc SQL
# ("every rating since the corridor fix", "how many precise labels") without
# JSON operators. `record` is the source of truth and is never rewritten.
# All of them are nullable so a legacy record missing a field still imports.


def _finite(value: Any) -> Any:
    """NaN/Infinity -> None, recursively. Python's json emits them, Postgres
    JSONB rejects them, and one stray NaN in a snapshot must not cost a label."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def canonical_json(record: dict[str, Any]) -> str:
    """The one serialisation of a record: stable key order, no NaN."""
    return json.dumps(
        _finite(record), default=str, ensure_ascii=False, sort_keys=True, allow_nan=False
    )


def record_hash(record: dict[str, Any]) -> str:
    """Identity of a record. Every record carries its recorded_at timestamp,
    so two genuine ratings never collide — but re-importing the same file
    does, which is what makes `scripts/ratings.py import` safe to re-run."""
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _as_date(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def _as_timestamp(value: Any) -> datetime:
    try:
        ts = datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.now(timezone.utc)
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _row_args(record: dict[str, Any]) -> tuple:
    rating = label_0_100(record)
    precise = record.get("rating_is_precise")
    version = record.get("schema_version")
    moment = record.get("observed_moment")
    algo = record.get("algorithm_version")
    return (
        record_hash(record),
        _as_timestamp(record.get("recorded_at")),
        _as_date(record.get("target_date")),
        _as_float(record.get("latitude")),
        _as_float(record.get("longitude")),
        rating,
        precise if isinstance(precise, bool) else None,
        moment if isinstance(moment, str) else None,
        _as_float(record.get("predicted_score")),
        algo if isinstance(algo, str) else None,
        version if isinstance(version, int) and not isinstance(version, bool) else None,
        bool(record.get("window_snapshots")),
        canonical_json(record),
    )


_INSERT = """
INSERT INTO sunset_ratings (
    record_hash, recorded_at, target_date, latitude, longitude, rating_0_100,
    rating_is_precise, observed_moment, predicted_score, algorithm_version,
    schema_version, has_raw_inputs, record
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb)
ON CONFLICT (record_hash) DO NOTHING
"""


class PostgresRatingStore:
    """Append-only ratings table. Shares the app's asyncpg pool (the
    subscription store's) rather than opening a second one."""

    def __init__(self, pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> "PostgresRatingStore":
        """Own pool, for the offline scripts. The app passes its shared one."""
        import asyncpg  # lazily, as in subscription_store

        from app.services.subscription_store import _clean_dsn

        pool = await asyncpg.create_pool(_clean_dsn(dsn), min_size=1, max_size=2)
        store = cls(pool)
        await store.ensure_schema()
        return store

    async def ensure_schema(self) -> None:
        await self._pool.execute(_SCHEMA)

    async def close(self) -> None:
        await self._pool.close()

    def describe(self) -> str:
        return "postgres:sunset_ratings"

    async def append(self, record: dict[str, Any]) -> int:
        """Insert *record* and return the new total row count.

        Raises on a database failure: the caller must not tell someone their
        rating was saved when it was not.
        """
        await self._pool.execute(_INSERT, *_row_args(record))
        return await self.total()

    async def append_many(self, records: list[dict[str, Any]]) -> int:
        """Insert *records*, skipping any already stored. Returns how many
        were new."""
        before = await self.total()
        await self._pool.executemany(_INSERT, [_row_args(r) for r in records])
        return await self.total() - before

    async def records(self, with_raw: bool = True) -> list[dict[str, Any]]:
        """Every record, oldest first — the order dedupe_latest needs for
        last-write-wins. Without *with_raw* the bulky raw inputs are dropped
        in the database, not after transferring them."""
        column = "record"
        if not with_raw:
            column = "record" + "".join(f" - '{k}'" for k in RAW_KEYS)
        rows = await self._pool.fetch(f"SELECT {column} AS record FROM sunset_ratings ORDER BY id")
        return [json.loads(r["record"]) for r in rows]

    async def total(self) -> int:
        return int(await self._pool.fetchval("SELECT count(*) FROM sunset_ratings"))


def is_database_url(source: str) -> bool:
    return source.startswith(("postgres://", "postgresql://"))


async def load_records(source: str) -> list[dict[str, Any]]:
    """Every record from a JSONL path or a Postgres URL, oldest first.

    The single entry point for offline tools, so evaluation and training read
    labels the same way whichever store they came from.
    """
    if is_database_url(source):
        store = await PostgresRatingStore.connect(source)
        try:
            return await store.records()
        finally:
            await store.close()
    return await RatingStore(source).records()
