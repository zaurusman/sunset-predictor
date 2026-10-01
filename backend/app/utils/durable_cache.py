"""Postgres-backed durable tier for TTLCache's long-lived entries.

Render's disk is ephemeral, so the /tmp pickle is wiped on every deploy or
restart and each location then rebuilds its climatology from Open-Meteo
archive data (~111 weighted calls per location) — bursts suspected behind the
intermittent 503s (Open-Meteo 503 investigation, PRs #21/#23). Entries cached
for a day or longer (climatology curves, archive months, frozen evenings) are
written here and bulk-loaded at startup. Short forecast/ensemble entries are
never written, so Neon's compute can stay suspended between bursts.

This tier only moves bytes; TTLCache owns pickling and expiry semantics.
Values are stored in the same compressed form TTLCache keeps them in memory
(see encode_value), so a startup load never inflates them into live objects.
"""
from __future__ import annotations

import pickle
import zlib
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

# Entries whose effective TTL is at least this long are written through.
DURABLE_MIN_TTL_SECONDS = 86_400

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key        text PRIMARY KEY,
    value      bytea NOT NULL,
    expires_at timestamptz NOT NULL
);
"""


# zlib level 1: cached weather (lists of floats, repeated dict keys) shrinks
# ~6x at a fraction of a millisecond per entry. Higher levels gain little.
_ZLIB_LEVEL = 1
# Every pickle at protocol >= 2 starts with PROTO (0x80); a zlib stream never
# does. Rows written before compression was introduced are bare pickles.
_PICKLE_PROTO = b"\x80"


def encode_value(value: Any) -> bytes:
    """Pickle + compress one cache value. Raises if it can't be pickled."""
    return zlib.compress(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), _ZLIB_LEVEL)


def decode_value(blob: bytes) -> Any:
    """Inverse of encode_value; also reads legacy uncompressed pickles."""
    if blob[:1] == _PICKLE_PROTO:
        return pickle.loads(blob)
    return pickle.loads(zlib.decompress(blob))


def normalize_blob(blob: bytes) -> bytes:
    """A legacy bare pickle re-encoded compressed; anything else as is."""
    return zlib.compress(blob, _ZLIB_LEVEL) if blob[:1] == _PICKLE_PROTO else blob


def _to_ts(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


class PostgresCacheTier:
    """Reuses the app's asyncpg pool (the subscription store's) — no second pool."""

    def __init__(self, pool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        await self._pool.execute(_SCHEMA)

    async def load_all(
        self, grace: float, budget_bytes: Optional[int] = None
    ) -> list[tuple[str, bytes, float]]:
        """Rows not yet past expiry + *grace* (kept for get_stale), latest
        expiry first, stopping once *budget_bytes* of values are collected.

        Streamed through a cursor so the table can outgrow the instance's RAM
        without the startup load running it out of memory: before this, an
        OOM restart reloaded everything and ran out again.
        """
        since = datetime.now(timezone.utc) - timedelta(seconds=grace)
        rows: list[tuple[str, bytes, float]] = []
        total = 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                async for r in conn.cursor(
                    "SELECT key, value, expires_at FROM cache_entries"
                    " WHERE expires_at > $1 ORDER BY expires_at DESC",
                    since, prefetch=50,
                ):
                    blob = bytes(r["value"])
                    if budget_bytes is not None and total + len(blob) > budget_bytes:
                        break
                    total += len(blob)
                    rows.append((r["key"], blob, r["expires_at"].timestamp()))
        return rows

    async def write_many(self, rows: list[tuple[str, bytes, float]]) -> None:
        await self._pool.executemany(
            """
            INSERT INTO cache_entries (key, value, expires_at) VALUES ($1, $2, $3)
            ON CONFLICT (key) DO UPDATE
               SET value = EXCLUDED.value, expires_at = EXCLUDED.expires_at
            """,
            [(k, v, _to_ts(exp)) for k, v, exp in rows],
        )

    async def purge_expired(self, grace: float) -> None:
        await self._pool.execute(
            "DELETE FROM cache_entries WHERE expires_at <= $1",
            datetime.now(timezone.utc) - timedelta(seconds=grace),
        )
