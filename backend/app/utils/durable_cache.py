"""Postgres-backed durable tier for TTLCache's long-lived entries.

Render's disk is ephemeral, so the /tmp pickle is wiped on every deploy or
restart and each location then rebuilds its climatology from Open-Meteo
archive data (~111 weighted calls per location) — bursts suspected behind the
intermittent 503s (Open-Meteo 503 investigation, PRs #21/#23). Entries cached
for a day or longer (climatology curves, archive months, frozen evenings) are
written here and bulk-loaded at startup. Short forecast/ensemble entries are
never written, so Neon's compute can stay suspended between bursts.

This tier only moves bytes; TTLCache owns pickling and expiry semantics.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Entries whose effective TTL is at least this long are written through.
DURABLE_MIN_TTL_SECONDS = 86_400

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key        text PRIMARY KEY,
    value      bytea NOT NULL,
    expires_at timestamptz NOT NULL
);
"""


def _to_ts(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


class PostgresCacheTier:
    """Reuses the app's asyncpg pool (the subscription store's) — no second pool."""

    def __init__(self, pool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        await self._pool.execute(_SCHEMA)

    async def load_all(self, grace: float) -> list[tuple[str, bytes, float]]:
        """Every row not yet past expiry + *grace* (kept for get_stale)."""
        rows = await self._pool.fetch(
            "SELECT key, value, expires_at FROM cache_entries WHERE expires_at > $1",
            datetime.now(timezone.utc) - timedelta(seconds=grace),
        )
        return [(r["key"], bytes(r["value"]), r["expires_at"].timestamp()) for r in rows]

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
