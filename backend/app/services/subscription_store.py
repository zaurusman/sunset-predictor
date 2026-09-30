"""Where Web Push subscriptions and alert bookkeeping live.

Render's free tier has an ephemeral disk, so production uses Postgres (Neon).
Tests and local runs without DATABASE_URL use the in-memory store. Both honour
the same contract; see tests/test_subscription_store.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def cell_key(lat: float, lon: float, decimals: int = 1) -> str:
    """Grid cell for alert grouping — same rounding as the weather cache key,
    so an alert check and a user's own lookup share one Open-Meteo fetch."""
    return f"{round(lat, decimals):.{decimals}f},{round(lon, decimals):.{decimals}f}"


@dataclass
class StoredSubscription:
    endpoint: str
    p256dh: str
    auth: str
    places: list[dict]
    tz: str = "UTC"
    last_notified: dict[str, str] = field(default_factory=dict)


class SubscriptionStore(Protocol):
    async def upsert(self, endpoint: str, p256dh: str, auth: str, places: list[dict], tz: str) -> None: ...
    async def delete(self, endpoint: str) -> None: ...
    async def all(self) -> list[StoredSubscription]: ...
    async def mark_notified(self, endpoint: str, cell: str, day: date) -> None: ...
    async def cell_checked(self, cell: str, day: date) -> bool: ...
    async def record_cell_check(self, cell: str, day: date, score: float) -> None: ...


class InMemorySubscriptionStore:
    def __init__(self) -> None:
        self._subs: dict[str, StoredSubscription] = {}
        self._checks: set[tuple[str, str]] = set()

    async def upsert(self, endpoint, p256dh, auth, places, tz) -> None:
        prev = self._subs.get(endpoint)
        self._subs[endpoint] = StoredSubscription(
            endpoint, p256dh, auth, list(places), tz,
            dict(prev.last_notified) if prev else {},
        )

    async def delete(self, endpoint) -> None:
        self._subs.pop(endpoint, None)

    async def all(self) -> list[StoredSubscription]:
        return list(self._subs.values())

    async def mark_notified(self, endpoint, cell, day) -> None:
        if endpoint in self._subs:
            self._subs[endpoint].last_notified[cell] = day.isoformat()

    async def cell_checked(self, cell, day) -> bool:
        return (cell, day.isoformat()) in self._checks

    async def record_cell_check(self, cell, day, score) -> None:
        self._checks.add((cell, day.isoformat()))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS push_subscriptions (
    endpoint      text PRIMARY KEY,
    p256dh        text NOT NULL,
    auth          text NOT NULL,
    places        jsonb NOT NULL DEFAULT '[]',
    tz            text NOT NULL DEFAULT 'UTC',
    last_notified jsonb NOT NULL DEFAULT '{}',
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS alert_cell_checks (
    cell_key   text NOT NULL,
    local_date date NOT NULL,
    score      real NOT NULL,
    checked_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (cell_key, local_date)
);
"""


def _clean_dsn(dsn: str) -> str:
    """Neon's copy-paste DSN includes channel_binding, which asyncpg rejects."""
    parts = urlsplit(dsn)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "channel_binding"]
    return urlunsplit(parts._replace(query=urlencode(query)))


class PostgresSubscriptionStore:
    def __init__(self, pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> "PostgresSubscriptionStore":
        import asyncpg  # imported lazily so the app runs without the driver configured

        pool = await asyncpg.create_pool(_clean_dsn(dsn), min_size=1, max_size=3)
        await pool.execute(_SCHEMA)
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def upsert(self, endpoint, p256dh, auth, places, tz) -> None:
        await self._pool.execute(
            """
            INSERT INTO push_subscriptions (endpoint, p256dh, auth, places, tz)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (endpoint) DO UPDATE
               SET p256dh = EXCLUDED.p256dh, auth = EXCLUDED.auth,
                   places = EXCLUDED.places, tz = EXCLUDED.tz, updated_at = now()
            """,
            endpoint, p256dh, auth, json.dumps(places), tz,
        )

    async def delete(self, endpoint) -> None:
        await self._pool.execute("DELETE FROM push_subscriptions WHERE endpoint = $1", endpoint)

    async def all(self) -> list[StoredSubscription]:
        rows = await self._pool.fetch(
            "SELECT endpoint, p256dh, auth, places, tz, last_notified FROM push_subscriptions"
        )
        return [
            StoredSubscription(
                r["endpoint"], r["p256dh"], r["auth"],
                json.loads(r["places"]), r["tz"], json.loads(r["last_notified"]),
            )
            for r in rows
        ]

    async def mark_notified(self, endpoint, cell, day) -> None:
        await self._pool.execute(
            """
            UPDATE push_subscriptions
               SET last_notified = jsonb_set(last_notified, ARRAY[$2::text], to_jsonb($3::text))
             WHERE endpoint = $1
            """,
            endpoint, cell, day.isoformat(),
        )

    async def cell_checked(self, cell, day) -> bool:
        row = await self._pool.fetchrow(
            "SELECT 1 FROM alert_cell_checks WHERE cell_key = $1 AND local_date = $2", cell, day
        )
        return row is not None

    async def record_cell_check(self, cell, day, score) -> None:
        await self._pool.execute(
            """
            INSERT INTO alert_cell_checks (cell_key, local_date, score) VALUES ($1, $2, $3)
            ON CONFLICT (cell_key, local_date) DO NOTHING
            """,
            cell, day, score,
        )
