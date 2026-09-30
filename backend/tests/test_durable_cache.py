"""TTLCache's durable (Postgres) tier.

Long-lived entries (>= 1 day: climatology curves, archive months, frozen
evenings) are written through to the database so a Render restart doesn't
force a full Open-Meteo rebuild of every location's climatology.
"""
from __future__ import annotations

import asyncio
import os
import pickle
import time
from datetime import date, datetime, timezone

import pytest

from app.utils.cache import TTLCache
from app.utils.durable_cache import DURABLE_MIN_TTL_SECONDS, PostgresCacheTier

DAY = 86_400


class FakeTier:
    """In-memory stand-in for PostgresCacheTier (same contract)."""

    def __init__(self, rows=None, fail_writes=False):
        self.rows: dict[str, tuple[bytes, float]] = dict(rows or {})
        self.write_calls: list[list[tuple[str, bytes, float]]] = []
        self.purges = 0
        self.loads = 0
        self.fail_writes = fail_writes

    async def load_all(self, grace):
        self.loads += 1
        now = time.time()
        return [(k, v, exp) for k, (v, exp) in self.rows.items() if exp + grace > now]

    async def write_many(self, rows):
        self.write_calls.append(list(rows))
        if self.fail_writes:
            raise ConnectionError("neon is asleep")
        for k, v, exp in rows:
            self.rows[k] = (v, exp)

    async def purge_expired(self, grace):
        self.purges += 1
        now = time.time()
        for k in [k for k, (_, exp) in self.rows.items() if exp + grace <= now]:
            del self.rows[k]


def _run(coro):
    return asyncio.run(coro)


def test_threshold_is_one_day():
    assert DURABLE_MIN_TTL_SECONDS == DAY


def test_only_long_lived_entries_are_written_through():
    tier = FakeTier()

    async def go():
        cache = TTLCache(ttl_seconds=7200)
        await cache.attach_durable(tier, linger_seconds=0)
        cache.set("forecast", 1)                              # default 2 h
        cache.set("ensemble", 2, ttl_override=3600)
        cache.set("frozen", 3, ttl_override=DAY)
        cache.set("climatology", [4.0, 5.0], ttl_override=30 * DAY)
        await cache.close_durable()

    _run(go())
    assert set(tier.rows) == {"frozen", "climatology"}
    assert pickle.loads(tier.rows["climatology"][0]) == [4.0, 5.0]


def test_a_burst_of_sets_is_flushed_in_one_write():
    tier = FakeTier()

    async def go():
        cache = TTLCache()
        await cache.attach_durable(tier, linger_seconds=0.05)
        for i in range(50):
            cache.set(f"archive_month_{i}", {"i": i}, ttl_override=30 * DAY)
        cache.set("archive_month_0", {"i": "again"}, ttl_override=30 * DAY)
        await asyncio.sleep(0.2)
        await cache.close_durable()

    _run(go())
    assert len(tier.write_calls) == 1
    batch = tier.write_calls[0]
    assert len(batch) == 50, "duplicate keys in one burst collapse to the latest value"
    assert pickle.loads(tier.rows["archive_month_0"][0]) == {"i": "again"}


def test_failed_write_is_logged_not_raised():
    tier = FakeTier(fail_writes=True)

    async def go():
        cache = TTLCache()
        await cache.attach_durable(tier, linger_seconds=0)
        cache.set("climatology", [1.0], ttl_override=30 * DAY)
        await asyncio.sleep(0.05)
        # Writer survives the failure and keeps serving later writes.
        tier.fail_writes = False
        cache.set("climatology2", [2.0], ttl_override=30 * DAY)
        await cache.close_durable()
        return cache.get("climatology")

    assert _run(go()) == [1.0], "the in-memory entry is unaffected by a failed write"
    assert "climatology2" in tier.rows


def test_attach_loads_unexpired_rows_into_memory():
    now = time.time()
    tier = FakeTier(rows={
        "clim": (pickle.dumps([1.0, 2.0]), now + 10 * DAY),
        "gone": (pickle.dumps("old"), now - 10 * DAY),
    })

    async def go():
        cache = TTLCache()
        loaded = await cache.attach_durable(tier, linger_seconds=0)
        await cache.close_durable()
        return cache, loaded

    cache, loaded = _run(go())
    assert loaded == 1
    assert cache.get("clim") == [1.0, 2.0]
    assert cache.get("gone") is None
    assert tier.loads == 1


def test_attach_keeps_stale_rows_within_grace_for_get_stale():
    """Frozen evenings are served via get_stale() — the grace window must survive restart."""
    now = time.time()
    tier = FakeTier(rows={"frozen": (pickle.dumps({"score": 71}), now - 3600)})

    async def go():
        cache = TTLCache(stale_grace_seconds=12 * 3600)
        await cache.attach_durable(tier, linger_seconds=0)
        await cache.close_durable()
        return cache

    cache = _run(go())
    assert cache.get("frozen") is None
    assert cache.get_stale("frozen") == {"score": 71}


def test_attach_keeps_later_expiry_when_pickle_also_has_key(tmp_path):
    path = str(tmp_path / "cache.pkl")
    local = TTLCache(persist_path=path)
    local.set("newer_local", "local", ttl_override=20 * DAY)
    local.set("newer_db", "local", ttl_override=2 * DAY)

    now = time.time()
    tier = FakeTier(rows={
        "newer_local": (pickle.dumps("db"), now + 5 * DAY),
        "newer_db": (pickle.dumps("db"), now + 25 * DAY),
    })

    async def go():
        cache = TTLCache(persist_path=path)
        await cache.attach_durable(tier, linger_seconds=0)
        await cache.close_durable()
        return cache

    cache = _run(go())
    assert cache.get("newer_local") == "local"
    assert cache.get("newer_db") == "db"


def test_undecodable_row_is_skipped():
    now = time.time()
    tier = FakeTier(rows={
        "bad": (b"not a pickle", now + DAY),
        "good": (pickle.dumps(1), now + DAY),
    })

    async def go():
        cache = TTLCache()
        loaded = await cache.attach_durable(tier, linger_seconds=0)
        await cache.close_durable()
        return cache, loaded

    cache, loaded = _run(go())
    assert loaded == 1 and cache.get("good") == 1 and cache.get("bad") is None


def test_attach_purges_expired_rows():
    tier = FakeTier(rows={"gone": (pickle.dumps(1), time.time() - 30 * DAY)})

    async def go():
        cache = TTLCache()
        await cache.attach_durable(tier, linger_seconds=0)
        await cache.close_durable()

    _run(go())
    assert tier.purges == 1 and tier.rows == {}


def test_database_outage_at_attach_leaves_cache_working():
    class DownTier(FakeTier):
        async def load_all(self, grace):
            raise ConnectionError("down")

    async def go():
        cache = TTLCache()
        loaded = await cache.attach_durable(DownTier(), linger_seconds=0)
        cache.set("x", 1, ttl_override=DAY)
        await cache.close_durable()
        return cache, loaded

    cache, loaded = _run(go())
    assert loaded == 0 and cache.get("x") == 1


def test_set_without_running_loop_does_not_crash():
    """Sync callers (scripts, tests) with an attached tier must not blow up."""
    tier = FakeTier()
    cache = TTLCache()
    _run(cache.attach_durable(tier, linger_seconds=0))
    cache.set("x", 1, ttl_override=DAY)  # no running loop here
    assert cache.get("x") == 1


def test_no_durable_tier_behaves_as_before():
    cache = TTLCache()
    cache.set("x", 1, ttl_override=30 * DAY)
    assert cache.get("x") == 1


def test_real_cached_value_types_round_trip_through_pickle(ideal_weather):
    snap = ideal_weather  # a real WeatherSnapshot (pydantic)
    values = [
        snap,
        [snap, snap],
        {date(2026, 9, 1): [(1.0, 2.0, 3.0)]},
        ([1.0, 2.0], None),
        {"hourly": {"time": ["2026-09-01T00:00"], "cloud_cover": [12]}},
    ]
    for v in values:
        assert pickle.loads(pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)) == v


# ---------------------------------------------------------------------------
# Real Postgres
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to run")
def test_postgres_cache_tier_contract():
    import asyncpg

    from app.services.subscription_store import _clean_dsn

    async def go():
        pool = await asyncpg.create_pool(_clean_dsn(os.environ["TEST_DATABASE_URL"]), min_size=1, max_size=2)
        try:
            tier = PostgresCacheTier(pool)
            await tier.ensure_schema()
            await tier.ensure_schema()  # idempotent
            await pool.execute("TRUNCATE cache_entries")
            now = time.time()
            await tier.write_many([
                ("live", b"\x00\x01", now + DAY),
                ("stale", b"s", now - 3600),
                ("dead", b"d", now - 10 * DAY),
            ])
            await tier.write_many([("live", b"\x02", now + 2 * DAY)])  # upsert
            rows = {k: (v, exp) for k, v, exp in await tier.load_all(grace=12 * 3600)}
            assert set(rows) == {"live", "stale"}
            assert rows["live"][0] == b"\x02"
            assert abs(rows["live"][1] - (now + 2 * DAY)) < 1
            await tier.purge_expired(grace=12 * 3600)
            left = await pool.fetch("SELECT key FROM cache_entries ORDER BY key")
            assert [r["key"] for r in left] == ["live", "stale"]

            # End to end through TTLCache: a restart sees the curve.
            a = TTLCache()
            await a.attach_durable(tier, linger_seconds=0)
            a.set("clim", [1.5, 2.5], ttl_override=30 * DAY)
            await a.close_durable()
            b = TTLCache()
            await b.attach_durable(tier, linger_seconds=0)
            await b.close_durable()
            assert b.get("clim") == [1.5, 2.5]
        finally:
            await pool.close()

    _run(go())


def test_expires_at_is_timezone_aware():
    from app.utils.durable_cache import _to_ts

    ts = _to_ts(0.0)
    assert ts == datetime(1970, 1, 1, tzinfo=timezone.utc)
