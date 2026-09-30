from __future__ import annotations

import asyncio
import os
from datetime import date

import pytest

from app.services.subscription_store import (
    InMemorySubscriptionStore,
    PostgresSubscriptionStore,
    _clean_dsn,
    cell_key,
)

TLV = {"latitude": 32.0853, "longitude": 34.7818, "name": "Tel Aviv"}


def test_cell_key_rounds_to_tenth_degree():
    assert cell_key(32.0853, 34.7818) == "32.1,34.8"
    assert cell_key(-0.04, -0.06) == "-0.0,-0.1"


def test_clean_dsn_strips_channel_binding():
    dsn = "postgresql://u:p@h/db?sslmode=require&channel_binding=require"
    assert _clean_dsn(dsn) == "postgresql://u:p@h/db?sslmode=require"


async def _exercise(store) -> None:
    await store.upsert("https://push/a", "k1", "a1", [TLV], "Asia/Jerusalem")
    await store.upsert("https://push/a", "k2", "a2", [TLV], "Asia/Jerusalem")  # update, not dup
    subs = await store.all()
    assert len(subs) == 1 and subs[0].p256dh == "k2" and subs[0].places == [TLV]

    day = date(2026, 9, 30)
    await store.mark_notified("https://push/a", "32.1,34.8", day)
    await store.upsert("https://push/a", "k2", "a2", [TLV], "Asia/Jerusalem")
    (sub,) = await store.all()
    assert sub.last_notified == {"32.1,34.8": "2026-09-30"}, "upsert must keep dedupe state"

    assert not await store.cell_checked("32.1,34.8", day)
    await store.record_cell_check("32.1,34.8", day, 42.0)
    await store.record_cell_check("32.1,34.8", day, 43.0)  # idempotent
    assert await store.cell_checked("32.1,34.8", day)

    await store.delete("https://push/a")
    assert await store.all() == []


def test_in_memory_store_contract():
    asyncio.run(_exercise(InMemorySubscriptionStore()))


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to run")
def test_postgres_store_contract():
    async def go():
        store = await PostgresSubscriptionStore.connect(os.environ["TEST_DATABASE_URL"])
        try:
            await store._pool.execute("TRUNCATE push_subscriptions, alert_cell_checks")
            await _exercise(store)
        finally:
            await store.close()
    asyncio.run(go())
