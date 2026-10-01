"""Past days that can no longer change are kept until the next day settles.

The history page (heatmap) used to re-download the last week, the current
month's archive days and that month's corridor every CACHE_TTL (2 h) — about
20 weighted calls per location each time, for data that does not change
within a day. Data still open to revision (a day that ended less than
_PAST_SETTLED_AFTER ago, or today) keeps the default TTL.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import pytest

from app.services.weather_service import _PAST_SETTLED_AFTER, _settled_ttl
from app.utils.cache import TTLCache
from tests.test_day_windows_cache import LAT, LON, FakeOpenMeteo, _service

UTC = timezone.utc


def _later(cache: TTLCache, seconds: float) -> None:
    """Age the cache by *seconds*: drop whatever would have expired."""
    now = time.time() + seconds
    for key, (_, exp) in list(cache._store.items()):
        if exp < now:
            cache._drop(key)


def test_settled_ttl_only_for_days_that_can_no_longer_change():
    today = datetime.now(UTC).date()
    assert _settled_ttl(today) is None
    assert _settled_ttl(today + timedelta(days=3)) is None
    # Two days ago ended over a day ago: settled, kept until the next settle
    # point, which is at most a day away.
    ttl = _settled_ttl(today - timedelta(days=2))
    assert ttl is not None and 60 <= ttl <= 86_400
    # Yesterday is settled exactly once the margin has passed.
    settled = datetime.now(UTC) >= datetime(today.year, today.month, today.day, tzinfo=UTC) + _PAST_SETTLED_AFTER
    assert (_settled_ttl(today - timedelta(days=1)) is not None) == settled


@pytest.mark.asyncio
async def test_history_reload_after_the_default_ttl_downloads_nothing():
    """Ends two days ago (always settled, whatever the time of day) but still
    reaches into the recent week, which comes from the forecast API."""
    today = datetime.now(UTC).date()
    start, end = date(today.year, today.month, 1) - timedelta(days=40), today - timedelta(days=2)
    fake = FakeOpenMeteo()
    svc = _service(TTLCache(), fake)

    first = await svc.get_historical_range_windows(LAT, LON, start, end)
    corridor = await svc.get_corridor_samples_map(LAT, LON, [d for d, _ in first])
    assert fake.requests > 0
    before = fake.requests

    _later(svc._cache, svc._settings.CACHE_TTL_SECONDS + 300)
    again = await svc.get_historical_range_windows(LAT, LON, start, end)
    again_corridor = await svc.get_corridor_samples_map(LAT, LON, [d for d, _ in again])
    assert fake.requests == before, "settled past days must not be downloaded again"
    assert again == first and again_corridor == corridor


@pytest.mark.asyncio
async def test_failed_aerosol_is_not_kept_for_the_day():
    today = datetime.now(UTC).date()
    start, end = today - timedelta(days=6), today - timedelta(days=2)
    svc = _service(TTLCache(), FakeOpenMeteo(aq_down=True))
    first = await svc.get_historical_range_windows(LAT, LON, start, end)
    assert any(s.aerosol_is_estimated for _, snaps in first for s in snaps)

    svc._http = _service(TTLCache(), FakeOpenMeteo())._http   # the API recovers
    _later(svc._cache, svc._settings.CACHE_TTL_SECONDS + 300)
    again = await svc.get_historical_range_windows(LAT, LON, start, end)
    assert all(not s.aerosol_is_estimated for _, snaps in again for s in snaps)
