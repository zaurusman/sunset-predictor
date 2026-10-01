"""Per-month caching of archive window snapshots.

The heatmap's per-day window snapshots used to be cached only as one block for
the whole requested range, with a 2 h TTL because the range ends yesterday.
After a restart every past day was re-extracted from the raw archive months.
Complete archive months never change, so their windows are now cached per
calendar month for 30 days, long enough for the durable tier to keep them.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from app.core.config import Settings
from app.services.astronomy_service import AstronomyService
from app.services.weather_service import WeatherService
from app.utils.cache import TTLCache
from app.utils.durable_cache import DURABLE_MIN_TTL_SECONDS

UTC = timezone.utc
LAT, LON = 32.08, 34.78


def _hourly(first: date, last: date) -> list[str]:
    start = datetime(first.year, first.month, first.day, tzinfo=UTC)
    n = ((last - first).days + 1) * 24
    return [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(n)]


def _weather(stamps):
    n = len(stamps)
    return {"hourly": {
        "time": stamps,
        "cloud_cover_low": [10.0] * n, "cloud_cover_mid": [20.0] * n,
        "cloud_cover_high": [30.0] * n, "cloud_cover": [40.0] * n,
        "relative_humidity_2m": [50.0] * n, "dew_point_2m": [10.0] * n,
        "temperature_2m": [20.0] * n, "precipitation": [0.0] * n,
        "wind_speed_10m": [5.0] * n, "surface_pressure": [1013.0] * n,
    }}


class FakeOpenMeteo:
    def __init__(self, aq_down: bool = False):
        self.aq_down = aq_down
        self.requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        p = request.url.params
        today = datetime.now(UTC).date()
        if "start_date" in p:
            first, last = date.fromisoformat(p["start_date"]), date.fromisoformat(p["end_date"])
        else:
            first = today - timedelta(days=int(p.get("past_days", 0)))
            last = today + timedelta(days=int(p.get("forecast_days", 1)))
        stamps = _hourly(first, last)
        if "air-quality" in request.url.host:
            if self.aq_down:
                return httpx.Response(400, json={"error": True, "reason": "down"})
            return httpx.Response(200, json={"hourly": {"time": stamps, "aerosol_optical_depth": [0.2] * len(stamps)}})
        return httpx.Response(200, json=_weather(stamps))


def _service(cache: TTLCache, handler) -> WeatherService:
    return WeatherService(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        astro_service=AstronomyService(),
        cache=cache,
        settings=Settings(HTTP_MAX_RETRIES=0),
    )


def _restarted(cache: TTLCache) -> TTLCache:
    """What a fresh process gets back from the durable tier: only entries
    whose TTL was at least a day (approximated by remaining lifetime)."""
    fresh = TTLCache()
    now = time.time()
    for k, (packed, exp) in cache._store.items():
        if exp - now >= 0.9 * DURABLE_MIN_TTL_SECONDS:
            fresh._store[k] = (packed, exp)
    return fresh


def _range():
    """Heatmap-shaped: the 1st of the month three months back .. yesterday."""
    today = datetime.now(UTC).date()
    m = today.month - 3
    y = today.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    return date(y, m, 1), today - timedelta(days=1)


def _complete_archive_days(start: date, end: date) -> set[date]:
    boundary = datetime.now(UTC).date() - timedelta(days=8)
    days, d = set(), start
    while d <= end:
        nxt = date(d.year + d.month // 12, d.month % 12 + 1, 1)
        last = nxt - timedelta(days=1)
        if d.day == 1 and last <= boundary and last <= end:
            days.update(d + timedelta(days=i) for i in range(last.day))
        d = nxt
    return days


def _count_extractions(monkeypatch, svc) -> list[date]:
    seen: list[date] = []
    orig = svc._extract_window_snapshots_from_raw

    def spy(weather, aq, lat, lon, sunset_time, data_source):
        seen.append(sunset_time.date())
        return orig(weather, aq, lat, lon, sunset_time, data_source)

    monkeypatch.setattr(svc, "_extract_window_snapshots_from_raw", spy)
    return seen


@pytest.mark.asyncio
async def test_restart_reuses_complete_archive_months(monkeypatch):
    start, end = _range()
    cold = _service(TTLCache(), FakeOpenMeteo())
    first = await cold.get_historical_range_windows(LAT, LON, start, end)

    warm = _service(_restarted(cold._cache), FakeOpenMeteo())
    seen = _count_extractions(monkeypatch, warm)
    second = await warm.get_historical_range_windows(LAT, LON, start, end)

    cached_days = _complete_archive_days(start, end)
    assert len(cached_days) >= 59, "the range must include two complete archive months"
    assert not cached_days & set(seen), "complete archive months must not be re-extracted"
    assert second == first


@pytest.mark.asyncio
async def test_month_with_estimated_aerosol_is_not_cached_long(monkeypatch):
    """A failed air-quality fetch falls back to the humidity proxy. Pinning that
    for 30 days would hide the measured AOD long after the API recovers."""
    start, end = _range()
    cold = _service(TTLCache(), FakeOpenMeteo(aq_down=True))
    await cold.get_historical_range_windows(LAT, LON, start, end)

    warm = _service(_restarted(cold._cache), FakeOpenMeteo())
    seen = _count_extractions(monkeypatch, warm)
    second = await warm.get_historical_range_windows(LAT, LON, start, end)
    assert _complete_archive_days(start, end) <= set(seen)
    assert all(not s.aerosol_is_estimated for _, snaps in second for s in snaps)


@pytest.mark.asyncio
async def test_partially_requested_month_is_not_cached(monkeypatch):
    start, end = _range()
    mid = start + timedelta(days=10)  # a month that's only partly in range
    cold = _service(TTLCache(), FakeOpenMeteo())
    await cold.get_historical_range_windows(LAT, LON, mid, mid + timedelta(days=5))

    warm = _service(_restarted(cold._cache), FakeOpenMeteo())
    seen = _count_extractions(monkeypatch, warm)
    await warm.get_historical_range_windows(LAT, LON, start, end)
    assert mid in seen, "a month built from a partial request must not have been cached"
