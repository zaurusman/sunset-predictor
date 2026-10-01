"""Tests for Open-Meteo rate-limit (429) resilience in WeatherService._get_json.

Open-Meteo's free tier rate-limits per IP. A transient 429 must be retried
with backoff (honouring the Retry-After header) rather than surfaced as a
fatal error. A persistent 429 must raise WeatherUnavailableError, which the
API layer maps to a clean 503.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.core.config import Settings

UTC = timezone.utc
from app.services.astronomy_service import AstronomyService
from app.services.weather_service import WeatherService, WeatherUnavailableError
from app.utils.cache import TTLCache


def _make_service(handler, **settings_overrides) -> WeatherService:
    """Build a WeatherService whose HTTP client is driven by a MockTransport."""
    defaults = dict(HTTP_MAX_RETRIES=3, HTTP_BACKOFF_BASE=0.0, HTTP_MAX_RETRY_DELAY=30.0)
    defaults.update(settings_overrides)
    settings = Settings(**defaults)
    return WeatherService(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        astro_service=AstronomyService(),
        cache=TTLCache(ttl_seconds=900),
        settings=settings,
    )


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds():
    """A 429 followed by a 200 should transparently succeed after retrying."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    svc = _make_service(handler)
    result = await svc._get_json("https://api.open-meteo.com/v1/forecast", {})

    assert result == {"ok": True}
    assert calls["n"] == 2  # one failure, one success


@pytest.mark.asyncio
async def test_persistent_429_raises_weather_unavailable():
    """Exhausting retries on 429 should raise WeatherUnavailableError."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429)

    svc = _make_service(handler, HTTP_MAX_RETRIES=2)

    with pytest.raises(WeatherUnavailableError):
        await svc._get_json("https://api.open-meteo.com/v1/forecast", {})

    assert calls["n"] == 3  # initial attempt + 2 retries


@pytest.mark.asyncio
async def test_respects_retry_after_header(monkeypatch):
    """The Retry-After header value should drive the sleep delay."""
    import app.services.weather_service as ws

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(ws.asyncio, "sleep", fake_sleep)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "5"})
        return httpx.Response(200, json={"ok": True})

    svc = _make_service(handler)
    await svc._get_json("https://api.open-meteo.com/v1/forecast", {})

    assert sleeps == [5.0]


@pytest.mark.asyncio
async def test_nearby_coordinates_reuse_one_fetch(monkeypatch):
    """Two lookups several km apart should share one cached fetch.

    Open-Meteo grid-snaps coordinates, so rounding them in the cache key turns
    nearby requests — different users, jittery geolocation — into cache hits,
    directly cutting call volume. With CACHE_COORD_DECIMALS=1 (~11 km cell),
    two points a few km apart collapse onto a single Open-Meteo call.
    """
    svc = _make_service(
        lambda request: httpx.Response(200, json={}), CACHE_COORD_DECIMALS=1
    )

    fetches = {"n": 0}

    start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    hours = [(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(7 * 24)]

    async def fake_forecast(*args, **kwargs):
        fetches["n"] += 1
        return {"hourly": {"time": hours}}

    async def fake_aq(*args, **kwargs):
        return None

    monkeypatch.setattr(svc, "_fetch_forecast_raw", fake_forecast)
    monkeypatch.setattr(svc, "_fetch_air_quality_raw", fake_aq)
    monkeypatch.setattr(svc, "_extract_window_snapshots_from_raw", lambda *a, **k: ["snap"])

    today = datetime.now(UTC).date()
    sunset = datetime(today.year, today.month, today.day, 17, 0, tzinfo=UTC)

    await svc.get_window_snapshots(32.11, 34.81, today, sunset)
    await svc.get_window_snapshots(32.14, 34.83, today, sunset)  # ~3.5 km away

    assert fetches["n"] == 1  # second lookup served entirely from cache


@pytest.mark.asyncio
async def test_does_not_retry_on_client_400():
    """A non-retryable 4xx (e.g. 400) should fail fast without retrying."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": True, "reason": "bad"})

    svc = _make_service(handler)

    with pytest.raises(httpx.HTTPStatusError):
        await svc._get_json("https://api.open-meteo.com/v1/forecast", {})

    assert calls["n"] == 1  # no retries


@pytest.mark.asyncio
async def test_failure_reason_from_open_meteo_is_surfaced(caplog):
    """Open-Meteo's `reason` says WHICH limit was hit (minutely/hourly/daily);
    it must reach the logs and the exception so a 503 can be diagnosed."""
    reason = "Daily API request limit exceeded. Please try again tomorrow."

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": True, "reason": reason})

    svc = _make_service(handler, HTTP_MAX_RETRIES=1)

    with caplog.at_level("WARNING"):
        with pytest.raises(WeatherUnavailableError, match="Daily API request limit"):
            await svc._get_json("https://api.open-meteo.com/v1/forecast", {})

    assert reason in caplog.text


def _stale_capable(svc: WeatherService) -> WeatherService:
    svc._cache = TTLCache(ttl_seconds=900, stale_grace_seconds=3600)
    return svc


@pytest.mark.asyncio
async def test_window_snapshots_fall_back_to_stale_when_provider_down(monkeypatch):
    """With Open-Meteo down, an expired-but-recent cached result is served
    instead of raising — and is not re-cached as if it were fresh."""
    svc = _stale_capable(_make_service(lambda request: httpx.Response(429)))

    async def down(*args, **kwargs):
        raise WeatherUnavailableError("Minutely API request limit exceeded")

    monkeypatch.setattr(svc, "_fetch_forecast_raw", down)

    # Sunset a few hours ahead of the real clock, so the viewing window is
    # never over — otherwise the after-sunset freeze re-pins the stale entry.
    now = datetime.now(UTC)
    today = now.date()
    sunset = now + timedelta(hours=3)
    key = TTLCache.make_key("window_snaps", *svc._ckey_coords(32.1, 34.8), str(today))
    svc._cache.set(key, ["old-snap"], ttl_override=-1)  # expired

    assert await svc.get_window_snapshots(32.1, 34.8, today, sunset) == ["old-snap"]
    assert svc._cache.get(key) is None  # still stale, not refreshed


@pytest.mark.asyncio
async def test_no_stale_data_still_raises(monkeypatch):
    svc = _stale_capable(_make_service(lambda request: httpx.Response(429)))

    async def down(*args, **kwargs):
        raise WeatherUnavailableError("down")

    monkeypatch.setattr(svc, "_fetch_forecast_raw", down)

    today = datetime.now(UTC).date()
    sunset = datetime(today.year, today.month, today.day, 17, 0, tzinfo=UTC)
    with pytest.raises(WeatherUnavailableError):
        await svc.get_window_snapshots(32.1, 34.8, today, sunset)


@pytest.mark.asyncio
async def test_stale_forecast_range_drops_days_already_past(monkeypatch):
    """A range cached before midnight must not show yesterday as day one."""
    from datetime import timedelta

    svc = _stale_capable(_make_service(lambda request: httpx.Response(429)))

    async def down(*args, **kwargs):
        raise WeatherUnavailableError("down")

    monkeypatch.setattr(svc, "_fetch_forecast_raw", down)

    today = datetime.now(UTC).date()
    yesterday = today - timedelta(days=1)
    key = TTLCache.make_key("forecast_range_windows", *svc._ckey_coords(32.1, 34.8), 3)
    svc._cache.set(key, [(yesterday, ["y"]), (today, ["t"])], ttl_override=-1)

    assert await svc.get_forecast_range_windows(32.1, 34.8, 3) == [(today, ["t"])]


# ---------------------------------------------------------------------------
# Shared archive months (heatmap ↔ climatology) and the after-sunset freeze
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overlapping_history_ranges_share_cached_archive_months(monkeypatch):
    """The climatology's few months must come out of the heatmap's cache (and
    vice versa) instead of being fetched again."""
    from datetime import timedelta

    svc = _make_service(lambda request: httpx.Response(200, json={}))
    fetched: list[tuple] = []

    async def fake_archive(lat, lon, start, end):
        fetched.append(("wx", start, end))
        return {"hourly": {"time": []}}

    async def fake_aq(lat, lon, start, end):
        fetched.append(("aq", start, end))
        return {"hourly": {"time": []}}

    monkeypatch.setattr(svc, "_fetch_archive_range_raw", fake_archive)
    monkeypatch.setattr(svc, "_fetch_air_quality_range_raw", fake_aq)

    today = datetime.now(UTC).date()
    boundary = today - timedelta(days=8)
    year_ago = today - timedelta(days=365)

    # "Heatmap": a year up to the archive boundary.
    await svc._archive_months_raw(32.1, 34.8, year_ago, boundary, boundary)
    n = len(fetched)
    # "Climatology": a slice from inside that year, touching at least half of
    # each month it spans — nothing new to fetch.
    m = (year_ago + timedelta(days=45)).replace(day=10)
    await svc._archive_months_raw(32.1, 34.8, m, m + timedelta(days=45), boundary)
    assert len(fetched) == n

    # A range that only clips a month (1 day) fetches just that day, not the
    # whole month — the shared-month rule must not make a small need expensive.
    fetched.clear()
    one = (year_ago + timedelta(days=200)).replace(day=28)
    await svc._archive_months_raw(32.5, 35.2, one, one, boundary)
    assert fetched == [("wx", one, one), ("aq", one, one)]


def test_merge_raw_concatenates_and_never_mutates_chunks():
    from app.services.weather_service import _merge_raw

    a = {"latitude": 1, "hourly": {"time": ["t1"], "x": [1]}}
    b = {"latitude": 1, "hourly": {"time": ["t2"], "x": [2], "_times_parsed": ["junk"]}}
    m = _merge_raw([a, b])
    assert m["hourly"] == {"time": ["t1", "t2"], "x": [1, 2]}
    m["hourly"]["time"].append("t3")
    assert a["hourly"]["time"] == ["t1"]


@pytest.mark.asyncio
async def test_tonight_is_frozen_once_the_viewing_window_is_over(monkeypatch):
    """After the window ends, an expired reading is kept (and re-pinned)
    rather than re-fetched, so tonight's answer doesn't drift."""
    from datetime import timedelta

    svc = _make_service(lambda request: httpx.Response(200, json={}))
    svc._cache = TTLCache(ttl_seconds=900, stale_grace_seconds=3600)
    fetches = {"n": 0}

    async def fake_forecast(*a, **k):
        fetches["n"] += 1
        return {"hourly": {}}

    monkeypatch.setattr(svc, "_fetch_forecast_raw", fake_forecast)
    monkeypatch.setattr(svc, "_fetch_air_quality_raw", fake_forecast)
    monkeypatch.setattr(svc, "_extract_window_snapshots_from_raw", lambda *a, **k: ["new"])

    today = datetime.now(UTC).date()
    sunset = datetime.now(UTC) - timedelta(hours=2)          # window long over
    key = TTLCache.make_key("window_snaps", *svc._ckey_coords(32.1, 34.8), str(today))
    svc._cache.set(key, ["before-sunset"], ttl_override=-1)   # expired

    assert await svc.get_window_snapshots(32.1, 34.8, today, sunset) == ["before-sunset"]
    assert fetches["n"] == 0
    assert svc._cache.get(key) == ["before-sunset"]           # pinned fresh again


@pytest.mark.asyncio
async def test_before_the_window_ends_an_expired_reading_is_refreshed(monkeypatch):
    from datetime import timedelta

    svc = _make_service(lambda request: httpx.Response(200, json={}))
    svc._cache = TTLCache(ttl_seconds=900, stale_grace_seconds=3600)

    async def fake_forecast(*a, **k):
        return {"hourly": {}}

    monkeypatch.setattr(svc, "_fetch_forecast_raw", fake_forecast)
    monkeypatch.setattr(svc, "_fetch_air_quality_raw", fake_forecast)
    monkeypatch.setattr(svc, "_extract_window_snapshots_from_raw", lambda *a, **k: ["new"])

    today = datetime.now(UTC).date()
    sunset = datetime.now(UTC) + timedelta(hours=2)          # still to come
    key = TTLCache.make_key("window_snaps", *svc._ckey_coords(32.1, 34.8), str(today))
    svc._cache.set(key, ["old"], ttl_override=-1)

    assert await svc.get_window_snapshots(32.1, 34.8, today, sunset) == ["new"]


@pytest.mark.asyncio
async def test_ensemble_spread_is_frozen_after_the_window(monkeypatch):
    from datetime import timedelta

    svc = _make_service(lambda request: httpx.Response(200, json={}))

    async def must_not_fetch(*a, **k):
        raise AssertionError("ensemble re-fetched after sunset")

    monkeypatch.setattr(svc, "_fetch_ensemble_raw", must_not_fetch)
    today = datetime.now(UTC).date()
    svc._cache.set(TTLCache.make_key("ensemble_day", *svc._ckey_coords(32.1, 34.8), str(today)), 12.5)

    sunset = datetime.now(UTC) - timedelta(hours=1)
    assert await svc.get_ensemble_cloud_spread(32.1, 34.8, today, sunset) == 12.5


# ---------------------------------------------------------------------------
# Exhausted quotas and the light corridor during an outage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [
    "Daily API request limit exceeded. Please try again tomorrow.",
    "Hourly API request limit exceeded. Please try again in the next hour.",
])
async def test_exhausted_quota_is_not_retried(reason):
    """A used-up daily/hourly quota won't recover within the backoff — retrying
    only adds seconds to every request (and calls to the count)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"error": True, "reason": reason})

    svc = _make_service(handler, HTTP_MAX_RETRIES=3)

    with pytest.raises(WeatherUnavailableError, match="limit exceeded"):
        await svc._get_json("https://api.open-meteo.com/v1/forecast", {})

    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_minutely_limit_is_still_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={
                "error": True, "reason": "Minutely API request limit exceeded. Please try again in one minute.",
            })
        return httpx.Response(200, json={"ok": True})

    svc = _make_service(handler)
    assert await svc._get_json("https://api.open-meteo.com/v1/forecast", {}) == {"ok": True}
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_corridor_falls_back_to_stale_when_provider_down(monkeypatch):
    """Dropping the corridor in an outage removes its penalty and inflates the
    score (seen live: 62.7 → 72.6). The last good samples must be used instead."""
    svc = _stale_capable(_make_service(lambda request: httpx.Response(429)))

    async def down(*args, **kwargs):
        raise WeatherUnavailableError("Daily API request limit exceeded")

    monkeypatch.setattr(svc, "_fetch_forecast_raw_multi", down)

    now = datetime.now(UTC)
    today = now.date()
    sunset = now + timedelta(hours=3)
    key = TTLCache.make_key("corridor", *svc._ckey_coords(32.1, 34.8), str(today))
    old = [(100.0, 80.0, 0.0)]
    svc._cache.set(key, old, ttl_override=-1)  # expired

    assert await svc.get_corridor_samples(32.1, 34.8, today, sunset) == old
    assert svc._cache.get(key) is None  # still stale, not refreshed


@pytest.mark.asyncio
async def test_corridor_map_falls_back_to_stale_when_provider_down(monkeypatch):
    svc = _stale_capable(_make_service(lambda request: httpx.Response(429)))

    async def down(*args, **kwargs):
        raise WeatherUnavailableError("Daily API request limit exceeded")

    monkeypatch.setattr(svc, "_fetch_corridor_month", down)

    today = datetime.now(UTC).date()
    tomorrow = today + timedelta(days=1)
    if tomorrow.month != today.month:  # keep both dates in one month's batch
        today, tomorrow = today - timedelta(days=1), today
    key = TTLCache.make_key(
        "corridor_month", *svc._ckey_coords(32.1, 34.8), today.year, today.month,
        str(today), str(tomorrow),
    )
    old = {today: [(100.0, 80.0, 0.0)], tomorrow: [(100.0, 10.0, 0.0)]}
    svc._cache.set(key, old, ttl_override=-1)  # expired

    assert await svc.get_corridor_samples_map(32.1, 34.8, [today, tomorrow]) == old
