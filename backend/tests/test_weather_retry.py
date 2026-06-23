"""Tests for Open-Meteo rate-limit (429) resilience in WeatherService._get_json.

Open-Meteo's free tier rate-limits per IP. A transient 429 must be retried
with backoff (honouring the Retry-After header) rather than surfaced as a
fatal error. A persistent 429 must raise WeatherUnavailableError, which the
API layer maps to a clean 503.
"""
from __future__ import annotations

from datetime import datetime, timezone

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

    async def fake_forecast(*args, **kwargs):
        fetches["n"] += 1
        return {"hourly": {}}

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
