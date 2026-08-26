"""Tests that WeatherSnapshot.data_source reports the endpoint that actually
produced it.

Regression: the field used to be inferred by checking whether the string
"archive" appeared inside `generationtime_ms` (a float, e.g. 0.0159) — a
check that can never be true. Every snapshot silently reported "forecast",
including ones built from the archive API.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from app.core.config import Settings
from app.services.astronomy_service import AstronomyService
from app.services.weather_service import WeatherService
from app.utils.cache import TTLCache

UTC = timezone.utc


def _make_service() -> WeatherService:
    return WeatherService(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={})
        )),
        astro_service=AstronomyService(),
        cache=TTLCache(ttl_seconds=900),
        settings=Settings(),
    )


def _minimal_weather(hour: datetime) -> dict:
    return {"hourly": {"time": [hour.strftime("%Y-%m-%dT%H:%M")]}}


@pytest.mark.asyncio
async def test_data_source_is_archive_for_old_dates(monkeypatch):
    """A date well beyond the forecast endpoint's past_days window is fetched
    from the archive API, and every window snapshot must say so."""
    svc = _make_service()
    target_date = datetime.now(UTC).date() - timedelta(days=30)
    sunset_time = datetime(target_date.year, target_date.month, target_date.day, 17, 0, tzinfo=UTC)

    async def fake_archive_raw(*args, **kwargs):
        return _minimal_weather(sunset_time)

    async def fake_aq_range(*args, **kwargs):
        return None

    monkeypatch.setattr(svc, "_fetch_archive_raw", fake_archive_raw)
    monkeypatch.setattr(svc, "_fetch_air_quality_range_raw", fake_aq_range)

    snaps = await svc.get_window_snapshots(32.08, 34.78, target_date, sunset_time)

    assert snaps
    assert all(s.data_source == "archive" for s in snaps)


@pytest.mark.asyncio
async def test_data_source_is_forecast_for_recent_dates(monkeypatch):
    """A date within the forecast endpoint's past_days window (<=7 days ago)
    is fetched from the forecast API, not the archive."""
    svc = _make_service()
    target_date = datetime.now(UTC).date() - timedelta(days=2)
    sunset_time = datetime(target_date.year, target_date.month, target_date.day, 17, 0, tzinfo=UTC)

    async def fake_forecast_raw(*args, **kwargs):
        return _minimal_weather(sunset_time)

    async def fake_aq_raw(*args, **kwargs):
        return None

    monkeypatch.setattr(svc, "_fetch_forecast_raw", fake_forecast_raw)
    monkeypatch.setattr(svc, "_fetch_air_quality_raw", fake_aq_raw)

    snaps = await svc.get_window_snapshots(32.08, 34.78, target_date, sunset_time)

    assert snaps
    assert all(s.data_source == "forecast" for s in snaps)
