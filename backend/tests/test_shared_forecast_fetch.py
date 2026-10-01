"""One weather/aerosol/ensemble fetch per location per refresh, shared.

Tonight's /predict, another date's /predict and the 7-day /forecast used to
each fetch the same forecast hours for weather, aerosol and the ensemble.
They now slice one cached fetch, which returns the same values (same model,
same hours). The six-point corridor follows each date's own sunset azimuth:
exact for tonight and tomorrow, rounded to 1° after that so consecutive
evenings share one fetch — and /predict and /forecast share them all.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from app.core.config import Settings
from app.services.astronomy_service import AstronomyService
from app.services.weather_service import WeatherService
from app.utils.cache import TTLCache

UTC = timezone.utc
LAT, LON = 32.08, 34.78


def _hours(first: date, days: int) -> list[str]:
    start = datetime(first.year, first.month, first.day, tzinfo=UTC)
    return [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(days * 24)]


class CountingOpenMeteo:
    """Answers like Open-Meteo (forecast_days counts from today 00:00 UTC) and
    counts requests per endpoint; multi-coordinate requests count separately."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        p = request.url.params
        multi = "," in p["latitude"]
        kind = request.url.host.split(".")[0] + ("+multi" if multi else "")
        self.calls[kind] += 1
        today = datetime.now(UTC).date()
        past = int(p.get("past_days", 0))
        stamps = _hours(today - timedelta(days=past), int(p.get("forecast_days", 7)) + past)
        n = len(stamps)
        if request.url.host.startswith("ensemble"):
            hourly = {"time": stamps, **{f"cloud_cover_member{i:02d}": [float(i * 5)] * n for i in range(10)}}
        elif request.url.host.startswith("air-quality"):
            hourly = {"time": stamps, "aerosol_optical_depth": [0.2] * n, "dust": [5.0] * n}
        else:
            hourly = {
                "time": stamps,
                "cloud_cover_low": [10.0] * n, "cloud_cover_mid": [20.0] * n,
                "cloud_cover_high": [30.0] * n, "cloud_cover": [40.0] * n,
                "relative_humidity_2m": [50.0] * n, "dew_point_2m": [10.0] * n,
                "temperature_2m": [20.0] * n, "precipitation": [0.0] * n,
                "wind_speed_10m": [5.0] * n, "surface_pressure": [1013.0] * n,
            }
        body = {"hourly": hourly}
        return httpx.Response(200, json=[body] * len(p["latitude"].split(",")) if multi else body)


def _service(fake) -> WeatherService:
    return WeatherService(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake)),
        astro_service=AstronomyService(),
        cache=TTLCache(ttl_seconds=7200),
        settings=Settings(HTTP_MAX_RETRIES=0),
    )


async def _predict_reads(svc: WeatherService, d: date):
    """What PredictionService.predict asks the weather service for."""
    sunset = svc._astro.get_sunset_time(LAT, LON, d)
    snaps = await svc.get_window_snapshots(LAT, LON, d, sunset)
    corridor = await svc.get_corridor_samples(LAT, LON, d, sunset)
    spread = await svc.get_ensemble_cloud_spread(LAT, LON, d, sunset)
    return snaps, corridor, spread


async def _forecast_reads(svc: WeatherService, days: int = 7):
    """What PredictionService.forecast asks the weather service for."""
    windows = await svc.get_forecast_range_windows(LAT, LON, days=days)
    dates = [d for d, _ in windows]
    corridor = await svc.get_corridor_samples_map(LAT, LON, dates)
    spread = await svc.get_ensemble_cloud_spread_map(
        LAT, LON, [(d, svc._astro.get_sunset_time(LAT, LON, d)) for d in dates]
    )
    return windows, corridor, spread


@pytest.mark.asyncio
async def test_tonight_other_date_and_forecast_share_one_fetch_per_endpoint():
    fake = CountingOpenMeteo()
    svc = _service(fake)
    today = datetime.now(UTC).date()

    await _predict_reads(svc, today)
    await _predict_reads(svc, today + timedelta(days=3))
    windows, corridor, spread = await _forecast_reads(svc)

    # Corridor: tonight, tomorrow, then one per distinct rounded azimuth
    # (day+3's /predict fetch is one of those, reused by the forecast).
    astro = AstronomyService()
    buckets = {round(astro.get_sunset_azimuth(LAT, LON, today + timedelta(days=k))) for k in range(2, 7)}
    assert fake.calls == {
        "api": 1, "air-quality-api": 1, "ensemble-api": 1,
        "api+multi": 2 + len(buckets),
    }
    assert len(windows) == 7
    assert set(corridor) == {d for d, _ in windows}
    assert set(spread) == {d for d, _ in windows}


@pytest.mark.asyncio
async def test_predict_and_forecast_agree_on_tonight():
    """Both slice the same bundle, so /forecast's first day is exactly
    /predict's tonight (they used to come from separate fetches)."""
    today = datetime.now(UTC).date()
    shared = _service(CountingOpenMeteo())
    snaps, corridor, spread = await _predict_reads(shared, today)
    windows, corridor_map, spread_map = await _forecast_reads(shared)

    assert dict(windows)[today] == snaps
    assert spread_map[today] == spread
    assert len(corridor) == 6


@pytest.mark.asyncio
async def test_concurrent_cold_requests_make_one_fetch():
    fake = CountingOpenMeteo()
    svc = _service(fake)
    today = datetime.now(UTC).date()

    await asyncio.gather(*(_predict_reads(svc, today) for _ in range(10)))

    assert fake.calls == {
        "api": 1, "api+multi": 1, "air-quality-api": 1, "ensemble-api": 1,
    }


@pytest.mark.asyncio
async def test_date_past_the_bundle_falls_back_to_its_own_fetch():
    fake = CountingOpenMeteo()
    svc = _service(fake)
    far = datetime.now(UTC).date() + timedelta(days=9)

    snaps, corridor, _ = await _predict_reads(svc, far)

    assert snaps and corridor
    # Its own fetch (the `auto` model past icon_seamless), not the bundle.
    assert fake.calls["api"] == 1


@pytest.mark.asyncio
async def test_day_six_reads_the_icon_bundle():
    """Day 6's window ends inside the 7-day icon_seamless horizon, so it
    reads the bundle /forecast reads. (Its own fetch used to ask for 8 days,
    which fell back to `auto` and failed the 7-day aerosol API.)"""
    seen = []
    fake = CountingOpenMeteo()

    def spy(request):
        if request.url.host.startswith("api.") and "," not in request.url.params["latitude"]:
            seen.append(request.url.params.get("models"))
        return fake(request)

    svc = _service(spy)
    today = datetime.now(UTC).date()
    await _predict_reads(svc, today + timedelta(days=5))
    await _predict_reads(svc, today + timedelta(days=6))
    assert seen == ["icon_seamless"]
