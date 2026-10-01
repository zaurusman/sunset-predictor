"""/predict and /forecast must read the same data for the same evening.

They used to disagree by up to 15 points: /predict day+6 asked the air-quality
API for 8 days (rejected, so the aerosol fell back to the humidity proxy) and
the weather API for 8 (past icon_seamless's horizon, so `auto`); /forecast's
corridor asked for 8 days (`auto` again) along one azimuth per month.

The fake below makes every reading depend on the model, the coordinates and
the hour, and rejects aerosol requests beyond 7 days as the real API does, so
any difference in model, geometry or hours shows up as a different snapshot.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from app.core.config import Settings
from app.services.astronomy_service import AstronomyService
from app.services.weather_service import (
    ICON_SEAMLESS_MAX_DAYS,
    WeatherService,
    _forecast_fetch_days,
)
from app.utils.cache import TTLCache

UTC = timezone.utc
TEL_AVIV = (32.08, 34.78)
LOS_ANGELES = (34.05, -118.25)


class FakeOpenMeteo:
    def __init__(self):
        self.aq_days: list[int] = []
        self.weather_models: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        p = request.url.params
        days = int(p.get("forecast_days", 1))
        today = datetime.now(UTC).date()
        start = datetime(today.year, today.month, today.day, tzinfo=UTC)
        stamps = [start + timedelta(hours=h) for h in range(days * 24)]
        times = [t.strftime("%Y-%m-%dT%H:%M") for t in stamps]

        if "air-quality" in request.url.host:
            self.aq_days.append(days)
            if days > 7:
                return httpx.Response(400, json={"error": True, "reason": "Forecast days is invalid."})
            return httpx.Response(200, json={"hourly": {
                "time": times,
                "aerosol_optical_depth": [0.05 + (t.hour % 7) / 50 for t in stamps],
            }})

        model = p.get("models", "auto")
        bump = 0.0 if model == "auto" else 30.0

        def series(lat: float, scale: float) -> list[float]:
            return [
                min(100.0, bump + (t.hour * scale + lat * 7 + t.day) % 60)
                for t in stamps
            ]

        def entry(lat: float) -> dict:
            n = len(stamps)
            return {"hourly": {
                "time": times,
                "cloud_cover_low": series(lat, 1.0),
                "cloud_cover_mid": series(lat, 2.0),
                "cloud_cover_high": series(lat, 3.0),
                "cloud_cover": series(lat, 1.5),
                "relative_humidity_2m": series(lat, 0.5),
                "dew_point_2m": [12.0] * n, "temperature_2m": [24.0] * n,
                "precipitation": [0.0] * n, "wind_speed_10m": [8.0] * n,
                "surface_pressure": [1010.0 + t.hour / 10 for t in stamps],
            }}

        lats = [float(x) for x in p["latitude"].split(",")]
        if len(lats) == 1:
            self.weather_models.append(model)
            return httpx.Response(200, json=entry(lats[0]))
        return httpx.Response(200, json=[entry(x) for x in lats])


def _service(fake: FakeOpenMeteo) -> WeatherService:
    return WeatherService(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake)),
        astro_service=AstronomyService(),
        cache=TTLCache(),
        settings=Settings(HTTP_MAX_RETRIES=0),
    )


def _dump(snaps) -> list[dict]:
    return [s.model_dump() for s in snaps]


@pytest.mark.parametrize("lat,lon", [TEL_AVIV, LOS_ANGELES])
@pytest.mark.asyncio
async def test_every_forecast_day_matches_its_own_prediction(lat, lon):
    forecast_side, predict_side = FakeOpenMeteo(), FakeOpenMeteo()
    forecast_svc, predict_svc = _service(forecast_side), _service(predict_side)
    astro = AstronomyService()

    windows = dict(await forecast_svc.get_forecast_range_windows(lat, lon, 7))
    corridors = await forecast_svc.get_corridor_samples_map(lat, lon, sorted(windows))
    assert len(windows) == 7

    for d, forecast_snaps in sorted(windows.items()):
        sunset = astro.get_sunset_time(lat, lon, d)
        predict_snaps = await predict_svc.get_window_snapshots(lat, lon, d, sunset)
        assert _dump(predict_snaps) == _dump(forecast_snaps), d
        assert all(not s.aerosol_is_estimated for s in predict_snaps), d
        predict_corridor = await predict_svc.get_corridor_samples(lat, lon, d, sunset)
        assert predict_corridor and predict_corridor == corridors[d], d

    for side in (forecast_side, predict_side):
        assert max(side.aq_days) <= 7


@pytest.mark.asyncio
async def test_day6_in_israel_reads_icon_seamless():
    """Day+6's window ends well before 00:00 UTC in Israel, so it is inside
    the horizon /forecast fetches with icon_seamless — no `auto` fallback."""
    fake = FakeOpenMeteo()
    svc = _service(fake)
    d = datetime.now(UTC).date() + timedelta(days=6)
    await svc.get_window_snapshots(*TEL_AVIV, d, AstronomyService().get_sunset_time(*TEL_AVIV, d))
    assert fake.weather_models == ["icon_seamless"]
    assert fake.aq_days == [7]


def test_fetch_days_cover_the_window_and_stay_on_icon_when_they_can():
    today = date(2026, 10, 1)
    at = lambda days, hh, mm: datetime(2026, 10, 1, hh, mm, tzinfo=UTC) + timedelta(days=days)
    # Tonight and day+6, window inside the UTC day: the icon horizon.
    assert _forecast_fetch_days(at(0, 15, 30), today) == ICON_SEAMLESS_MAX_DAYS
    assert _forecast_fetch_days(at(6, 15, 30), today) == ICON_SEAMLESS_MAX_DAYS
    # A window that runs past 00:00 UTC needs the next day as well: still
    # icon up to day+5, `auto` with 8 days for day+6.
    assert _forecast_fetch_days(at(5, 23, 45), today) == ICON_SEAMLESS_MAX_DAYS
    assert _forecast_fetch_days(at(6, 23, 45), today) == ICON_SEAMLESS_MAX_DAYS + 1


@pytest.mark.asyncio
async def test_tonight_and_tomorrow_corridor_follow_the_exact_azimuth():
    """Rounding the azimuth (to share fetches) starts at day+2: tonight's
    corridor points are exactly where they were before the change."""
    from app.services.weather_service import CORRIDOR_DISTANCES_KM
    from app.utils.geo import destination_point

    seen: list[str] = []
    fake = FakeOpenMeteo()

    def handler(request: httpx.Request) -> httpx.Response:
        if "," in request.url.params.get("latitude", ""):
            seen.append(request.url.params["latitude"])
        return fake(request)

    svc = WeatherService(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        astro_service=AstronomyService(), cache=TTLCache(),
        settings=Settings(HTTP_MAX_RETRIES=0),
    )
    astro = AstronomyService()
    today = datetime.now(UTC).date()
    for k in (0, 1):
        d = today + timedelta(days=k)
        await svc.get_corridor_samples(*TEL_AVIV, d, astro.get_sunset_time(*TEL_AVIV, d))
        az = astro.get_sunset_azimuth(*TEL_AVIV, d)
        exact = ",".join(
            f"{destination_point(*TEL_AVIV, az, km)[0]:.4f}" for km in CORRIDOR_DISTANCES_KM
        )
        assert seen[-1] == exact
