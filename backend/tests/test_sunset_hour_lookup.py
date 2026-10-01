"""Finding the hourly row nearest sunset.

The heatmap extracts 4 window snapshots for each of ~214 days from one merged
6-month hourly series (~4,800 rows). A linear scan per lookup, plus
re-parsing the air-quality timestamps on every call, cost ~10 s of CPU per
heatmap locally and ~50 s on Render's fractional CPU, blocking the event loop
the whole time.
"""
from __future__ import annotations

import random
import time
from datetime import date, datetime, timedelta, timezone

import httpx

from app.core.config import Settings
from app.services.astronomy_service import AstronomyService
from app.services.weather_service import (
    WeatherService,
    _nearest_hour_index,
    _prepopulate_parsed_times,
)
from app.utils.cache import TTLCache

UTC = timezone.utc
LAT, LON = 32.08, 34.78


def _brute_force(times, target):
    """The original lookup: the first index with the smallest distance."""
    return min(range(len(times)), key=lambda i: abs((times[i] - target).total_seconds()))


def _hours(start: datetime, n: int) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(n)]


def test_nearest_hour_index_matches_linear_scan():
    rng = random.Random(7)
    times = _hours(datetime(2026, 3, 1, tzinfo=UTC), 500)
    targets = [times[0] - timedelta(days=2), times[-1] + timedelta(days=2)]
    targets += [times[0] + timedelta(minutes=rng.randint(0, 500 * 60)) for _ in range(500)]
    targets += [t + timedelta(minutes=30) for t in times[:50]]  # exact ties
    targets += times[:50]                                       # exact hits
    for target in targets:
        assert _nearest_hour_index(times, target) == _brute_force(times, target), target


def test_nearest_hour_index_single_row():
    t = [datetime(2026, 3, 1, tzinfo=UTC)]
    assert _nearest_hour_index(t, t[0] + timedelta(days=9)) == 0


def _raw(start: datetime, n: int) -> tuple[dict, dict]:
    """Hourly weather + air quality where every value encodes its row index."""
    stamps = [t.strftime("%Y-%m-%dT%H:%M") for t in _hours(start, n)]
    weather = {"hourly": {
        "time": stamps,
        "cloud_cover_low": [i % 100 for i in range(n)],
        "cloud_cover_mid": [0.0] * n,
        "cloud_cover_high": [0.0] * n,
        "cloud_cover": [i % 100 for i in range(n)],
        "relative_humidity_2m": [50.0] * n,
        "dew_point_2m": [10.0] * n,
        "temperature_2m": [20.0] * n,
        "precipitation": [0.0] * n,
        "wind_speed_10m": [5.0] * n,
        "surface_pressure": [1013.0] * n,
    }}
    aq = {"hourly": {"time": list(stamps), "aerosol_optical_depth": [(i % 1000) / 1000 for i in range(n)]}}
    return weather, aq


def _service() -> WeatherService:
    return WeatherService(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
        astro_service=AstronomyService(),
        cache=TTLCache(ttl_seconds=900),
        settings=Settings(),
    )


def test_window_snapshots_read_the_nearest_rows():
    start = datetime(2026, 3, 1, tzinfo=UTC)
    weather, aq = _raw(start, 24 * 30)
    _prepopulate_parsed_times(weather)
    _prepopulate_parsed_times(aq)
    times = weather["hourly"]["_times_parsed"]
    svc, astro = _service(), AstronomyService()
    for day in range(1, 29):
        sunset = astro.get_sunset_time(LAT, LON, date(2026, 3, day))
        snaps = svc._extract_window_snapshots_from_raw(weather, aq, LAT, LON, sunset, "archive")
        for snap, offset in zip(snaps, (-15, 0, 15, 30)):
            idx = _brute_force(times, sunset + timedelta(minutes=offset))
            assert snap.cloud_low == idx % 100
            assert snap.aerosol_optical_depth == (idx % 1000) / 1000


def test_heatmap_scale_extraction_is_fast():
    """214 days × 4 window points over a ~6.5-month series. The linear scan
    took ~10 s here; the budget is generous so slow CI doesn't flake."""
    start = datetime(2026, 3, 1, tzinfo=UTC)
    weather, aq = _raw(start, 24 * 200)
    _prepopulate_parsed_times(weather)
    _prepopulate_parsed_times(aq)
    svc, astro = _service(), AstronomyService()
    t0 = time.process_time()
    for day in range(214):
        sunset = astro.get_sunset_time(LAT, LON, date(2026, 3, 1) + timedelta(days=day % 199))
        svc._extract_window_snapshots_from_raw(weather, aq, LAT, LON, sunset, "archive")
    assert time.process_time() - t0 < 1.5
