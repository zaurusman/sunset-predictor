"""A date means the location's local evening — everywhere.

AstronomyService used to return the sunset whose UTC date was the given date.
West of ~90°W the sunset falls after 00:00 UTC, so Honolulu's "2026-10-03"
was the evening of Oct 2 there, and /predict's default date (the local date)
showed the evening before. Israel's sunsets are on the same UTC date as its
local one, so nothing there may change: every check below that touches Israel
compares against the old UTC-date rule.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import httpx
import pytest
from astral import Observer
from astral.sun import sun

from app.core.config import Settings
from app.services.astronomy_service import AstronomyService
from app.services.weather_service import WeatherService
from app.utils.cache import TTLCache
from app.utils.time_utils import (
    first_forecast_date,
    get_timezone_for_coordinates,
    local_sunset_date,
    tonight_dates,
)

UTC = timezone.utc
HONOLULU = (21.307, -157.858)
CHICAGO = (41.88, -87.63)
TEL_AVIV = (32.08, 34.78)
ISRAEL = [TEL_AVIV, (29.56, 34.95), (32.79, 34.99), (31.77, 35.21), (33.2, 35.57)]

astro = AstronomyService()


def _local_date(lat: float, lon: float, at: datetime) -> date:
    return at.astimezone(get_timezone_for_coordinates(lat, lon)).date()


def test_honolulu_date_is_its_own_evening():
    """The prod report: "2026-10-03" came back as 2026-10-03T04:17Z, which is
    the evening of Oct 2 in Honolulu."""
    sunset = astro.get_sunset_time(*HONOLULU, date(2026, 10, 2))
    assert sunset.strftime("%Y-%m-%dT%H:%M") == "2026-10-03T04:17"
    assert _local_date(*HONOLULU, sunset) == date(2026, 10, 2)


@pytest.mark.parametrize("lat,lon", [HONOLULU, CHICAGO, (34.05, -118.25), (-13.83, -171.76), (40.71, -74.0)])
def test_every_date_is_one_local_evening_a_day_apart(lat, lon):
    """Near 90°W the sunset crosses 00:00 UTC twice a year: under the UTC rule
    one date had no sunset (a made-up 18:00 UTC) and another had two."""
    start = date(2026, 1, 1)
    prev = None
    for i in range(366):
        d = start + timedelta(days=i)
        st = astro.get_sunset_time(lat, lon, d)
        assert _local_date(lat, lon, st) == d, d
        if prev is not None:
            assert timedelta(hours=23) < st - prev < timedelta(hours=25), d
        prev = st


@pytest.mark.parametrize("lat,lon", ISRAEL)
def test_israel_sunsets_are_exactly_what_they_were(lat, lon):
    for i in range(3 * 366):
        d = date(2025, 1, 1) + timedelta(days=i)
        old = sun(Observer(lat, lon), date=d, tzinfo=UTC)
        new = astro.get_sun_times(lat, lon, d)
        assert new["sunset"] == old["sunset"], d
        assert new["sunset"].utcoffset() == timedelta(0)


def test_tonight_includes_the_local_date_not_the_utc_date_out_west():
    # Some date in tonight's set is always the local date.
    assert local_sunset_date(*HONOLULU) in tonight_dates(*HONOLULU)
    assert local_sunset_date(*TEL_AVIV) in tonight_dates(*TEL_AVIV)
    # Never more than a day either side.
    for lat, lon in (HONOLULU, TEL_AVIV):
        today = local_sunset_date(lat, lon)
        assert all(abs((d - today).days) <= 1 for d in tonight_dates(lat, lon))


def test_first_forecast_date():
    utc_today = datetime.now(UTC).date()
    # Israel: always the UTC date (as /forecast always started), which is the
    # local date except after local midnight.
    assert first_forecast_date(*TEL_AVIV) == utc_today
    # Out west: the local date — tonight there, never tomorrow.
    assert first_forecast_date(*HONOLULU) == local_sunset_date(*HONOLULU)


# ----------------------------------------------------------------------
# The weather service: fetches follow the evening's sunset instant
# ----------------------------------------------------------------------


class FakeOpenMeteo:
    """Hourly values that encode their own timestamp, for any request shape."""

    def __init__(self):
        self.ranges: list[tuple[date, date]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        p = request.url.params
        today = datetime.now(UTC).date()
        if "start_date" in p:
            s, e = date.fromisoformat(p["start_date"]), date.fromisoformat(p["end_date"])
            self.ranges.append((s, e))
            start, n = datetime(s.year, s.month, s.day, tzinfo=UTC), ((e - s).days + 1) * 24
        else:
            past, days = int(p.get("past_days", 0)), int(p.get("forecast_days", 7))
            start = datetime(today.year, today.month, today.day, tzinfo=UTC) - timedelta(days=past)
            n = (past + days) * 24
        stamps = [start + timedelta(hours=h) for h in range(n)]
        # cloud_cover_high = the hour of day, so a snapshot says which hour it read
        def entry() -> dict:
            if "air-quality" in request.url.host:
                return {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in stamps],
                                   "aerosol_optical_depth": [0.1] * n}}
            if "ensemble" in request.url.host:
                h = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in stamps]}
                for i in range(6):
                    h[f"cloud_cover_member{i:02d}"] = [float(t.hour + i) for t in stamps]
                return {"hourly": h}
            return {"hourly": {
                "time": [t.strftime("%Y-%m-%dT%H:%M") for t in stamps],
                "cloud_cover_low": [float(t.day) for t in stamps],
                "cloud_cover_mid": [float(t.month) for t in stamps],
                "cloud_cover_high": [float(t.hour) for t in stamps],
                "cloud_cover": [50.0] * n, "relative_humidity_2m": [50.0] * n,
                "dew_point_2m": [12.0] * n, "temperature_2m": [24.0] * n,
                "precipitation": [0.0] * n, "wind_speed_10m": [8.0] * n,
                "surface_pressure": [1010.0] * n, "visibility": [20000.0] * n,
            }}
        lats = p["latitude"].split(",")
        return httpx.Response(200, json=entry() if len(lats) == 1 else [entry() for _ in lats])


def _service(fake: FakeOpenMeteo) -> WeatherService:
    return WeatherService(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake)),
        astro_service=AstronomyService(),
        cache=TTLCache(),
        settings=Settings(HTTP_MAX_RETRIES=0),
    )


def _read_at(snap) -> tuple[int, int, int]:
    """(month, day, hour) in UTC that a FakeOpenMeteo snapshot was read from."""
    return int(snap.cloud_mid), int(snap.cloud_low), int(snap.cloud_high)


def _expected(sunset: datetime) -> tuple[int, int, int]:
    """The hour nearest the sunset (what the "sunset" snapshot reads)."""
    t = (sunset + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    return t.month, t.day, t.hour


@pytest.mark.parametrize("back", [0, 3, 20])
@pytest.mark.asyncio
async def test_west_window_reads_its_own_sunset_hour(back):
    """Tonight, a recent evening (forecast + past_days) and an archive one."""
    d = local_sunset_date(*HONOLULU) - timedelta(days=back)
    sunset = astro.get_sunset_time(*HONOLULU, d)
    snaps = await _service(FakeOpenMeteo()).get_window_snapshots(*HONOLULU, d, sunset)
    sunset_snap = next(s for s in snaps if s.timestamp_label == "sunset")
    assert _read_at(sunset_snap) == _expected(sunset), (d, sunset)


@pytest.mark.asyncio
async def test_west_history_fetches_cover_every_evening():
    """Archive requests run in UTC days: the last local evening of a range
    sets on the next UTC day out west, which the fetch has to include."""
    fake = FakeOpenMeteo()
    svc = _service(fake)
    end = local_sunset_date(*HONOLULU) - timedelta(days=1)
    start = end - timedelta(days=40)
    windows = dict(await svc.get_historical_range_windows(*HONOLULU, start, end))
    assert sorted(windows) == [start + timedelta(days=i) for i in range(41)]
    for d, snaps in windows.items():
        sunset = astro.get_sunset_time(*HONOLULU, d)
        sunset_snap = next(s for s in snaps if s.timestamp_label == "sunset")
        assert _read_at(sunset_snap) == _expected(sunset), d


@pytest.mark.asyncio
async def test_west_forecast_starts_tonight_and_matches_predict():
    fake = FakeOpenMeteo()
    svc = _service(fake)
    windows = await svc.get_forecast_range_windows(*HONOLULU, 7)
    assert [d for d, _ in windows] == [
        local_sunset_date(*HONOLULU) + timedelta(days=i) for i in range(7)
    ]
    other = _service(FakeOpenMeteo())
    for d, snaps in windows:
        sunset = astro.get_sunset_time(*HONOLULU, d)
        own = await other.get_window_snapshots(*HONOLULU, d, sunset)
        assert [s.model_dump() for s in own] == [s.model_dump() for s in snaps], d
        assert _read_at(next(s for s in snaps if s.timestamp_label == "sunset")) == _expected(sunset)


@pytest.mark.asyncio
async def test_cache_entries_from_the_utc_rule_are_not_read_as_another_evening():
    """Entries keyed by the old rule survive a deploy in the durable tier. In
    Israel they name the same evening and must still be found (a frozen
    reading must not move); out west they named the evening before."""
    svc = _service(FakeOpenMeteo())
    ta = local_sunset_date(*TEL_AVIV)
    assert svc._evening_key(*TEL_AVIV, ta) == str(ta)
    hn = local_sunset_date(*HONOLULU)
    assert svc._evening_key(*HONOLULU, hn) != str(hn)
