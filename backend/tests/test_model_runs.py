"""Model-run-aware caching: refetch when a model publishes, not on a timer."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.core.config import Settings
from app.services.model_runs import ModelRunClock
from tests.test_shared_forecast_fetch import LAT, LON, CountingOpenMeteo, _predict_reads, _service

UTC = timezone.utc
T0 = 1_790_000_000.0
EU_BBOX = "BBOX[29.5,-23.5,70.5,62.5]"
D2_BBOX = "BBOX[43.18,-3.94,58.08,20.339998]"
CAMS_EU_BBOX = "BBOX[71.95,-24.95,30.049995,44.95]"  # north-first, as served


class Clock:
    def __init__(self, t: float = T0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class MetaServer:
    def __init__(self) -> None:
        self.runs = {
            "dwd_icon_d2": (T0 - 3000, 10800, D2_BBOX),
            "dwd_icon_eu": (T0 - 2000, 10800, EU_BBOX),
            "dwd_icon": (T0 - 9000, 21600, ""),
            "dwd_icon_d2_eps": (T0 - 3000, 10800, D2_BBOX),
            "dwd_icon_eu_eps": (T0 - 4000, 21600, EU_BBOX),
            "dwd_icon_eps": (T0 - 20000, 43200, ""),
            "cams_europe": (T0 - 30000, 86400, CAMS_EU_BBOX),
            "cams_global": (T0 - 10000, 43200, ""),
        }
        self.reads: Counter[str] = Counter()
        self.down = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        model = request.url.path.split("/")[2]
        self.reads[model] += 1
        if self.down:
            return httpx.Response(503, json={"error": True})
        available, interval, bbox = self.runs[model]
        return httpx.Response(200, json={
            "last_run_availability_time": int(available),
            "update_interval_seconds": interval,
            "crs_wkt": f"GEOGCRS[...USAGE[SCOPE[\"grid\"],{bbox}]]" if bbox else "GEOGCRS[]",
        })


def _clock_and_server():
    clock, meta = Clock(), MetaServer()
    runs = ModelRunClock(httpx.AsyncClient(transport=httpx.MockTransport(meta)), Settings(), clock=clock)
    return clock, meta, runs


# ---------------------------------------------------------------------------
# ModelRunClock
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_latest_run_counts_only_models_covering_the_location():
    _, meta, runs = _clock_and_server()
    # Tel Aviv: ICON-EU and global cover it, ICON-D2 (central Europe) doesn't.
    assert await runs.latest_run("forecast", 32.08, 34.78) == T0 - 2000
    # Munich is inside ICON-D2's area, so a newer D2 run counts there.
    meta.runs["dwd_icon_d2"] = (T0 - 100, 10800, D2_BBOX)
    runs._last_attempt.clear()
    assert await runs.latest_run("forecast", 48.1, 11.6) == T0 - 100
    # CAMS Europe's bbox is served north-first and still parses.
    assert await runs.latest_run("aq", 32.08, 34.78) == T0 - 10000
    assert await runs.latest_run("aq", -33.9, 18.4) == T0 - 10000


@pytest.mark.asyncio
async def test_polls_lazily_more_often_once_a_run_is_due():
    clock, meta, runs = _clock_and_server()
    await runs.latest_run("forecast", 32.08, 34.78)
    assert meta.reads["dwd_icon_eu"] == 1 and meta.reads["dwd_icon_d2"] == 1

    clock.t += 600          # EU's next run is due at T0 - 2000 + 10800: not yet
    await runs.latest_run("forecast", 32.08, 34.78)
    assert meta.reads["dwd_icon_eu"] == 1
    assert meta.reads["dwd_icon_d2"] == 1, "known not to cover Tel Aviv: never re-read"

    clock.t = T0 - 2000 + 10800 - 300   # within 10 min of due
    await runs.latest_run("forecast", 32.08, 34.78)
    assert meta.reads["dwd_icon_eu"] == 2
    clock.t += 60                        # < 3 min later: no re-read
    await runs.latest_run("forecast", 32.08, 34.78)
    assert meta.reads["dwd_icon_eu"] == 2
    meta.runs["dwd_icon_eu"] = (clock.t, 10800, EU_BBOX)
    clock.t += 180
    assert await runs.latest_run("forecast", 32.08, 34.78) == clock.t - 180


@pytest.mark.asyncio
async def test_unknown_or_untrusted_metadata_means_none():
    clock, meta, runs = _clock_and_server()
    meta.down = True
    assert await runs.latest_run("forecast", 32.08, 34.78) is None

    meta.down = False
    clock.t += 200
    assert await runs.latest_run("forecast", 32.08, 34.78) == T0 - 2000
    meta.down = True
    clock.t += 3 * 3600   # reads keep failing: stop trusting what we had
    assert await runs.latest_run("forecast", 32.08, 34.78) is None


# ---------------------------------------------------------------------------
# WeatherService: refetch exactly when a covering model publishes
# ---------------------------------------------------------------------------

class FakeRuns:
    def __init__(self) -> None:
        now = datetime.now(UTC).timestamp()
        self.latest = {"forecast": now - 3600, "aq": now - 3600, "ensemble": now - 3600}

    async def latest_run(self, family, lat, lon):
        return self.latest[family]

    def publish(self, family: str) -> None:
        self.latest[family] = datetime.now(UTC).timestamp()


def _tracked_service(fake):
    svc = _service(fake)
    svc._runs = FakeRuns()
    return svc


@pytest.mark.asyncio
async def test_cached_until_a_new_run_then_refetched():
    fake = CountingOpenMeteo()
    svc = _tracked_service(fake)
    today = datetime.now(UTC).date()
    d = today + timedelta(days=1)

    await _predict_reads(svc, d)
    await _predict_reads(svc, d)
    first = dict(fake.calls)
    assert first == {"api": 1, "api+multi": 1, "air-quality-api": 1, "ensemble-api": 1}

    svc._runs.publish("ensemble")          # only the ensemble changed
    await _predict_reads(svc, d)
    assert fake.calls["ensemble-api"] == 2
    assert fake.calls["api"] == 1 and fake.calls["air-quality-api"] == 1

    svc._runs.publish("forecast")          # weather + corridor model changed
    await _predict_reads(svc, d)
    assert fake.calls["api"] == 2 and fake.calls["api+multi"] == 2
    assert fake.calls["air-quality-api"] == 1, "aerosol model didn't publish"


@pytest.mark.asyncio
async def test_without_run_tracking_the_old_ttl_applies():
    fake = CountingOpenMeteo()
    svc = _service(fake)   # no run clock: CACHE_TTL as before
    d = datetime.now(UTC).date() + timedelta(days=1)
    await _predict_reads(svc, d)
    await _predict_reads(svc, d)
    assert fake.calls["api"] == 1
    for key, (packed, exp) in list(svc._cache._store.items()):
        svc._cache._stored_at[key] -= svc._settings.CACHE_TTL_SECONDS + 1
    await _predict_reads(svc, d)
    assert fake.calls["api"] == 2


@pytest.mark.asyncio
async def test_tonight_stays_frozen_after_sunset_even_if_a_run_publishes():
    fake = CountingOpenMeteo()
    svc = _tracked_service(fake)
    today = datetime.now(UTC).date()
    ahead = datetime.now(UTC) + timedelta(hours=1)
    before = await svc.get_window_snapshots(LAT, LON, today, ahead)

    svc._runs.publish("forecast")
    over = datetime.now(UTC) - timedelta(hours=1)   # the window has ended
    after = await svc.get_window_snapshots(LAT, LON, today, over)
    assert after == before
    assert fake.calls["api"] == 1
