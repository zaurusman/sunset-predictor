"""When Open-Meteo refuses the server, the browser fetches for it.

The server answers 503 with `client_fetch` (the exact URLs it still needs);
the browser fetches them and repeats the request with `client_data`. The
scores must be exactly what the server computes when it can download the data
itself — for tonight's /predict and for the 7-day /forecast.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.utils.call_budget import CallBudget
from app.utils.client_fetch import ClientFetchNeeded, client_data
from tests.test_shared_forecast_fetch import LAT, LON, CountingOpenMeteo, _service

DROP = {"requested_at", "generated_at"}


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in DROP}


def _browser_loop(client, fake, path: str, body: dict):
    """What frontend/src/lib/api.ts does. Returns (response, rounds)."""
    data: dict = {}
    rounds: list[list[str]] = []
    for _ in range(10):
        r = client.post(path, json={**body, **({"client_data": data} if data else {})})
        if r.status_code != 503 or "client_fetch" not in r.json():
            return r, rounds
        urls = r.json()["client_fetch"]
        assert urls and not any(u in data for u in urls)   # never asks twice
        rounds.append([httpx.URL(u).host for u in urls])
        for u in urls:
            data[u] = fake(httpx.Request("GET", u)).json()
    raise AssertionError("too many rounds")


@pytest.fixture()
def wired():
    fake = CountingOpenMeteo()
    with TestClient(app) as client:
        ps = app.state.prediction_service
        ws = ps._weather
        saved = (ws._http, ws._runs, ws.budget, ps._climatology, ws._settings.OPEN_METEO_SIMULATE_DAILY_LIMIT)
        ws._http = httpx.AsyncClient(transport=httpx.MockTransport(fake))
        ws._runs = None
        ws.budget = CallBudget(client_hourly_limit=0, daily_soft_cap=0)
        ps._climatology = None
        try:
            yield client, ws, fake
        finally:
            ws._http, ws._runs, ws.budget, ps._climatology, ws._settings.OPEN_METEO_SIMULATE_DAILY_LIMIT = saved
            ws._cache.clear()


@pytest.mark.parametrize("path,body,max_rounds", [
    ("/predict", {"latitude": LAT, "longitude": LON}, 3),
    ("/forecast", {"latitude": LAT, "longitude": LON, "days": 7}, 2),
])
def test_browser_fetch_gives_the_same_scores_as_a_server_fetch(wired, path, body, max_rounds):
    client, ws, fake = wired
    normal = client.post(path, json=body)          # the server downloads everything
    assert normal.status_code == 200
    ws._cache.clear()

    ws._settings.OPEN_METEO_SIMULATE_DAILY_LIMIT = True   # Open-Meteo refuses the server
    r, rounds = _browser_loop(client, fake, path, body)
    assert r.status_code == 200, r.text
    assert _clean(r.json()) == _clean(normal.json())
    if path == "/forecast":
        assert [d["beauty_score_0_100"] for d in r.json()["days"]] == [
            d["beauty_score_0_100"] for d in normal.json()["days"]
        ]
        assert len(r.json()["days"]) == 7
    # Independent fetches are asked for together: weather + aerosol first.
    assert len(rounds) <= max_rounds
    assert sorted(rounds[0]) == ["air-quality-api.open-meteo.com", "api.open-meteo.com"]
    # Nothing the browser sent was written to the shared cache.
    assert ws._cache.size() == 0


def test_forecast_hands_off_when_our_own_share_is_used_up(wired):
    """Open-Meteo still accepts the server, but the share kept for tonight
    must not be spent on the 7-day page: the browser fetches instead of the
    page failing as "busy" — and tonight is still fetched by the server."""
    client, ws, fake = wired
    ws.budget.note_rate_limited("Hourly API request limit exceeded.")
    week = {"latitude": LAT, "longitude": LON, "days": 7}
    r, rounds = _browser_loop(client, fake, "/forecast", week)
    assert r.status_code == 200 and rounds
    assert all("ensemble" not in h for rnd in rounds for h in rnd)  # tonight's data: server
    tonight = client.post("/predict", json={"latitude": LAT, "longitude": LON})
    assert tonight.status_code == 200


def test_heatmap_never_hands_fetches_to_the_browser(wired):
    client, ws, fake = wired
    ws._settings.OPEN_METEO_SIMULATE_DAILY_LIMIT = True
    r = client.get("/heatmap", params={"lat": LAT, "lon": LON, "months": 1})
    assert r.status_code == 503 and "client_fetch" not in r.json()


def test_another_date_never_hands_fetches_to_the_browser(wired):
    client, ws, fake = wired
    ws._settings.OPEN_METEO_SIMULATE_DAILY_LIMIT = True
    d3 = (datetime.now(timezone.utc).date() + timedelta(days=3)).isoformat()
    r = client.post("/predict", json={"latitude": LAT, "longitude": LON, "target_date": d3,
                                      "client_data": {"https://api.open-meteo.com/x": {}}})
    assert r.status_code == 503 and "client_fetch" not in r.json()


def test_tonight_by_the_utc_date_hands_off_too(wired, monkeypatch):
    """Between local midnight and 00:00 UTC (00:00-03:00 in Israel) the app's
    date and the location's date differ by a day. Either one is tonight: a
    7-day page asks for the UTC date, and it must not fail as "another date"."""
    client, ws, fake = wired
    today = datetime.now(timezone.utc).date()
    late = datetime(today.year, today.month, today.day, 23, 30, tzinfo=timezone.utc)

    class LateEvening(datetime):
        @classmethod
        def now(cls, tz=None):
            return late.astimezone(tz) if tz else late.replace(tzinfo=None)

    # Only the "what is tonight" clock: 01:30 the next day in Israel.
    monkeypatch.setattr("app.utils.time_utils.datetime", LateEvening)
    ws._settings.OPEN_METEO_SIMULATE_DAILY_LIMIT = True
    body = {"latitude": LAT, "longitude": LON, "target_date": today.isoformat()}
    r, rounds = _browser_loop(client, fake, "/predict", body)
    assert r.status_code == 200, r.text
    assert rounds


@pytest.mark.asyncio
async def test_browser_data_never_reaches_a_shared_in_flight_fetch():
    """A request using browser data neither starts a fetch others join, nor
    joins one: its data can't become another visitor's weather."""
    fake = CountingOpenMeteo()
    svc = _service(fake)
    today = datetime.now(timezone.utc).date()

    token = client_data.set({})
    try:
        with pytest.raises(ClientFetchNeeded):
            await svc.get_forecast_range_windows(LAT, LON, days=7)
        assert not svc._inflight
    finally:
        client_data.reset(token)

    # A normal request in flight is not joined by a browser-data request.
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_build():
        started.set()
        await release.wait()
        return "server data"

    shared = asyncio.create_task(svc._shared_fetch(("k",), slow_build, since=None))
    await started.wait()
    token = client_data.set({})
    try:
        own = await svc._shared_fetch(("k",), lambda: asyncio.sleep(0, "browser data"), since=None)
    finally:
        client_data.reset(token)
    assert own == "browser data"
    release.set()
    assert await shared == "server data"
