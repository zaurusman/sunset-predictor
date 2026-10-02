"""PoC: when Open-Meteo refuses the server, the browser fetches for tonight.

The server answers 503 with `client_fetch` (the exact URL it needed); the
browser fetches it and repeats the request with `client_data`. The score must
be exactly what the server computes when it can download the data itself.
"""
from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.utils.call_budget import CallBudget
from tests.test_shared_forecast_fetch import LAT, LON, CountingOpenMeteo

DROP = {"requested_at"}


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in DROP}


def test_browser_fetch_gives_the_same_score_as_a_server_fetch():
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
            # The server downloads everything itself.
            normal = client.post("/predict", json={"latitude": LAT, "longitude": LON})
            assert normal.status_code == 200
            ws._cache.clear()

            # Open-Meteo refuses the server: the browser fetches, round by round.
            ws._settings.OPEN_METEO_SIMULATE_DAILY_LIMIT = True
            server_calls = sum(fake.calls.values())
            data: dict = {}
            rounds = []
            for _ in range(10):
                r = client.post("/predict", json={"latitude": LAT, "longitude": LON,
                                                  **({"client_data": data} if data else {})})
                if r.status_code == 200:
                    break
                assert r.status_code == 503
                url = r.json()["client_fetch"]
                rounds.append(httpx.URL(url).host)
                data[url] = fake(httpx.Request("GET", url)).json()   # "the browser"
            assert r.status_code == 200, r.text
            assert _clean(r.json()) == _clean(normal.json())
            # Weather, aerosol, corridor and ensemble — each fetched once, by the browser.
            assert len(rounds) == len(set(data)) <= 6
            assert sum(fake.calls.values()) - server_calls == len(rounds)
            # Nothing the browser sent was written to the shared cache.
            assert ws._cache.size() == 0 or all(
                url not in str(k) for k in ws._cache._store for url in data
            )
        finally:
            ws._http, ws._runs, ws.budget, ps._climatology, ws._settings.OPEN_METEO_SIMULATE_DAILY_LIMIT = saved


def test_other_endpoints_never_hand_fetches_to_the_browser():
    with TestClient(app) as client:
        ws = app.state.prediction_service._weather
        ws._settings.OPEN_METEO_SIMULATE_DAILY_LIMIT = True
        try:
            r = client.post("/forecast", json={"latitude": 31.3, "longitude": 34.4, "days": 7})
            assert r.status_code == 503 and "client_fetch" not in r.json()
        finally:
            ws._settings.OPEN_METEO_SIMULATE_DAILY_LIMIT = False
