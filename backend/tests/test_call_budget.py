"""Open-Meteo call budget: per-client limit and process-wide soft cap."""
from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.main import CallBudgetMiddleware
from app.utils.call_budget import CallBudget, client_key, current_client, weighted_cost


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


# ---------------------------------------------------------------------------
# Weighting — must match what scripts/capacity measured
# ---------------------------------------------------------------------------

def test_weighted_cost_matches_open_meteo_rules():
    forecast = {"latitude": 32.0, "hourly": ",".join(f"v{i}" for i in range(13)), "forecast_days": 7}
    corridor = {"latitude": ",".join(["32"] * 6), "hourly": "a,b", "forecast_days": 4}
    archive = {"latitude": 32.0, "hourly": ",".join(f"v{i}" for i in range(12)),
               "start_date": "2026-08-01", "end_date": "2026-08-31"}
    assert weighted_cost(forecast) == pytest.approx(1.3)
    assert weighted_cost(corridor) == pytest.approx(6.0)
    assert weighted_cost(archive) == pytest.approx(12 / 10 * 31 / 14)
    assert weighted_cost({"latitude": 1, "hourly": "a"}) == 1.0


# ---------------------------------------------------------------------------
# Per-client hourly limit
# ---------------------------------------------------------------------------

def test_client_over_limit_until_calls_age_out():
    clock = Clock()
    b = CallBudget(client_hourly_limit=100, daily_soft_cap=0, clock=clock)
    b.charge(60, client="a")
    clock.t += 600
    b.charge(60, client="a")

    assert b.client_retry_after("b") is None
    retry = b.client_retry_after("a")
    assert retry == 3000  # the first 60 ages out an hour after it was spent

    clock.t += retry
    assert b.client_retry_after("a") is None


def test_charge_uses_current_client_context():
    b = CallBudget(client_hourly_limit=10, daily_soft_cap=0)
    token = current_client.set("ctx")
    try:
        b.charge(11)
    finally:
        current_client.reset(token)
    b.charge(50)  # no client: alert runs, scripts — never limited
    assert b.client_retry_after("ctx") is not None
    assert b.day_total() == 61


def test_zero_limits_disable_checks():
    b = CallBudget(client_hourly_limit=0, daily_soft_cap=0)
    b.charge(1e9, client="a")
    assert b.client_retry_after("a") is None
    assert b.optional_work_allowed()


# ---------------------------------------------------------------------------
# Process-wide soft cap
# ---------------------------------------------------------------------------

def test_soft_cap_pauses_optional_work_for_a_rolling_day():
    clock = Clock()
    b = CallBudget(client_hourly_limit=0, daily_soft_cap=100, clock=clock)
    b.charge(99)
    assert b.optional_work_allowed()
    b.charge(1)
    assert not b.optional_work_allowed()
    clock.t += 86_400
    assert b.optional_work_allowed()


# ---------------------------------------------------------------------------
# Client identity behind Render (Cloudflare in front)
# ---------------------------------------------------------------------------

def test_client_key_prefers_cloudflare_header():
    assert client_key({"cf-connecting-ip": "1.1.1.1", "x-forwarded-for": "9.9.9.9, 2.2.2.2"}, "10.0.0.1") == "1.1.1.1"
    assert client_key({"x-forwarded-for": "9.9.9.9, 2.2.2.2"}, "10.0.0.1") == "9.9.9.9"
    assert client_key({}, "10.0.0.1") == "10.0.0.1"


# ---------------------------------------------------------------------------
# Middleware: refuse at request start, never mid-request
# ---------------------------------------------------------------------------

def _app(budget: CallBudget, cost: float) -> FastAPI:
    app = FastAPI()
    app.add_middleware(CallBudgetMiddleware)
    app.state.call_budget = budget
    seen: list = []

    @app.post("/predict")
    async def predict(request: Request):
        # Charge from a background task too: it must count for the same client.
        await asyncio.get_running_loop().create_task(_charge_later(budget, cost / 2))
        budget.charge(cost / 2)
        seen.append(current_client.get())
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True}

    app.state.seen = seen
    return app


async def _charge_later(budget: CallBudget, w: float) -> None:
    budget.charge(w)


def test_middleware_refuses_client_over_limit_with_429():
    budget = CallBudget(client_hourly_limit=1000, daily_soft_cap=0)
    app = _app(budget, cost=600)
    with TestClient(app) as c:
        me = {"cf-connecting-ip": "203.0.113.5"}
        assert c.post("/predict", headers=me).status_code == 200
        assert c.post("/predict", headers=me).status_code == 200  # 1,200 spent: now over
        r = c.post("/predict", headers=me)
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) > 0
        assert r.headers["access-control-allow-origin"] == "*"
        assert "try again" in r.json()["detail"]

        # Someone else, and endpoints that never call Open-Meteo, are unaffected.
        assert c.post("/predict", headers={"cf-connecting-ip": "198.51.100.7"}).status_code == 200
        assert c.get("/health", headers=me).status_code == 200
    assert app.state.seen[0] == "203.0.113.5"


def test_heatmap_refused_up_front_past_soft_cap():
    from app.main import app as real_app

    with TestClient(real_app) as c:
        budget = real_app.state.call_budget
        budget.charge(real_app.state.settings.OPEN_METEO_DAILY_SOFT_CAP)
        r = c.get("/heatmap?lat=32.08&lon=34.78&months=1")
        assert r.status_code == 503
        assert "tonight" in r.json()["detail"]


def test_climatology_build_spends_nothing_past_soft_cap():
    """Past the daily soft cap a build may still run from cached data, but it
    can't send a single Open-Meteo request (see tests/test_tonight_first.py):
    it fails as "busy" and is retried after its cooldown."""

    from app.services.climatology_service import ClimatologyService
    from app.services.scoring_engine import ScoringEngine
    from app.services.weather_service import WeatherBusyError
    from tests.test_shared_forecast_fetch import CountingOpenMeteo, _service

    fake = CountingOpenMeteo()
    weather = _service(fake)
    weather.budget = CallBudget(client_hourly_limit=0, daily_soft_cap=1)
    weather.budget.charge(1)
    svc = ClimatologyService(weather, weather._astro, ScoringEngine(), weather._cache)

    with pytest.raises(WeatherBusyError):
        asyncio.run(svc.build(32.08, 34.78))
    assert sum(fake.calls.values()) == 0
