"""Tonight first: nothing else may use the share of Open-Meteo's limits kept
for tonight's prediction.

Open-Meteo's free tier allows 600 weighted calls a minute, 5,000 an hour and
10,000 a day, shared by everything this app does. A burst of cold heatmaps
(~234 calls each) or a busy day of 7-day forecasts used to be able to spend
them all, and then tonight's prediction failed too. Now everything except
tonight is paced to a per-minute share (it waits) and capped per hour and day
(it is refused up front), and tonight's calls are never held back.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.utils.call_budget import (
    OTHER,
    TONIGHT,
    BudgetExhausted,
    CallBudget,
    PrioritySlots,
    priority,
    request_deadline,
)
from app.services.weather_service import WeatherBusyError
from tests.test_shared_forecast_fetch import LAT, LON, CountingOpenMeteo, _service

UTC = timezone.utc


class Clock:
    def __init__(self, t: float = 1_790_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _budget(clock: Clock, **kw) -> CallBudget:
    async def sleep(seconds: float) -> None:  # time passes instantly
        clock.t += seconds

    kw.setdefault("client_hourly_limit", 0)
    kw.setdefault("daily_soft_cap", 0)
    return CallBudget(clock=clock, sleep=sleep, **kw)


# ---------------------------------------------------------------------------
# CallBudget
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_other_work_waits_for_the_minute_tonight_does_not():
    clock = Clock()
    budget = _budget(clock, other_minute_cap=10)
    await budget.acquire_other(8)
    budget.charge(50)                # tonight: recorded, never held back
    start = clock.t
    await budget.acquire_other(5)    # 63 in the minute > 10: waits it out
    assert clock.t - start >= 59


@pytest.mark.asyncio
async def test_hourly_and_daily_shares_refuse_other_work_not_tonight():
    clock = Clock()
    budget = _budget(clock, other_hour_cap=100, daily_soft_cap=1000)
    budget.charge(95)
    assert not budget.optional_work_allowed(headroom=10)
    with pytest.raises(BudgetExhausted):
        await budget.acquire_other(10)
    budget.charge(500)               # tonight still goes through
    clock.t += 3601                  # the hour rolls over; the day hasn't
    assert budget.optional_work_allowed()
    budget.charge(500)
    with pytest.raises(BudgetExhausted):
        await budget.acquire_other(10)   # 1,095 today > the 1,000 daily share


@pytest.mark.asyncio
async def test_open_meteo_limit_messages_hold_other_work_back():
    clock = Clock()
    budget = _budget(clock, other_minute_cap=400)
    budget.note_rate_limited("Minutely API request limit exceeded. Please try again in one minute.")
    assert budget.other_wait_remaining() == pytest.approx(60)
    start = clock.t
    await budget.acquire_other(1)
    assert clock.t - start >= 60

    budget.note_rate_limited("Daily API request limit exceeded. Please try again tomorrow.")
    assert not budget.optional_work_allowed()
    with pytest.raises(BudgetExhausted):
        await budget.acquire_other(1)
    clock.t += 86_400                # past the next 00:00 UTC
    assert budget.optional_work_allowed()


@pytest.mark.asyncio
async def test_a_wait_longer_than_allowed_is_refused():
    clock = Clock()
    budget = _budget(clock, other_minute_cap=10, other_max_wait=5)
    await budget.acquire_other(10)
    with pytest.raises(BudgetExhausted):
        await budget.acquire_other(10)


# ---------------------------------------------------------------------------
# PrioritySlots
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_free_slot_goes_to_tonight_before_queued_other_work():
    slots = PrioritySlots(1)
    await slots.acquire(OTHER)
    order: list[str] = []

    async def take(level: str) -> None:
        await slots.acquire(level)
        order.append(level)
        slots.release()

    queued = [asyncio.create_task(take(OTHER)) for _ in range(3)]
    await asyncio.sleep(0)
    tonight = asyncio.create_task(take(TONIGHT))
    await asyncio.sleep(0)
    slots.release()
    await asyncio.gather(*queued, tonight)
    assert order[0] == TONIGHT


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_a_slot():
    slots = PrioritySlots(1)
    await slots.acquire(OTHER)
    waiter = asyncio.create_task(slots.acquire(OTHER))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    slots.release()
    await asyncio.wait_for(slots.acquire(TONIGHT), 1)  # the slot is free again


# ---------------------------------------------------------------------------
# WeatherService: classified by the data, not by who asks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tonights_data_is_fetched_even_when_other_work_is_refused():
    """The 7-day page and /predict tonight share the weather bundle, the
    ensemble and tonight's corridor: those are fetched at tonight's priority
    even when the request asking is not tonight's."""
    fake = CountingOpenMeteo()
    svc = _service(fake)
    svc.budget = _budget(Clock(datetime.now(UTC).timestamp()), other_hour_cap=1)
    svc.budget.charge(5)             # OTHER's hourly share is already used up
    today = datetime.now(UTC).date()
    sunset = svc._astro.get_sunset_time(LAT, LON, today)

    with priority(OTHER):            # e.g. the 7-day page asking first
        snaps = await svc.get_window_snapshots(LAT, LON, today, sunset)
        corridor = await svc.get_corridor_samples(LAT, LON, today, sunset)
        spread = await svc.get_ensemble_cloud_spread(LAT, LON, today, sunset)
    assert snaps and corridor and spread is not None

    # Day+3's corridor is not tonight's: refused as "busy" — never quietly
    # scored without its corridor.
    d3 = today + timedelta(days=3)
    with priority(OTHER), pytest.raises(WeatherBusyError):
        await svc.get_corridor_samples(LAT, LON, d3, svc._astro.get_sunset_time(LAT, LON, d3))


@pytest.mark.asyncio
async def test_busy_history_fails_whole_never_partial():
    """No month scored without its corridor, no proxy aerosol: a history
    request that can't get its share fails as a whole ("busy")."""
    fake = CountingOpenMeteo()
    svc = _service(fake)
    svc.budget = _budget(Clock(datetime.now(UTC).timestamp()), other_hour_cap=1)
    svc.budget.charge(5)
    today = datetime.now(UTC).date()
    past = [today - timedelta(days=k) for k in range(40, 30, -1)]
    with pytest.raises(WeatherBusyError):
        await svc.get_corridor_samples_map(LAT, LON, past)
    with pytest.raises(WeatherBusyError):
        await svc.get_historical_range_windows(LAT, LON, past[0], past[-1])
    with pytest.raises(WeatherBusyError):
        await svc._fetch_air_quality_raw(LAT, LON, days=7)


@pytest.mark.asyncio
async def test_one_wait_deadline_for_the_whole_request():
    clock = Clock()
    budget = _budget(clock, other_minute_cap=10, other_max_wait=60)
    await budget.acquire_other(10)
    token = request_deadline.set(clock.t + 5)   # the request has 5 s left
    try:
        with pytest.raises(BudgetExhausted):
            await budget.acquire_other(10)        # would need ~60 s
    finally:
        request_deadline.reset(token)


# ---------------------------------------------------------------------------
# Endpoints: refused up front, tonight unaffected
# ---------------------------------------------------------------------------

def test_7_day_and_other_dates_refused_up_front_tonight_still_served():
    override = {
        "cloud_low": 10, "cloud_mid": 20, "cloud_high": 30, "cloud_total": 40,
        "visibility_m": 20000, "relative_humidity": 50, "precipitation_mm": 0,
    }
    with TestClient(app) as client:
        budget = app.state.call_budget
        budget.note_rate_limited("Hourly API request limit exceeded.")
        try:
            r = client.post("/forecast", json={"latitude": LAT, "longitude": LON, "days": 7})
            assert r.status_code == 503 and "tonight" in r.json()["detail"]
            other = (date.today() + timedelta(days=3)).isoformat()
            r = client.post("/predict", json={"latitude": LAT, "longitude": LON, "target_date": other})
            assert r.status_code == 503
            r = client.get("/heatmap", params={"lat": LAT, "lon": LON, "months": 6})
            assert r.status_code == 503
            r = client.post("/predict", json={"latitude": LAT, "longitude": LON,
                                              "weather_override": override})
            assert r.status_code == 200
        finally:
            budget._other_refuse_until = 0.0
