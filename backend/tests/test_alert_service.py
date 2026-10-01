from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from app.services.alert_service import AlertService, build_payload
from app.services.subscription_store import InMemorySubscriptionStore
from app.services.weather_service import WeatherUnavailableError

NOW = datetime(2026, 9, 30, 11, 0, tzinfo=timezone.utc)
DAY = date(2026, 9, 30)
TLV = {"latitude": 32.0853, "longitude": 34.7818, "name": "Tel Aviv"}
JAFFA = {"latitude": 32.0540, "longitude": 34.7520, "name": "Jaffa"}   # same 0.1° cell as TLV
HAIFA = {"latitude": 32.7940, "longitude": 34.9896, "name": "Haifa"}


def fake_prediction(score: float, category: str):
    sunset = NOW + timedelta(hours=4)
    return SimpleNamespace(
        beauty_score_0_100=score, category=category,
        sunset_time=sunset, best_window_point="+15m",
    )


class FakeSender:
    def __init__(self, result="ok"):
        self.result = result
        self.sent: list[tuple[str, dict]] = []

    async def send(self, sub, payload):
        self.sent.append((sub.endpoint, payload))
        return self.result


def make_service(store, sender, *, score=85.0, category="Epic", lead_hours=4.0, fail=False, leads=None):
    calls: list[tuple[float, float, date]] = []

    async def predictor(lat, lon, day):
        calls.append((lat, lon, day))
        if fail:
            raise WeatherUnavailableError("rate limited")
        return fake_prediction(score, category)

    svc = AlertService(
        store=store,
        predictor=predictor,
        sunset_for=lambda lat, lon, d: NOW + timedelta(hours=(leads or {}).get(round(lat, 1), lead_hours)),
        local_date_for=lambda lat, lon: DAY,
        sender=sender,
        clock=lambda: NOW,
    )
    return svc, calls


def run(coro):
    return asyncio.run(coro)


def seed(store, *subs):
    for endpoint, places in subs:
        run(store.upsert(endpoint, "k", "a", places, "Asia/Jerusalem"))


def test_one_prediction_per_cell_regardless_of_subscribers():
    store = InMemorySubscriptionStore()
    seed(store, ("https://p/1", [TLV]), ("https://p/2", [JAFFA]), ("https://p/3", [HAIFA]))
    sender = FakeSender()
    svc, calls = make_service(store, sender)

    summary = run(svc.run())

    assert len(calls) == 2, "TLV+Jaffa share a cell; Haifa is another"
    assert summary.cells == 2 and summary.cells_checked == 2
    assert summary.notifications_sent == 3


def test_same_subscriber_two_places_in_one_cell_gets_one_push():
    store = InMemorySubscriptionStore()
    seed(store, ("https://p/1", [TLV, JAFFA]))
    sender = FakeSender()
    svc, _ = make_service(store, sender)
    run(svc.run())
    assert len(sender.sent) == 1


def test_outside_window_is_not_checked():
    store = InMemorySubscriptionStore()
    seed(store, ("https://p/1", [TLV]))
    for lead in (5.0, 2.0, -1.0):
        svc, calls = make_service(store, FakeSender(), lead_hours=lead)
        run(svc.run())
        assert calls == [], f"lead {lead}h must not be due"


def test_cell_checked_once_per_day_even_if_not_epic():
    store = InMemorySubscriptionStore()
    seed(store, ("https://p/1", [TLV]))
    sender = FakeSender()
    svc, calls = make_service(store, sender, score=40.0, category="Decent")
    run(svc.run())
    run(svc.run())
    assert len(calls) == 1
    assert sender.sent == []


def test_no_duplicate_notification_same_day():
    store = InMemorySubscriptionStore()
    seed(store, ("https://p/1", [TLV]))
    sender = FakeSender()
    svc, _ = make_service(store, sender)
    run(svc.run())
    run(svc.run(force=False))
    assert len(sender.sent) == 1


def test_weather_failure_is_retried_next_run():
    store = InMemorySubscriptionStore()
    seed(store, ("https://p/1", [TLV]))
    svc, calls = make_service(store, FakeSender(), fail=True)
    summary = run(svc.run())
    assert summary.cells_due == 1 and summary.cells_checked == 0
    assert not run(store.cell_checked("32.1,34.8", DAY))


def test_gone_subscription_is_pruned():
    store = InMemorySubscriptionStore()
    seed(store, ("https://p/1", [TLV, HAIFA]))
    sender = FakeSender(result="gone")
    svc, _ = make_service(store, sender)
    summary = run(svc.run())
    assert summary.pruned == 1
    assert len(sender.sent) == 1, "a gone endpoint is not retried for its other cells"
    assert run(store.all()) == []


def test_force_ignores_window_threshold_and_dedupe():
    store = InMemorySubscriptionStore()
    seed(store, ("https://p/1", [TLV]))
    sender = FakeSender()
    svc, calls = make_service(store, sender, score=20.0, category="Poor", lead_hours=10.0)
    run(svc.run(force=True))
    run(svc.run(force=True))
    assert len(calls) == 2 and len(sender.sent) == 2


def test_payload_uses_subscriber_timezone_and_best_point():
    pred = fake_prediction(86.4, "Epic")   # sunset 15:00Z, best +15m → 15:15Z
    p = build_payload(TLV, pred, "Asia/Jerusalem", "32.1,34.8", DAY)
    assert p["title"] == "🔥 Epic sunset tonight"
    assert p["body"] == "Tel Aviv — 86/100. Best around 18:15."
    assert p["url"] == "/?lat=32.0853&lon=34.7818&name=Tel%20Aviv"
    assert p["tag"] == "epic-32.1,34.8-2026-09-30"


def test_payload_falls_back_to_utc_for_unknown_tz():
    pred = fake_prediction(80.0, "Epic")
    p = build_payload(TLV, pred, "Not/AZone", "32.1,34.8", DAY)
    assert p["body"].endswith("Best around 15:15.")


def test_paced_run_checks_closest_to_sunset_first_and_reports_the_rest():
    store = InMemorySubscriptionStore()
    places = [{"latitude": 31.0 + i / 10, "longitude": 34.8, "name": f"P{i}"} for i in range(5)]
    seed(store, *((f"https://p/{i}", [p]) for i, p in enumerate(places)))
    leads = {round(p["latitude"], 1): 4.4 - i / 10 for i, p in enumerate(places)}
    svc, calls = make_service(store, FakeSender(), leads=leads)

    first = run(svc.run(max_cells=2))
    assert first.cells_checked == 2 and first.remaining == 3
    assert [round(c[0], 1) for c in calls] == [31.4, 31.3], "closest to sunset first"

    second = run(svc.run(max_cells=2))
    assert second.cells_checked == 2 and second.remaining == 1
    third = run(svc.run(max_cells=2))
    assert third.cells_checked == 1 and third.remaining == 0
    assert len(calls) == 5 and len(set(calls)) == 5


def test_cell_carried_over_is_still_checked_a_bit_later():
    store = InMemorySubscriptionStore()
    seed(store, ("https://p/1", [TLV]))
    svc, calls = make_service(store, FakeSender(), lead_hours=3.0)
    run(svc.run())
    assert len(calls) == 1
