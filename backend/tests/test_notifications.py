"""Tests for push subscriptions and the evening dispatch run.

Nothing here touches a real push service or Open-Meteo: the push service and
the prediction service are stubbed, so the tests assert on the DECISIONS the
dispatcher makes — who is in the window, who is scored, who gets an alert —
rather than on delivery.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.schemas.notification import SubscribeRequest
from app.services.dispatch_schedule import compute_schedule
from app.services.notification_dispatcher import MAX_FAILURES, NotificationDispatcher
from app.services.push_service import PushGoneError
from app.services.subscription_store import SubscriptionStore, endpoint_key

UTC = timezone.utc

# Tel Aviv — the same reference location the live sanity tests use.
LAT, LON = 32.0853, 34.7818


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakePush:
    """Records what would have been sent; can be told to fail."""

    def __init__(self, error: Exception | None = None) -> None:
        self.sent: list[dict] = []
        self.error = error
        self.is_configured = True

    async def send(self, *, subscription, payload, ttl_seconds=None):
        if self.error is not None:
            raise self.error
        self.sent.append({"subscription": subscription, "payload": payload})


class FakePredictions:
    """Returns a fixed score, and counts how often it was asked."""

    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0

    async def predict(self, request):
        self.calls += 1
        sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
        return SimpleNamespace(
            beauty_score_0_100=self.score,
            category="Great",
            sunset_time=sunset,
        )


class FakeAstro:
    """Pins sunset so the tests do not drift with the real calendar."""

    def __init__(self, sunset: datetime) -> None:
        self.sunset = sunset

    def get_sunset_time(self, lat, lon, target_date):
        return self.sunset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path) -> SubscriptionStore:
    return SubscriptionStore(path=str(tmp_path / "subscriptions.json"))


def _client(tmp_path, monkeypatch, *, configured: bool):
    """A client that has actually run the lifespan, so app.state is populated.

    The shared `client` fixture constructs TestClient without entering it,
    which never triggers startup — these endpoints read services off
    app.state, so they need the real thing.

    Push config is set EXPLICITLY rather than inherited from the environment.
    A developer with real keys in backend/.env would otherwise flip the
    disabled-path tests from passing to failing, which is a property of their
    machine and not of the code.
    """
    from fastapi.testclient import TestClient

    from app.core.config import settings as app_settings
    from app.main import app as fastapi_app

    monkeypatch.setattr(
        app_settings, "SUBSCRIPTIONS_PATH", str(tmp_path / "subscriptions.json")
    )
    monkeypatch.setattr(app_settings, "VAPID_PUBLIC_KEY", "test-public" if configured else "")
    monkeypatch.setattr(app_settings, "VAPID_PRIVATE_KEY", "test-private" if configured else "")
    monkeypatch.setattr(app_settings, "VAPID_SUBJECT", "mailto:test@example.com" if configured else "")
    monkeypatch.setattr(app_settings, "NOTIFY_DISPATCH_SECRET", "test-secret" if configured else "")

    with TestClient(fastapi_app) as c:
        yield c


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """Server with push deliberately NOT configured."""
    yield from _client(tmp_path, monkeypatch, configured=False)


@pytest.fixture
def api_client_on(tmp_path, monkeypatch):
    """Server with push configured."""
    yield from _client(tmp_path, monkeypatch, configured=True)


def make_record(**overrides) -> dict:
    record = {
        "endpoint": "https://push.example.com/abc123",
        "keys": {"p256dh": "fake-p256dh", "auth": "fake-auth"},
        "latitude": LAT,
        "longitude": LON,
        "location_name": "Tel Aviv",
        "threshold": 70.0,
        "lead_minutes": 120,
        "created_at": "2026-08-01T00:00:00+00:00",
        "last_checked_date": None,
        "failure_count": 0,
    }
    record.update(overrides)
    return record


def make_dispatcher(store, *, score=85.0, sunset=None, push=None):
    sunset = sunset or datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    predictions = FakePredictions(score)
    push = push or FakePush()
    dispatcher = NotificationDispatcher(
        store=store,
        push_service=push,
        prediction_service=predictions,
        astro_service=FakeAstro(sunset),
    )
    return dispatcher, push, predictions


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_roundtrips_through_disk(tmp_path):
    path = str(tmp_path / "subs.json")
    store = SubscriptionStore(path=path)
    await store.upsert(make_record())

    # A fresh instance must see it — this is what survives a process restart.
    reloaded = SubscriptionStore(path=path)
    assert reloaded.count() == 1
    assert reloaded.get("https://push.example.com/abc123")["location_name"] == "Tel Aviv"


@pytest.mark.asyncio
async def test_resubscribing_updates_rather_than_duplicates(store):
    await store.upsert(make_record())
    await store.upsert(make_record(location_name="Haifa", latitude=32.79))

    assert store.count() == 1, "one browser must not accumulate subscriptions"
    rec = store.get("https://push.example.com/abc123")
    assert rec["location_name"] == "Haifa"
    # The original signup date survives a location change.
    assert rec["created_at"] == "2026-08-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_delete_is_idempotent(store):
    await store.upsert(make_record())
    assert await store.delete("https://push.example.com/abc123") is True
    assert await store.delete("https://push.example.com/abc123") is False
    assert store.count() == 0


def test_unreadable_store_does_not_crash_startup(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{ this is not json", encoding="utf-8")
    # Serving predictions matters more than serving alerts.
    assert SubscriptionStore(path=str(path)).count() == 0


def test_endpoint_key_is_stable_and_distinct():
    assert endpoint_key("https://a.example") == endpoint_key("https://a.example")
    assert endpoint_key("https://a.example") != endpoint_key("https://b.example")


# ---------------------------------------------------------------------------
# Dispatch timing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_alert_before_the_lead_window_opens(store):
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    await store.upsert(make_record())
    dispatcher, push, predictions = make_dispatcher(store, sunset=sunset)

    # Three hours out, with a two-hour lead: not yet.
    result = await dispatcher.run(now=sunset - timedelta(hours=3))

    assert result.in_window == 0
    assert predictions.calls == 0, "must not burn an API call outside the window"
    assert push.sent == []


@pytest.mark.asyncio
async def test_alert_sent_inside_the_window(store):
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    await store.upsert(make_record())
    dispatcher, push, _ = make_dispatcher(store, score=85.0, sunset=sunset)

    result = await dispatcher.run(now=sunset - timedelta(hours=1))

    assert result.sent == 1
    payload = push.sent[0]["payload"]
    assert payload["score"] == 85
    assert "Tel Aviv" in payload["body"]
    assert payload["url"].startswith("/?lat=")


@pytest.mark.asyncio
async def test_no_alert_after_sunset_but_the_day_is_closed_out(store):
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    await store.upsert(make_record())
    dispatcher, push, predictions = make_dispatcher(store, sunset=sunset)

    result = await dispatcher.run(now=sunset + timedelta(minutes=30))

    assert push.sent == []
    assert predictions.calls == 0
    # Stamped, so a missed evening is not retried after dark.
    rec = store.get("https://push.example.com/abc123")
    assert rec["last_checked_date"] is not None


@pytest.mark.asyncio
async def test_each_subscriber_is_scored_at_most_once_a_day(store):
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    await store.upsert(make_record())
    dispatcher, push, predictions = make_dispatcher(store, score=85.0, sunset=sunset)

    # A cron every 15 minutes hits the window many times over.
    for minutes in (110, 95, 80, 65, 50):
        await dispatcher.run(now=sunset - timedelta(minutes=minutes))

    assert predictions.calls == 1, "repeated cron ticks must not re-score"
    assert len(push.sent) == 1, "one alert per evening, not one per tick"


# ---------------------------------------------------------------------------
# Dispatch decisions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_below_threshold_is_scored_but_silent(store):
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    await store.upsert(make_record(threshold=70.0))
    dispatcher, push, predictions = make_dispatcher(store, score=64.0, sunset=sunset)

    result = await dispatcher.run(now=sunset - timedelta(hours=1))

    assert predictions.calls == 1
    assert result.below_threshold == 1
    assert push.sent == [], "an ordinary evening must not buzz someone's phone"


@pytest.mark.asyncio
async def test_a_lower_threshold_lets_the_same_evening_through(store):
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    await store.upsert(make_record(threshold=60.0))
    dispatcher, push, _ = make_dispatcher(store, score=64.0, sunset=sunset)

    await dispatcher.run(now=sunset - timedelta(hours=1))

    assert len(push.sent) == 1


@pytest.mark.asyncio
async def test_dead_subscription_is_pruned(store):
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    await store.upsert(make_record())
    dispatcher, _, _ = make_dispatcher(
        store, score=85.0, sunset=sunset, push=FakePush(error=PushGoneError("410"))
    )

    result = await dispatcher.run(now=sunset - timedelta(hours=1))

    assert result.pruned == 1
    assert store.count() == 0, "a browser that unsubscribed must not linger"


@pytest.mark.asyncio
async def test_transient_failure_keeps_the_subscription(store):
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    await store.upsert(make_record())
    dispatcher, _, _ = make_dispatcher(
        store, score=85.0, sunset=sunset, push=FakePush(error=RuntimeError("503"))
    )

    result = await dispatcher.run(now=sunset - timedelta(hours=1))

    assert result.failed == 1
    assert store.count() == 1, "one bad night must not cost someone their alerts"
    assert store.get("https://push.example.com/abc123")["failure_count"] == 1


@pytest.mark.asyncio
async def test_persistent_failure_eventually_prunes(store):
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    await store.upsert(make_record(failure_count=MAX_FAILURES - 1))
    dispatcher, _, _ = make_dispatcher(
        store, score=85.0, sunset=sunset, push=FakePush(error=RuntimeError("503"))
    )

    await dispatcher.run(now=sunset - timedelta(hours=1))

    assert store.count() == 0


@pytest.mark.asyncio
async def test_one_broken_subscription_does_not_stop_the_run(store):
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    # Missing "latitude" — the dispatcher must survive it and carry on.
    broken = make_record(endpoint="https://push.example.com/broken")
    del broken["latitude"]
    await store.upsert(broken)
    await store.upsert(make_record(endpoint="https://push.example.com/good"))

    dispatcher, push, _ = make_dispatcher(store, score=85.0, sunset=sunset)
    result = await dispatcher.run(now=sunset - timedelta(hours=1))

    assert result.failed == 1
    assert len(push.sent) == 1, "the healthy subscriber still gets their alert"


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_config_reports_disabled_without_keys(api_client):
    body = api_client.get("/notifications/config").json()
    # With no VAPID keys the feature must announce itself as off rather than
    # half-working — that is what makes the frontend hide the toggle.
    assert body["enabled"] is False
    assert body["vapid_public_key"] == ""
    assert body["default_threshold"] == 70.0


def test_config_hands_out_the_key_when_configured(api_client_on):
    body = api_client_on.get("/notifications/config").json()
    assert body["enabled"] is True
    assert body["vapid_public_key"] == "test-public"


def test_schedule_is_served_with_the_right_secret(api_client_on):
    resp = api_client_on.get(
        "/notifications/schedule", headers={"X-Dispatch-Secret": "test-secret"}
    )
    assert resp.status_code == 200
    body = resp.json()
    # Nobody is subscribed, so the scheduler should never wake the service.
    assert body["subscriber_count"] == 0
    assert body["cron_hours"] == []
    assert body["cron_expression"] is None


def test_schedule_rejects_a_bad_secret(api_client_on):
    resp = api_client_on.get(
        "/notifications/schedule", headers={"X-Dispatch-Secret": "wrong"}
    )
    assert resp.status_code == 401


def test_subscribe_refused_when_push_is_not_configured(api_client):
    resp = api_client.post(
        "/notifications/subscribe",
        json={
            "endpoint": "https://push.example.com/x",
            "keys": {"p256dh": "a", "auth": "b"},
            "latitude": LAT,
            "longitude": LON,
        },
    )
    assert resp.status_code == 503


def test_dispatch_refused_without_a_secret(api_client):
    resp = api_client.post("/notifications/dispatch", headers={"X-Dispatch-Secret": "guess"})
    # No NOTIFY_DISPATCH_SECRET configured in tests, so the endpoint is shut.
    assert resp.status_code == 503


def test_subscribe_rejects_an_out_of_range_threshold(api_client):
    resp = api_client.post(
        "/notifications/subscribe",
        json={
            "endpoint": "https://push.example.com/x",
            "keys": {"p256dh": "a", "auth": "b"},
            "latitude": LAT,
            "longitude": LON,
            "threshold": 150,
        },
    )
    assert resp.status_code == 422


def test_schedule_requires_the_secret(api_client):
    # It leaks the subscriber count and roughly where they are.
    assert api_client.get("/notifications/schedule").status_code == 503


def test_lead_below_the_hourly_floor_is_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        # 30 minutes could open and close between two hourly checks.
        SubscribeRequest(
            endpoint="https://push.example.com/x",
            keys={"p256dh": "a", "auth": "b"},
            latitude=LAT,
            longitude=LON,
            lead_minutes=30,
        )


# ---------------------------------------------------------------------------
# Dispatch schedule — what keeps the free tier free
# ---------------------------------------------------------------------------


def test_no_subscribers_means_never_waking_the_service():
    schedule = compute_schedule([], FakeAstro(datetime(2026, 8, 25, 16, 30, tzinfo=UTC)))
    assert schedule["cron_hours"] == []
    assert schedule["cron_expression"] is None
    assert schedule["next_window_opens"] is None


def test_schedule_covers_every_hour_the_window_touches():
    # Window 14:30 → 16:30 spans the 14:00, 15:00 and 16:00 checks. Marking
    # only the opening hour would miss the checks that actually fire.
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    schedule = compute_schedule(
        [make_record(lead_minutes=120)],
        FakeAstro(sunset),
        now=datetime(2026, 8, 25, 6, 0, tzinfo=UTC),
    )
    assert {14, 15, 16}.issubset(set(schedule["cron_hours"]))


def test_schedule_is_far_smaller_than_polling_all_day():
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    schedule = compute_schedule(
        [make_record(lead_minutes=120)],
        FakeAstro(sunset),
        now=datetime(2026, 8, 25, 6, 0, tzinfo=UTC),
    )
    # The whole point: a handful of wake-ups instead of 24 (or 72 at the old
    # 20-minute cadence). Each avoided wake is ~15 Render instance-minutes.
    assert len(schedule["cron_hours"]) <= 4


def test_schedule_respects_the_hourly_floor():
    # A record asking for a 15-minute lead predates the floor; the schedule
    # must still reserve a full hour or the hourly check could miss it.
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    schedule = compute_schedule(
        [make_record(lead_minutes=15)],
        FakeAstro(sunset),
        now=datetime(2026, 8, 25, 6, 0, tzinfo=UTC),
    )
    assert 15 in schedule["cron_hours"], "must cover the hour a 60-min window opens"


def test_schedule_skips_a_malformed_record_without_failing():
    broken = make_record()
    del broken["latitude"]
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    schedule = compute_schedule(
        [broken, make_record(endpoint="https://push.example.com/ok")],
        FakeAstro(sunset),
        now=datetime(2026, 8, 25, 6, 0, tzinfo=UTC),
    )
    assert schedule["subscriber_count"] == 2
    assert schedule["cron_hours"], "the healthy subscriber still gets covered"


def test_cron_expression_is_a_usable_crontab_line():
    sunset = datetime(2026, 8, 25, 16, 30, tzinfo=UTC)
    schedule = compute_schedule(
        [make_record()], FakeAstro(sunset), now=datetime(2026, 8, 25, 6, 0, tzinfo=UTC)
    )
    expr = schedule["cron_expression"]
    assert expr is not None
    minute, hours, dom, month, dow = expr.split()
    assert minute == "0" and dom == "*" and month == "*" and dow == "*"
    assert all(0 <= int(h) <= 23 for h in hours.split(","))


def test_subscribe_request_defaults_match_the_ui_bar():
    req = SubscribeRequest(
        endpoint="https://push.example.com/x",
        keys={"p256dh": "a", "auth": "b"},
        latitude=LAT,
        longitude=LON,
    )
    # Must track GO_OUTSIDE_THRESHOLD in the scoring engine.
    assert req.threshold == 70.0
    assert req.lead_minutes == 120
