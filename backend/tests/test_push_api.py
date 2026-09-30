from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.push import AlertRunSummary
from app.services.subscription_store import InMemorySubscriptionStore

SUB = {
    "endpoint": "https://web.push.apple.com/abc",
    "keys": {"p256dh": "BPk", "auth": "xyz"},
}
TLV = {"latitude": 32.0853, "longitude": 34.7818, "name": "Tel Aviv"}


class FakeAlertService:
    def __init__(self):
        self.forced: list[bool] = []

    async def run(self, force=False):
        self.forced.append(force)
        return AlertRunSummary(cells=1, cells_due=1, cells_checked=1, notifications_sent=1)


@pytest.fixture
def api(monkeypatch):
    store = InMemorySubscriptionStore()
    alerts = FakeAlertService()
    monkeypatch.setattr(app.state, "subscription_store", store, raising=False)
    monkeypatch.setattr(app.state, "alert_service", alerts, raising=False)
    monkeypatch.setattr(app.state, "settings", settings, raising=False)
    monkeypatch.setattr(settings, "VAPID_PUBLIC_KEY", "PUBKEY")
    monkeypatch.setattr(settings, "ALERTS_SECRET", "s3cret")
    return TestClient(app), store, alerts


def test_vapid_key(api):
    client, _, _ = api
    r = client.get("/push/vapid-key")
    assert r.status_code == 200 and r.json() == {"public_key": "PUBKEY"}


def test_subscribe_upserts_and_delete_removes(api):
    client, store, _ = api
    r = client.post("/push/subscribe", json={"subscription": SUB, "places": [TLV], "tz": "Asia/Jerusalem"})
    assert r.status_code == 204
    (sub,) = asyncio.run(store.all())
    assert sub.places == [TLV] and sub.tz == "Asia/Jerusalem"

    r = client.request("DELETE", "/push/subscribe", json={"endpoint": SUB["endpoint"]})
    assert r.status_code == 204
    assert asyncio.run(store.all()) == []


def test_subscribe_rejects_too_many_places(api):
    client, _, _ = api
    r = client.post("/push/subscribe", json={"subscription": SUB, "places": [TLV] * 6})
    assert r.status_code == 422


def test_subscribe_rejects_non_https_endpoint(api):
    client, _, _ = api
    bad = {**SUB, "endpoint": "http://evil/x"}
    r = client.post("/push/subscribe", json={"subscription": bad, "places": [TLV]})
    assert r.status_code == 422


def test_push_disabled_returns_503(api, monkeypatch):
    client, _, _ = api
    monkeypatch.setattr(app.state, "subscription_store", None)
    assert client.get("/push/vapid-key").status_code == 503
    assert client.post("/push/subscribe", json={"subscription": SUB, "places": [TLV]}).status_code == 503


def test_alert_run_requires_secret(api):
    client, _, alerts = api
    assert client.post("/internal/alerts/run").status_code == 401
    assert client.post("/internal/alerts/run", headers={"X-Alerts-Secret": "nope"}).status_code == 401
    assert alerts.forced == []


def test_alert_run_with_secret(api):
    client, _, alerts = api
    r = client.post("/internal/alerts/run?force=1", headers={"X-Alerts-Secret": "s3cret"})
    assert r.status_code == 200 and r.json()["notifications_sent"] == 1
    assert alerts.forced == [True]


def test_alert_run_refuses_when_secret_unset(api, monkeypatch):
    client, _, _ = api
    monkeypatch.setattr(settings, "ALERTS_SECRET", "")
    r = client.post("/internal/alerts/run", headers={"X-Alerts-Secret": ""})
    assert r.status_code == 401
