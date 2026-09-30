# PWA Install + Epic Sunset Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users install Afterglow to their Home Screen (with a gentle, platform-aware prompt) and receive a web-push notification when an Epic sunset is forecast for a place they belled.

**Architecture:** Backend stores push subscriptions in Postgres (Neon) behind a small `SubscriptionStore` interface; an `AlertService` groups belled places into 0.1° cells and runs at most one prediction per cell per local day, ~4 h before sunset, triggered hourly by a GitHub Actions cron hitting a secret-protected endpoint. Frontend adds a hand-written service worker, an install/opt-in card under the verdict, and per-place bell toggles.

**Tech Stack:** FastAPI, asyncpg, pywebpush, pytest · Next.js 15 App Router, React 19, Tailwind, lucide-react · GitHub Actions.

Spec: `docs/superpowers/specs/2026-09-30-pwa-epic-alerts-design.md`

## Global Constraints

- Alert fires only when the prediction's `category == "Epic"` (score ≥ 80). No user-selectable threshold.
- At most one prediction per cell per local day; at most one notification per subscription per cell per local day.
- Cell = lat/lon rounded to `settings.CACHE_COORD_DECIMALS` (1 → 0.1°), shared with the weather cache.
- Due window: local sunset − 4.5 h ≤ now < sunset − 3.5 h.
- Alert predictions must NOT kick off a climatology warm-up (`warm_climatology=False`).
- Max 5 alert places per subscription (matches `MAX_SAVED_PLACES`).
- Install pitch never on first visit; only after the verdict renders; 14-day snooze per dismissal; hidden forever after 3 dismissals.
- If `DATABASE_URL` or VAPID keys are unset, the app behaves exactly as today; push endpoints return 503; the UI hides bells and the alerts card.
- Frontend theme: light default + `dark:` variants on every colour class.
- Frontend dev server must be started with NODE_ENV stripped (see Task 6 note).
- Backend tests run with the main checkout's venv: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -m pytest` from `backend/`.
- Known pre-existing failure: `test_predict_clear_sky_override` fails on main — not a regression.

## File Map

Backend
- Modify `backend/requirements.txt` — add `asyncpg`, `pywebpush`.
- Modify `backend/app/core/config.py` — push settings.
- Modify `backend/.env.example` — document push settings.
- Modify `backend/app/services/prediction_service.py` — `warm_climatology` kwarg.
- Create `backend/app/schemas/push.py` — request/response models.
- Create `backend/app/services/subscription_store.py` — `StoredSubscription`, `SubscriptionStore` protocol, `InMemorySubscriptionStore`, `PostgresSubscriptionStore`, `cell_key()`.
- Create `backend/app/services/push_sender.py` — `WebPushSender`.
- Create `backend/app/services/alert_service.py` — `AlertService`, `build_payload()`.
- Create `backend/app/api/push.py` — `/push/*` and `/internal/alerts/run`.
- Modify `backend/app/main.py` — wiring.
- Create `backend/scripts/generate_vapid_keys.py`.
- Tests: `backend/tests/test_subscription_store.py`, `backend/tests/test_alert_service.py`, `backend/tests/test_push_api.py`, `backend/tests/test_calibrate_warm_flag.py`.

Ops
- Create `.github/workflows/sunset-alerts.yml`.

Frontend
- Modify `frontend/src/app/manifest.ts`; create `frontend/public/icon-maskable-512.png`.
- Create `frontend/public/sw.js`.
- Create `frontend/src/components/ServiceWorkerRegistrar.tsx`; modify `frontend/src/app/layout.tsx`.
- Create `frontend/src/lib/install.ts`, `frontend/src/lib/push.ts`; modify `frontend/src/lib/api.ts`.
- Create `frontend/src/components/InstallPrompt.tsx`, `frontend/src/components/IosInstallSheet.tsx`.
- Modify `frontend/src/app/page.tsx`, `frontend/src/components/LocationSheet.tsx`.

---

### Task 1: Config, dependencies, and the `warm_climatology` flag

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/core/config.py` (append before the `settings = Settings()` line)
- Modify: `backend/.env.example`
- Modify: `backend/app/services/prediction_service.py:64` (`predict`) and `:210` (`_calibrate`)
- Test: `backend/tests/test_calibrate_warm_flag.py`

**Interfaces:**
- Produces: `PredictionService.predict(request, *, warm_climatology: bool = True) -> PredictResponse`; settings `DATABASE_URL`, `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`, `VAPID_SUBJECT`, `ALERTS_SECRET`, `ALERT_LEAD_MIN_HOURS`, `ALERT_LEAD_MAX_HOURS`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_calibrate_warm_flag.py`:
```python
"""Alert runs must never trigger a climatology build (extra archive fetches)."""
from __future__ import annotations

from app.core.config import settings
from app.services.prediction_service import PredictionService


class FakeClimatology:
    def __init__(self) -> None:
        self.warmed: list[tuple[float, float]] = []

    def percentile_of(self, lat, lon, raw_score, on_date=None):
        return 0.5, False  # not local → would normally warm

    def warm_in_background(self, lat, lon):
        self.warmed.append((lat, lon))


def _service(clim: FakeClimatology) -> PredictionService:
    return PredictionService(
        weather_service=None, astro_service=None, scoring_engine=None,
        explanation_engine=None, ml_model=None, settings=settings, climatology=clim,
    )


def test_calibrate_warms_by_default():
    clim = FakeClimatology()
    _service(clim)._calibrate(60.0, 32.1, 34.8)
    assert clim.warmed == [(32.1, 34.8)]


def test_calibrate_skips_warm_when_disabled():
    clim = FakeClimatology()
    _service(clim)._calibrate(60.0, 32.1, 34.8, warm=False)
    assert clim.warmed == []
```

- [ ] **Step 2: Run to verify it fails**

Run (from `backend/`): `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -m pytest tests/test_calibrate_warm_flag.py -v`
Expected: `test_calibrate_skips_warm_when_disabled` FAILS with `TypeError: ... unexpected keyword argument 'warm'`.

- [ ] **Step 3: Implement**

In `prediction_service.py`, change the `predict` signature and the `_calibrate` call:
```python
    async def predict(
        self, request: PredictRequest, *, warm_climatology: bool = True
    ) -> PredictResponse:
```
```python
        final_score, percentile, is_local = self._calibrate(
            raw_score, lat, lon, warm=warm_climatology
        )
```
Change `_calibrate`:
```python
    def _calibrate(
        self, raw_score: float, lat: float, lon: float, *, warm: bool = True
    ) -> tuple[float, Optional[float], bool]:
```
and its tail:
```python
        if not is_local and warm:
            self._climatology.warm_in_background(lat, lon)
        return raw_score, percentile, is_local
```
Add to the `_calibrate` docstring's last paragraph: `Alert runs pass warm=False: a background push check must not spend archive fetches building a curve nobody is looking at.`

In `config.py`, before `# Module-level singleton`, inside `Settings`:
```python
    # ── Push alerts (Epic sunset notifications) ──────────────────────────────
    # All empty → push is disabled and the app behaves exactly as without it.
    # Neon/Postgres DSN. Render's disk is ephemeral, so subscriptions can't live
    # in a local file.
    DATABASE_URL: str = ""
    # Generate with: python scripts/generate_vapid_keys.py
    VAPID_PUBLIC_KEY: str = ""
    VAPID_PRIVATE_KEY: str = ""
    # Contact claim push services require (mailto: or https:).
    VAPID_SUBJECT: str = "https://sunset-predictor-henna.vercel.app"
    # Shared secret the hourly GitHub Actions cron sends in X-Alerts-Secret.
    ALERTS_SECRET: str = ""
    # A cell is checked once, when its sunset is this many hours away.
    ALERT_LEAD_MIN_HOURS: float = 3.5
    ALERT_LEAD_MAX_HOURS: float = 4.5
```

`requirements.txt`, under `# ── Utilities`:
```
asyncpg==0.30.0           # Postgres driver for push subscriptions
pywebpush==2.0.3          # Web Push (VAPID) sender
```

Append to `backend/.env.example`:
```
# ── Push alerts (optional; all empty = disabled) ──
# Neon DSN. asyncpg rejects `channel_binding=…`; the app strips it automatically.
DATABASE_URL=
VAPID_PUBLIC_KEY=
VAPID_PRIVATE_KEY=
VAPID_SUBJECT=https://sunset-predictor-henna.vercel.app
ALERTS_SECRET=
```

Install: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/pip install asyncpg==0.30.0 pywebpush==2.0.3`

- [ ] **Step 4: Run tests**

Run: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -m pytest tests/test_calibrate_warm_flag.py -v` → PASS.
Run full suite: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -m pytest -q` → only the known `test_predict_clear_sky_override` failure.

- [ ] **Step 5: Commit**
```bash
git add backend/requirements.txt backend/app/core/config.py backend/.env.example backend/app/services/prediction_service.py backend/tests/test_calibrate_warm_flag.py
git commit -m "feat(push): add push settings and let predict() skip climatology warm-up"
```

---

### Task 2: Subscription store (protocol, in-memory, Postgres)

**Files:**
- Create: `backend/app/services/subscription_store.py`
- Test: `backend/tests/test_subscription_store.py`

**Interfaces:**
- Produces:
  - `cell_key(lat: float, lon: float, decimals: int = 1) -> str` → e.g. `"32.1,34.8"`
  - `@dataclass StoredSubscription(endpoint: str, p256dh: str, auth: str, places: list[dict], tz: str, last_notified: dict[str, str])` — `places` items are `{"latitude": float, "longitude": float, "name": str}`; `last_notified` maps cell key → ISO date.
  - `SubscriptionStore` Protocol, async methods: `upsert(endpoint, p256dh, auth, places, tz) -> None`, `delete(endpoint) -> None`, `all() -> list[StoredSubscription]`, `mark_notified(endpoint, cell, day: date) -> None`, `cell_checked(cell, day: date) -> bool`, `record_cell_check(cell, day: date, score: float) -> None`.
  - `InMemorySubscriptionStore()`; `PostgresSubscriptionStore.connect(dsn: str) -> PostgresSubscriptionStore` (async classmethod), `.close()`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_subscription_store.py`:
```python
from __future__ import annotations

import asyncio
import os
from datetime import date

import pytest

from app.services.subscription_store import (
    InMemorySubscriptionStore,
    PostgresSubscriptionStore,
    _clean_dsn,
    cell_key,
)

TLV = {"latitude": 32.0853, "longitude": 34.7818, "name": "Tel Aviv"}


def test_cell_key_rounds_to_tenth_degree():
    assert cell_key(32.0853, 34.7818) == "32.1,34.8"
    assert cell_key(-0.04, -0.06) == "-0.0,-0.1"


def test_clean_dsn_strips_channel_binding():
    dsn = "postgresql://u:p@h/db?sslmode=require&channel_binding=require"
    assert _clean_dsn(dsn) == "postgresql://u:p@h/db?sslmode=require"


async def _exercise(store) -> None:
    await store.upsert("https://push/a", "k1", "a1", [TLV], "Asia/Jerusalem")
    await store.upsert("https://push/a", "k2", "a2", [TLV], "Asia/Jerusalem")  # update, not dup
    subs = await store.all()
    assert len(subs) == 1 and subs[0].p256dh == "k2" and subs[0].places == [TLV]

    day = date(2026, 9, 30)
    await store.mark_notified("https://push/a", "32.1,34.8", day)
    await store.upsert("https://push/a", "k2", "a2", [TLV], "Asia/Jerusalem")
    (sub,) = await store.all()
    assert sub.last_notified == {"32.1,34.8": "2026-09-30"}, "upsert must keep dedupe state"

    assert not await store.cell_checked("32.1,34.8", day)
    await store.record_cell_check("32.1,34.8", day, 42.0)
    await store.record_cell_check("32.1,34.8", day, 43.0)  # idempotent
    assert await store.cell_checked("32.1,34.8", day)

    await store.delete("https://push/a")
    assert await store.all() == []


def test_in_memory_store_contract():
    asyncio.run(_exercise(InMemorySubscriptionStore()))


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to run")
def test_postgres_store_contract():
    async def go():
        store = await PostgresSubscriptionStore.connect(os.environ["TEST_DATABASE_URL"])
        try:
            await store._pool.execute("TRUNCATE push_subscriptions, alert_cell_checks")
            await _exercise(store)
        finally:
            await store.close()
    asyncio.run(go())
```

- [ ] **Step 2: Run to verify failure**

Run: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -m pytest tests/test_subscription_store.py -v`
Expected: collection error `ModuleNotFoundError: app.services.subscription_store`.

- [ ] **Step 3: Implement**

`backend/app/services/subscription_store.py`:
```python
"""Where Web Push subscriptions and alert bookkeeping live.

Render's free tier has an ephemeral disk, so production uses Postgres (Neon).
Tests and local runs without DATABASE_URL use the in-memory store. Both honour
the same contract; see tests/test_subscription_store.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def cell_key(lat: float, lon: float, decimals: int = 1) -> str:
    """Grid cell for alert grouping — same rounding as the weather cache key,
    so an alert check and a user's own lookup share one Open-Meteo fetch."""
    return f"{round(lat, decimals):.{decimals}f},{round(lon, decimals):.{decimals}f}"


@dataclass
class StoredSubscription:
    endpoint: str
    p256dh: str
    auth: str
    places: list[dict]
    tz: str = "UTC"
    last_notified: dict[str, str] = field(default_factory=dict)


class SubscriptionStore(Protocol):
    async def upsert(self, endpoint: str, p256dh: str, auth: str, places: list[dict], tz: str) -> None: ...
    async def delete(self, endpoint: str) -> None: ...
    async def all(self) -> list[StoredSubscription]: ...
    async def mark_notified(self, endpoint: str, cell: str, day: date) -> None: ...
    async def cell_checked(self, cell: str, day: date) -> bool: ...
    async def record_cell_check(self, cell: str, day: date, score: float) -> None: ...


class InMemorySubscriptionStore:
    def __init__(self) -> None:
        self._subs: dict[str, StoredSubscription] = {}
        self._checks: set[tuple[str, str]] = set()

    async def upsert(self, endpoint, p256dh, auth, places, tz) -> None:
        prev = self._subs.get(endpoint)
        self._subs[endpoint] = StoredSubscription(
            endpoint, p256dh, auth, list(places), tz,
            dict(prev.last_notified) if prev else {},
        )

    async def delete(self, endpoint) -> None:
        self._subs.pop(endpoint, None)

    async def all(self) -> list[StoredSubscription]:
        return list(self._subs.values())

    async def mark_notified(self, endpoint, cell, day) -> None:
        if endpoint in self._subs:
            self._subs[endpoint].last_notified[cell] = day.isoformat()

    async def cell_checked(self, cell, day) -> bool:
        return (cell, day.isoformat()) in self._checks

    async def record_cell_check(self, cell, day, score) -> None:
        self._checks.add((cell, day.isoformat()))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS push_subscriptions (
    endpoint      text PRIMARY KEY,
    p256dh        text NOT NULL,
    auth          text NOT NULL,
    places        jsonb NOT NULL DEFAULT '[]',
    tz            text NOT NULL DEFAULT 'UTC',
    last_notified jsonb NOT NULL DEFAULT '{}',
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS alert_cell_checks (
    cell_key   text NOT NULL,
    local_date date NOT NULL,
    score      real NOT NULL,
    checked_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (cell_key, local_date)
);
"""


def _clean_dsn(dsn: str) -> str:
    """Neon's copy-paste DSN includes channel_binding, which asyncpg rejects."""
    parts = urlsplit(dsn)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "channel_binding"]
    return urlunsplit(parts._replace(query=urlencode(query)))


class PostgresSubscriptionStore:
    def __init__(self, pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> "PostgresSubscriptionStore":
        import asyncpg  # imported lazily so the app runs without the driver configured

        pool = await asyncpg.create_pool(_clean_dsn(dsn), min_size=1, max_size=3)
        await pool.execute(_SCHEMA)
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def upsert(self, endpoint, p256dh, auth, places, tz) -> None:
        await self._pool.execute(
            """
            INSERT INTO push_subscriptions (endpoint, p256dh, auth, places, tz)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (endpoint) DO UPDATE
               SET p256dh = EXCLUDED.p256dh, auth = EXCLUDED.auth,
                   places = EXCLUDED.places, tz = EXCLUDED.tz, updated_at = now()
            """,
            endpoint, p256dh, auth, json.dumps(places), tz,
        )

    async def delete(self, endpoint) -> None:
        await self._pool.execute("DELETE FROM push_subscriptions WHERE endpoint = $1", endpoint)

    async def all(self) -> list[StoredSubscription]:
        rows = await self._pool.fetch(
            "SELECT endpoint, p256dh, auth, places, tz, last_notified FROM push_subscriptions"
        )
        return [
            StoredSubscription(
                r["endpoint"], r["p256dh"], r["auth"],
                json.loads(r["places"]), r["tz"], json.loads(r["last_notified"]),
            )
            for r in rows
        ]

    async def mark_notified(self, endpoint, cell, day) -> None:
        await self._pool.execute(
            """
            UPDATE push_subscriptions
               SET last_notified = jsonb_set(last_notified, ARRAY[$2::text], to_jsonb($3::text))
             WHERE endpoint = $1
            """,
            endpoint, cell, day.isoformat(),
        )

    async def cell_checked(self, cell, day) -> bool:
        row = await self._pool.fetchrow(
            "SELECT 1 FROM alert_cell_checks WHERE cell_key = $1 AND local_date = $2", cell, day
        )
        return row is not None

    async def record_cell_check(self, cell, day, score) -> None:
        await self._pool.execute(
            """
            INSERT INTO alert_cell_checks (cell_key, local_date, score) VALUES ($1, $2, $3)
            ON CONFLICT (cell_key, local_date) DO NOTHING
            """,
            cell, day, score,
        )
```

- [ ] **Step 4: Run tests**

Run: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -m pytest tests/test_subscription_store.py -v`
Expected: 3 pass, postgres test skipped. (If a local Postgres or the Neon DB exists, also run with `TEST_DATABASE_URL=... ` and expect 4 pass.)

- [ ] **Step 5: Commit**
```bash
git add backend/app/services/subscription_store.py backend/tests/test_subscription_store.py
git commit -m "feat(push): subscription store with in-memory and Postgres backends"
```

---

### Task 3: Push sender + AlertService

**Files:**
- Create: `backend/app/services/push_sender.py`
- Create: `backend/app/services/alert_service.py`
- Create: `backend/app/schemas/push.py` (only `AlertRunSummary` is needed here; the full file is written now so Task 4 can use it)
- Test: `backend/tests/test_alert_service.py`

**Interfaces:**
- Consumes: `StoredSubscription`, `SubscriptionStore`, `cell_key` (Task 2); `PredictionService.predict(..., warm_climatology=False)` (Task 1); `WeatherUnavailableError` from `app.services.weather_service`.
- Produces:
  - `SendResult = Literal["ok", "gone", "error"]`; `WebPushSender(private_key: str, subject: str)` with `async send(sub: StoredSubscription, payload: dict) -> SendResult`.
  - `AlertService(store, predictor, sunset_for, local_date_for, sender, lead_min_hours=3.5, lead_max_hours=4.5, decimals=1, clock=utcnow)` with `async run(force: bool = False) -> AlertRunSummary`.
  - `Predictor = Callable[[float, float, date], Awaitable[PredictResponse]]`.
  - `prediction_predictor(prediction_service) -> Predictor`.
  - `build_payload(place: dict, prediction: PredictResponse, tz: str, cell: str, day: date) -> dict` with keys `title, body, url, tag`.

- [ ] **Step 1: Write the schemas file**

`backend/app/schemas/push.py`:
```python
"""Request / response schemas for Web Push subscription and alert runs."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PushKeys(BaseModel):
    p256dh: str = Field(..., min_length=1, max_length=256)
    auth: str = Field(..., min_length=1, max_length=128)


class PushSubscriptionIn(BaseModel):
    endpoint: str = Field(..., pattern=r"^https://", max_length=2048)
    keys: PushKeys


class AlertPlace(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    name: str = Field(..., min_length=1, max_length=120)


class SubscribeRequest(BaseModel):
    subscription: PushSubscriptionIn
    places: list[AlertPlace] = Field(..., max_length=5)
    tz: str = Field("UTC", max_length=64, description="IANA time zone of the device")


class UnsubscribeRequest(BaseModel):
    endpoint: str = Field(..., max_length=2048)


class VapidKeyResponse(BaseModel):
    public_key: str


class AlertRunSummary(BaseModel):
    cells: int = 0
    cells_due: int = 0
    cells_checked: int = 0
    notifications_sent: int = 0
    pruned: int = 0
```

- [ ] **Step 2: Write the failing tests**

`backend/tests/test_alert_service.py`:
```python
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


def make_service(store, sender, *, score=85.0, category="Epic", lead_hours=4.0, fail=False):
    calls: list[tuple[float, float, date]] = []

    async def predictor(lat, lon, day):
        calls.append((lat, lon, day))
        if fail:
            raise WeatherUnavailableError("rate limited")
        return fake_prediction(score, category)

    svc = AlertService(
        store=store,
        predictor=predictor,
        sunset_for=lambda lat, lon, d: NOW + timedelta(hours=lead_hours),
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
    for lead in (5.0, 3.0, -1.0):
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
```

- [ ] **Step 3: Run to verify failure**

Run: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -m pytest tests/test_alert_service.py -v`
Expected: `ModuleNotFoundError: app.services.alert_service`.

- [ ] **Step 4: Implement the sender**

`backend/app/services/push_sender.py`:
```python
"""Thin async wrapper around pywebpush."""
from __future__ import annotations

import asyncio
import json
from typing import Literal

from app.core.logging import get_logger
from app.services.subscription_store import StoredSubscription

logger = get_logger(__name__)

SendResult = Literal["ok", "gone", "error"]

# A sunset alert is worthless after sunset; don't let push services hold it longer.
_TTL_SECONDS = 4 * 3600


class WebPushSender:
    def __init__(self, private_key: str, subject: str) -> None:
        self._private_key = private_key
        self._subject = subject

    async def send(self, sub: StoredSubscription, payload: dict) -> SendResult:
        from pywebpush import WebPushException, webpush

        try:
            await asyncio.to_thread(
                webpush,
                subscription_info={"endpoint": sub.endpoint, "keys": {"p256dh": sub.p256dh, "auth": sub.auth}},
                data=json.dumps(payload),
                vapid_private_key=self._private_key,
                # Fresh dict per call: pywebpush mutates the claims (adds aud/exp).
                vapid_claims={"sub": self._subject},
                ttl=_TTL_SECONDS,
            )
            return "ok"
        except WebPushException as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (404, 410):
                return "gone"
            logger.warning("Web push failed (status=%s) for %s…: %s", status, sub.endpoint[:60], exc)
            return "error"
        except Exception as exc:
            logger.warning("Web push error for %s…: %s", sub.endpoint[:60], exc)
            return "error"
```

- [ ] **Step 5: Implement the alert service**

`backend/app/services/alert_service.py`:
```python
"""Hourly Epic-sunset alert run.

COST MODEL
----------
Belled places are grouped into the same 0.1° cells as the weather cache. Each
cell is predicted at most once per local day, when its sunset is ~4 h away —
so Open-Meteo usage scales with distinct places, not with subscribers, and a
cell a user already looked at today is served from cache.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Awaitable, Callable, Protocol
from urllib.parse import quote
from zoneinfo import ZoneInfo

from app.core.logging import get_logger
from app.schemas.prediction import PredictRequest, PredictResponse
from app.schemas.push import AlertRunSummary
from app.services.subscription_store import StoredSubscription, SubscriptionStore, cell_key
from app.services.weather_service import WeatherUnavailableError
from app.utils.time_utils import utcnow

logger = get_logger(__name__)

Predictor = Callable[[float, float, date], Awaitable[PredictResponse]]

_WINDOW_OFFSETS_MIN = {"-15m": -15, "sunset": 0, "+15m": 15, "+30m": 30}


class Sender(Protocol):
    async def send(self, sub: StoredSubscription, payload: dict) -> str: ...


def prediction_predictor(prediction_service) -> Predictor:
    async def predict(lat: float, lon: float, day: date) -> PredictResponse:
        return await prediction_service.predict(
            PredictRequest(latitude=lat, longitude=lon, target_date=day),
            warm_climatology=False,
        )
    return predict


def build_payload(place: dict, prediction, tz: str, cell: str, day: date) -> dict:
    try:
        zone = ZoneInfo(tz)
    except Exception:
        zone = ZoneInfo("UTC")
    offset = _WINDOW_OFFSETS_MIN.get(prediction.best_window_point, 0)
    best = (prediction.sunset_time + timedelta(minutes=offset)).astimezone(zone)
    score = int(math.floor(prediction.beauty_score_0_100 + 0.5))  # matches the UI's Math.round
    name = place["name"]
    return {
        "title": "🔥 Epic sunset tonight",
        "body": f"{name} — {score}/100. Best around {best:%H:%M}.",
        "url": f"/?lat={place['latitude']}&lon={place['longitude']}&name={quote(name)}",
        "tag": f"epic-{cell}-{day.isoformat()}",
    }


@dataclass
class _Cell:
    lat: float
    lon: float
    members: list[tuple[StoredSubscription, dict]] = field(default_factory=list)


class AlertService:
    def __init__(
        self,
        store: SubscriptionStore,
        predictor: Predictor,
        sunset_for: Callable[[float, float, date], datetime],
        local_date_for: Callable[[float, float], date],
        sender: Sender,
        lead_min_hours: float = 3.5,
        lead_max_hours: float = 4.5,
        decimals: int = 1,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._store = store
        self._predict = predictor
        self._sunset_for = sunset_for
        self._local_date_for = local_date_for
        self._sender = sender
        self._lead_min = lead_min_hours
        self._lead_max = lead_max_hours
        self._decimals = decimals
        self._clock = clock

    def _group(self, subs: list[StoredSubscription]) -> dict[str, _Cell]:
        cells: dict[str, _Cell] = {}
        for sub in subs:
            seen: set[str] = set()
            for place in sub.places:
                key = cell_key(place["latitude"], place["longitude"], self._decimals)
                if key in seen:
                    continue  # one push per subscriber per cell
                seen.add(key)
                cell = cells.setdefault(key, _Cell(place["latitude"], place["longitude"]))
                cell.members.append((sub, place))
        return cells

    async def run(self, force: bool = False) -> AlertRunSummary:
        summary = AlertRunSummary()
        cells = self._group(await self._store.all())
        summary.cells = len(cells)
        now = self._clock()
        gone: set[str] = set()

        for key, cell in cells.items():
            day = self._local_date_for(cell.lat, cell.lon)
            if not force:
                lead = (self._sunset_for(cell.lat, cell.lon, day) - now).total_seconds() / 3600
                if not (self._lead_min <= lead < self._lead_max):
                    continue
                if await self._store.cell_checked(key, day):
                    continue
            summary.cells_due += 1

            try:
                prediction = await self._predict(cell.lat, cell.lon, day)
            except WeatherUnavailableError as exc:
                logger.warning("Alert check for cell %s deferred — weather unavailable: %s", key, exc)
                continue
            except Exception:
                logger.exception("Alert check for cell %s failed", key)
                continue
            summary.cells_checked += 1

            if not force:
                await self._store.record_cell_check(key, day, prediction.beauty_score_0_100)
                if prediction.category != "Epic":
                    continue

            for sub, place in cell.members:
                if sub.endpoint in gone:
                    continue
                if not force and sub.last_notified.get(key) == day.isoformat():
                    continue
                result = await self._sender.send(sub, build_payload(place, prediction, sub.tz, key, day))
                if result == "ok":
                    summary.notifications_sent += 1
                    await self._store.mark_notified(sub.endpoint, key, day)
                elif result == "gone":
                    gone.add(sub.endpoint)
                    await self._store.delete(sub.endpoint)
                    summary.pruned += 1

        logger.info("Alert run: %s", summary.model_dump())
        return summary
```

Note for `test_no_duplicate_notification_same_day`: the second run is skipped by `cell_checked` before dedupe is even consulted — both guards are intentional (cell check protects Open-Meteo, `last_notified` protects the user if a check row is ever lost).

- [ ] **Step 6: Run tests**

Run: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -m pytest tests/test_alert_service.py -v` → all PASS.

- [ ] **Step 7: Commit**
```bash
git add backend/app/schemas/push.py backend/app/services/push_sender.py backend/app/services/alert_service.py backend/tests/test_alert_service.py
git commit -m "feat(push): alert service — one prediction per cell per day, Epic-only pushes"
```

---

### Task 4: Push API endpoints and app wiring

**Files:**
- Create: `backend/app/api/push.py`
- Modify: `backend/app/main.py` (imports; lifespan after `prediction_service` is built; shutdown; `create_app` router list)
- Test: `backend/tests/test_push_api.py`

**Interfaces:**
- Consumes: schemas (Task 3), `InMemorySubscriptionStore`/`PostgresSubscriptionStore` (Task 2), `AlertService`, `prediction_predictor`, `WebPushSender` (Task 3).
- Produces HTTP: `GET /push/vapid-key` → `{"public_key"}`; `POST /push/subscribe` (204); `DELETE /push/subscribe` (204); `POST /internal/alerts/run?force=0|1` with header `X-Alerts-Secret` → `AlertRunSummary`. `app.state.subscription_store`, `app.state.alert_service` (either may be `None`).

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_push_api.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -m pytest tests/test_push_api.py -v`
Expected: 404s / failures (routes don't exist).

- [ ] **Step 3: Implement the router**

`backend/app/api/push.py`:
```python
"""Web Push subscription endpoints and the secret-protected hourly alert run."""
from __future__ import annotations

import hmac
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request, Response

from app.schemas.push import (
    AlertRunSummary,
    SubscribeRequest,
    UnsubscribeRequest,
    VapidKeyResponse,
)

router = APIRouter(tags=["push"])

_DISABLED = "Sunset alerts are not configured on this server."


def _store(request: Request):
    store = getattr(request.app.state, "subscription_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail=_DISABLED)
    return store


@router.get("/push/vapid-key", response_model=VapidKeyResponse)
async def vapid_key(request: Request) -> VapidKeyResponse:
    _store(request)
    key = request.app.state.settings.VAPID_PUBLIC_KEY
    if not key:
        raise HTTPException(status_code=503, detail=_DISABLED)
    return VapidKeyResponse(public_key=key)


@router.post("/push/subscribe", status_code=204)
async def subscribe(body: SubscribeRequest, request: Request) -> Response:
    await _store(request).upsert(
        body.subscription.endpoint,
        body.subscription.keys.p256dh,
        body.subscription.keys.auth,
        [p.model_dump() for p in body.places],
        body.tz,
    )
    return Response(status_code=204)


@router.delete("/push/subscribe", status_code=204)
async def unsubscribe(body: UnsubscribeRequest, request: Request) -> Response:
    await _store(request).delete(body.endpoint)
    return Response(status_code=204)


@router.post("/internal/alerts/run", response_model=AlertRunSummary, include_in_schema=False)
async def run_alerts(
    request: Request,
    force: bool = False,
    x_alerts_secret: Optional[str] = Header(default=None),
) -> AlertRunSummary:
    secret = request.app.state.settings.ALERTS_SECRET
    if not secret or not hmac.compare_digest(x_alerts_secret or "", secret):
        raise HTTPException(status_code=401, detail="Unauthorized")
    service = getattr(request.app.state, "alert_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail=_DISABLED)
    return await service.run(force=force)
```

- [ ] **Step 4: Wire into `main.py`**

Imports — change the api import line to include `push`, and add:
```python
from app.api import health, predict, forecast, heatmap, model_info, geocode, submit, rate, push
from app.services.alert_service import AlertService, prediction_predictor
from app.services.push_sender import WebPushSender
from app.services.subscription_store import PostgresSubscriptionStore
from app.utils.time_utils import local_sunset_date
```
In `lifespan`, directly after the `prediction_service = PredictionService(...)` block:
```python
    # Epic-sunset push alerts — optional. Without a database or VAPID key the
    # app runs exactly as before and the /push endpoints answer 503.
    subscription_store = None
    alert_service = None
    if settings.DATABASE_URL:
        try:
            subscription_store = await PostgresSubscriptionStore.connect(settings.DATABASE_URL)
        except Exception as exc:
            logger.error("Push alerts disabled — could not connect to DATABASE_URL: %s", exc)
    if subscription_store is not None and settings.VAPID_PRIVATE_KEY:
        alert_service = AlertService(
            store=subscription_store,
            predictor=prediction_predictor(prediction_service),
            sunset_for=astro_service.get_sunset_time,
            local_date_for=local_sunset_date,
            sender=WebPushSender(settings.VAPID_PRIVATE_KEY, settings.VAPID_SUBJECT),
            lead_min_hours=settings.ALERT_LEAD_MIN_HOURS,
            lead_max_hours=settings.ALERT_LEAD_MAX_HOURS,
            decimals=settings.CACHE_COORD_DECIMALS,
        )
    logger.info(
        "Push alerts: store=%s, sender=%s",
        "postgres" if subscription_store else "off",
        "on" if alert_service else "off",
    )
```
With the other `app.state` assignments:
```python
    app.state.subscription_store = subscription_store
    app.state.alert_service = alert_service
```
After `await http_client.aclose()`:
```python
    if subscription_store is not None:
        await subscription_store.close()
```
In `create_app`, after `app.include_router(rate.router)`:
```python
    app.include_router(push.router)
```

- [ ] **Step 5: Run tests**

Run: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -m pytest tests/test_push_api.py -v` → PASS.
Run full suite: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -m pytest -q` → only the known pre-existing failure.

- [ ] **Step 6: Commit**
```bash
git add backend/app/api/push.py backend/app/main.py backend/tests/test_push_api.py
git commit -m "feat(push): subscribe/unsubscribe endpoints and secret-protected alert run"
```

---

### Task 5: VAPID key script + hourly GitHub Actions cron

**Files:**
- Create: `backend/scripts/generate_vapid_keys.py`
- Create: `.github/workflows/sunset-alerts.yml`

**Interfaces:**
- Consumes: `POST /internal/alerts/run` (Task 4).
- Produces: repo secrets contract `ALERTS_API_URL` (e.g. `https://sunset-predictor-b8ig.onrender.com`), `ALERTS_SECRET`.

- [ ] **Step 1: Write the key generator**

`backend/scripts/generate_vapid_keys.py`:
```python
"""Print a fresh VAPID key pair in the base64url form pywebpush and browsers expect.

Usage: python scripts/generate_vapid_keys.py
Put VAPID_PUBLIC_KEY / VAPID_PRIVATE_KEY in Render's env (and backend/.env for local).
"""
from cryptography.hazmat.primitives import serialization
from py_vapid import Vapid01
from py_vapid.utils import b64urlencode

v = Vapid01()
v.generate_keys()
private = b64urlencode(v.private_key.private_numbers().private_value.to_bytes(32, "big"))
public = b64urlencode(
    v.public_key.public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
)
print(f"VAPID_PUBLIC_KEY={public}")
print(f"VAPID_PRIVATE_KEY={private}")
```

- [ ] **Step 2: Verify the keys work with pywebpush**

Run (from `backend/`):
```bash
/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python scripts/generate_vapid_keys.py
```
Expected: two lines; public key is 87 chars, private key 43 chars. Then sanity-check the private key loads:
```bash
/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -c "from py_vapid import Vapid01; import subprocess,sys; out=subprocess.check_output([sys.executable,'scripts/generate_vapid_keys.py'],text=True); k=out.split('VAPID_PRIVATE_KEY=')[1].strip(); Vapid01.from_string(k); print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Write the workflow**

`.github/workflows/sunset-alerts.yml`:
```yaml
name: Sunset alerts

# Hourly nudge to the backend. Each cell is only checked when its sunset is
# ~4 h away, and at most once per day, so most runs cost zero Open-Meteo calls.
on:
  schedule:
    - cron: "7 * * * *"
  workflow_dispatch:
    inputs:
      force:
        description: "Force: ignore window, threshold and dedupe (testing only)"
        type: boolean
        default: false

jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - name: Trigger alert run
        env:
          API: ${{ secrets.ALERTS_API_URL }}
          SECRET: ${{ secrets.ALERTS_SECRET }}
          FORCE: ${{ inputs.force && '1' || '0' }}
        run: |
          curl -fsS --retry 3 --retry-delay 20 --max-time 120 \
            -X POST -H "X-Alerts-Secret: $SECRET" \
            "$API/internal/alerts/run?force=$FORCE"
```

- [ ] **Step 4: Commit**
```bash
git add backend/scripts/generate_vapid_keys.py .github/workflows/sunset-alerts.yml
git commit -m "feat(push): VAPID key generator and hourly alert cron workflow"
```

---

### Task 6: Manifest, maskable icon, service worker, registration

**Files:**
- Modify: `frontend/src/app/manifest.ts`
- Create: `frontend/public/icon-maskable-512.png`
- Create: `frontend/public/sw.js`
- Create: `frontend/src/components/ServiceWorkerRegistrar.tsx`
- Modify: `frontend/src/app/layout.tsx`

**Interfaces:**
- Produces: service worker at `/sw.js` scope `/`; handles push payload `{title, body, url, tag}` (Task 3 `build_payload`).

Dev server note: the shell has `NODE_ENV=production` globally, which breaks `next dev`. Add `frontend` to `.claude/launch.json` (create if missing) as:
```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "frontend",
      "runtimeExecutable": "env",
      "runtimeArgs": ["-u", "NODE_ENV", "npx", "--prefix", "frontend", "next", "dev", "frontend"],
      "port": 3000
    }
  ]
}
```
and start it with the preview tool (`preview_start {name: "frontend"}`), never via Bash.

- [ ] **Step 1: Generate the maskable icon** (logo at 80 % on the brand background, so Android's mask never crops it)

Run from repo root:
```bash
/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python - <<'EOF'
from PIL import Image
src = Image.open("frontend/public/icon-512.png").convert("RGBA")
canvas = Image.new("RGBA", (512, 512), (4, 5, 10, 255))
inner = src.resize((410, 410), Image.LANCZOS)
canvas.alpha_composite(inner, ((512 - 410) // 2, (512 - 410) // 2))
canvas.convert("RGB").save("frontend/public/icon-maskable-512.png")
EOF
```
If PIL is missing: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/pip install pillow` (dev-only, not added to requirements). Open the PNG with Read to eyeball it.

- [ ] **Step 2: Update the manifest**

Replace the returned object in `manifest.ts` with:
```ts
  return {
    id: "/",
    name: "Afterglow",
    short_name: "Afterglow",
    description: "How beautiful will tonight's sunset be? Get a score, reasons, and the best time to watch.",
    start_url: "/",
    scope: "/",
    display: "standalone",
    orientation: "portrait",
    background_color: "#04050A",
    theme_color: "#04050A",
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
      { src: "/icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
```

- [ ] **Step 3: Write the service worker**

`frontend/public/sw.js`:
```js
/* Afterglow service worker — push notifications only (no offline caching;
   localStorage already paints the last reading instantly). */

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("push", (event) => {
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch {
    data = { body: event.data ? event.data.text() : "" };
  }
  const title = data.title || "Afterglow";
  event.waitUntil(
    self.registration.showNotification(title, {
      body: data.body || "",
      icon: "/icon-192.png",
      badge: "/icon-192.png",
      tag: data.tag,
      data: { url: data.url || "/" },
    })
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = new URL(event.notification.data?.url || "/", self.location.origin).href;
  event.waitUntil(
    (async () => {
      const windows = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
      for (const client of windows) {
        if ("navigate" in client) {
          await client.focus();
          return client.navigate(url);
        }
      }
      return self.clients.openWindow(url);
    })()
  );
});
```

- [ ] **Step 4: Register it**

`frontend/src/components/ServiceWorkerRegistrar.tsx`:
```tsx
"use client";

import { useEffect } from "react";

/** Registers /sw.js once per page load. Renders nothing. */
export default function ServiceWorkerRegistrar() {
  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;
    navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => {
      // Push is a nice-to-have; never surface SW failures to the user.
    });
  }, []);
  return null;
}
```
In `layout.tsx`, import it and render it as the first child inside `<ThemeProvider …>`:
```tsx
import ServiceWorkerRegistrar from "@/components/ServiceWorkerRegistrar";
```
```tsx
          <ServiceWorkerRegistrar />
```
Also add `capable: true` to `appleWebApp` in `metadata` (emits `apple-mobile-web-app-capable`, needed for standalone launch on older iOS).

- [ ] **Step 5: Verify**

Run: `cd frontend && env -u NODE_ENV npx tsc --noEmit` → no errors.
Start the preview (`preview_start {name: "frontend"}`), then with `javascript_tool`:
```js
(await navigator.serviceWorker.getRegistration())?.active?.scriptURL
```
Expected: `"http://localhost:3000/sw.js"`. And:
```js
(await fetch('/manifest.webmanifest').then(r => r.json())).icons.length
```
Expected: `3`.

- [ ] **Step 6: Commit**
```bash
git add frontend/src/app/manifest.ts frontend/public/icon-maskable-512.png frontend/public/sw.js frontend/src/components/ServiceWorkerRegistrar.tsx frontend/src/app/layout.tsx .claude/launch.json
git commit -m "feat(pwa): service worker, maskable icon, standalone manifest"
```

---

### Task 7: Client libraries — install state, push, API

**Files:**
- Create: `frontend/src/lib/install.ts`
- Create: `frontend/src/lib/push.ts`
- Modify: `frontend/src/lib/api.ts` (append three functions)

**Interfaces:**
- Consumes: HTTP endpoints from Task 4; `LocationState`, `sameLocation`, `MAX_SAVED_PLACES` from `lib/storage.ts`/`lib/types.ts`.
- Produces:
  - `install.ts`: `type Platform = "ios" | "android" | "desktop" | "unsupported"`; `detectPlatform(): Platform`; `isStandalone(): boolean`; `pushSupported(): boolean`; `recordVisit(): number` (returns visit count incl. this one; counts once per session); `type PromptKind = "install" | "alerts"`; `canShowPrompt(kind): boolean`; `dismissPrompt(kind): void`; `captureInstallEvent(): void`; `hasInstallEvent(): boolean`; `triggerInstall(): Promise<boolean>`; `onInstallEvent(cb: () => void): () => void`.
  - `push.ts`: `permission(): NotificationPermission | "unsupported"`; `loadAlertPlaces(): LocationState[]`; `isAlertOn(place): boolean`; `enableAlerts(place): Promise<boolean>`; `setAlert(place, on): Promise<LocationState[]>`; `alertsAvailable(): Promise<boolean>`.
  - `api.ts`: `getVapidKey(): Promise<string>`; `subscribePush(body: {subscription: PushSubscriptionJSON; places: LocationState[]; tz: string}): Promise<void>`; `unsubscribePush(endpoint: string): Promise<void>`.

No frontend unit-test framework exists in this repo; verification is `tsc` plus browser checks in Task 8/9. Keep these modules small and pure where possible.

- [ ] **Step 1: API additions**

The existing `request<T>` helper calls `res.json()`, which throws on 204. Add a sibling helper and the three functions at the end of `api.ts`:
```ts
// ---------------------------------------------------------------------------
// Push alerts
// ---------------------------------------------------------------------------

async function send(url: string, options: RequestInit): Promise<void> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
}

/** Public VAPID key; throws (503) when alerts aren't configured on the server. */
export async function getVapidKey(): Promise<string> {
  const { public_key } = await request<{ public_key: string }>(`${API_BASE}/push/vapid-key`);
  return public_key;
}

export async function subscribePush(body: {
  subscription: PushSubscriptionJSON;
  places: LocationState[];
  tz: string;
}): Promise<void> {
  await send(`${API_BASE}/push/subscribe`, {
    method: "POST",
    body: JSON.stringify({
      subscription: body.subscription,
      places: body.places.map(({ latitude, longitude, name }) => ({ latitude, longitude, name })),
      tz: body.tz,
    }),
  });
}

export async function unsubscribePush(endpoint: string): Promise<void> {
  await send(`${API_BASE}/push/subscribe`, {
    method: "DELETE",
    body: JSON.stringify({ endpoint }),
  });
}
```
Add `LocationState` to the `import type { … } from "./types"` list.

- [ ] **Step 2: `install.ts`**

```ts
/**
 * Home-Screen install state.
 *
 * iOS only delivers web push to Home-Screen apps, so on iPhone the install
 * pitch IS the alerts pitch. Everything here is best-effort: storage may be
 * unavailable and every browser exposes a different subset of these APIs.
 */

export type Platform = "ios" | "android" | "desktop" | "unsupported";
export type PromptKind = "install" | "alerts";

const VISITS_KEY = "afterglow:visits";
const SESSION_KEY = "afterglow:visitCounted";
const promptKey = (kind: PromptKind) => `afterglow:prompt:${kind}`;

const SNOOZE_MS = 14 * 24 * 60 * 60 * 1000;
const MAX_DISMISSALS = 3;

interface PromptState {
  dismissals: number;
  snoozedUntil: number;
}

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // storage unavailable — prompts will simply reappear next visit
  }
}

/** In-app browsers (Instagram, Facebook, WhatsApp, …) can't install. */
function isInAppBrowser(ua: string): boolean {
  return /FBAN|FBAV|Instagram|WhatsApp|Line\/|Twitter|LinkedInApp|Snapchat|TikTok/i.test(ua);
}

function iosVersion(ua: string): number | null {
  const m = ua.match(/OS (\d+)_(\d+)/);
  return m ? Number(m[1]) + Number(m[2]) / 100 : null;
}

export function detectPlatform(): Platform {
  if (typeof navigator === "undefined") return "unsupported";
  const ua = navigator.userAgent;
  if (isInAppBrowser(ua)) return "unsupported";
  const isIos =
    /iPhone|iPad|iPod/.test(ua) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1); // iPadOS
  if (isIos) {
    const v = iosVersion(ua);
    return v !== null && v < 16.04 ? "unsupported" : "ios"; // web push needs 16.4
  }
  if (/Android/i.test(ua)) return "android";
  return "desktop";
}

export function isStandalone(): boolean {
  if (typeof window === "undefined") return false;
  return (
    window.matchMedia?.("(display-mode: standalone)").matches ||
    (navigator as Navigator & { standalone?: boolean }).standalone === true
  );
}

export function pushSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    "serviceWorker" in navigator &&
    "PushManager" in window &&
    "Notification" in window
  );
}

/** Counts at most once per browser session; returns the running total. */
export function recordVisit(): number {
  let visits = read<number>(VISITS_KEY, 0);
  try {
    if (!sessionStorage.getItem(SESSION_KEY)) {
      sessionStorage.setItem(SESSION_KEY, "1");
      visits += 1;
      write(VISITS_KEY, visits);
    }
  } catch {
    // no sessionStorage — leave the count alone
  }
  return visits;
}

export function canShowPrompt(kind: PromptKind): boolean {
  const s = read<PromptState>(promptKey(kind), { dismissals: 0, snoozedUntil: 0 });
  return s.dismissals < MAX_DISMISSALS && Date.now() >= s.snoozedUntil;
}

export function dismissPrompt(kind: PromptKind): void {
  const s = read<PromptState>(promptKey(kind), { dismissals: 0, snoozedUntil: 0 });
  write(promptKey(kind), { dismissals: s.dismissals + 1, snoozedUntil: Date.now() + SNOOZE_MS });
}

// ── Android / Chromium native install prompt ─────────────────────────────────

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

let deferred: BeforeInstallPromptEvent | null = null;
const listeners = new Set<() => void>();
let captured = false;

/** Call once on mount; the event fires early and only once per page load. */
export function captureInstallEvent(): void {
  if (captured || typeof window === "undefined") return;
  captured = true;
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferred = e as BeforeInstallPromptEvent;
    listeners.forEach((cb) => cb());
  });
}

export function hasInstallEvent(): boolean {
  return deferred !== null;
}

export function onInstallEvent(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export async function triggerInstall(): Promise<boolean> {
  if (!deferred) return false;
  await deferred.prompt();
  const { outcome } = await deferred.userChoice;
  deferred = null;
  return outcome === "accepted";
}
```

- [ ] **Step 3: `push.ts`**

```ts
/**
 * Web Push opt-in and per-place alert bells.
 *
 * The belled places are mirrored in localStorage so the bells render instantly;
 * the backend copy (keyed by the push endpoint) is what the hourly alert run reads.
 */

import { getVapidKey, subscribePush, unsubscribePush } from "./api";
import { MAX_SAVED_PLACES, sameLocation } from "./storage";
import { pushSupported } from "./install";
import type { LocationState } from "./types";

const ALERT_PLACES_KEY = "afterglow:alertPlaces";

export function permission(): NotificationPermission | "unsupported" {
  return pushSupported() ? Notification.permission : "unsupported";
}

export function loadAlertPlaces(): LocationState[] {
  try {
    const raw = localStorage.getItem(ALERT_PLACES_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveAlertPlaces(places: LocationState[]): void {
  try {
    localStorage.setItem(ALERT_PLACES_KEY, JSON.stringify(places));
  } catch {
    // ignore
  }
}

export function isAlertOn(place: LocationState): boolean {
  return loadAlertPlaces().some((p) => sameLocation(p, place));
}

let vapidCheck: Promise<boolean> | null = null;

/** Whether the server has alerts configured (cached for the page's lifetime). */
export function alertsAvailable(): Promise<boolean> {
  if (!pushSupported()) return Promise.resolve(false);
  vapidCheck ??= getVapidKey().then(() => true, () => false);
  return vapidCheck;
}

function urlBase64ToUint8Array(base64: string): Uint8Array {
  const padded = (base64 + "=".repeat((4 - (base64.length % 4)) % 4)).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(padded);
  return Uint8Array.from(raw, (c) => c.charCodeAt(0));
}

async function currentSubscription(create: boolean): Promise<PushSubscription | null> {
  const reg = await navigator.serviceWorker.ready;
  const existing = await reg.pushManager.getSubscription();
  if (existing || !create) return existing;
  return reg.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: urlBase64ToUint8Array(await getVapidKey()),
  });
}

async function sync(places: LocationState[]): Promise<void> {
  if (places.length === 0) {
    const sub = await currentSubscription(false);
    if (sub) {
      await unsubscribePush(sub.endpoint);
      await sub.unsubscribe();
    }
    return;
  }
  const sub = await currentSubscription(true);
  if (!sub) return;
  await subscribePush({
    subscription: sub.toJSON(),
    places,
    tz: Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC",
  });
}

/**
 * Ask for permission (MUST be called from a tap handler — iOS requires a user
 * gesture) and turn the bell on for *place*. Returns false if not granted.
 */
export async function enableAlerts(place: LocationState): Promise<boolean> {
  if (!pushSupported()) return false;
  const result = await Notification.requestPermission();
  if (result !== "granted") return false;
  await setAlert(place, true);
  return true;
}

/** Toggle one place's bell and push the full list to the backend. */
export async function setAlert(place: LocationState, on: boolean): Promise<LocationState[]> {
  const others = loadAlertPlaces().filter((p) => !sameLocation(p, place));
  const next = (on ? [place, ...others] : others).slice(0, MAX_SAVED_PLACES);
  await sync(next);
  saveAlertPlaces(next);
  return next;
}
```

- [ ] **Step 4: Verify types**

Run: `cd frontend && env -u NODE_ENV npx tsc --noEmit` → no errors. (If `Uint8Array` is rejected as `applicationServerKey` by the TS DOM lib version, cast: `applicationServerKey: urlBase64ToUint8Array(...) as BufferSource`.)

- [ ] **Step 5: Commit**
```bash
git add frontend/src/lib/install.ts frontend/src/lib/push.ts frontend/src/lib/api.ts
git commit -m "feat(pwa): client install-state and push subscription helpers"
```

---

### Task 8: Install / alerts card and iOS guide sheet on the home page

**Files:**
- Create: `frontend/src/components/IosInstallSheet.tsx`
- Create: `frontend/src/components/InstallPrompt.tsx`
- Modify: `frontend/src/app/page.tsx` (render under `VerdictCard`)

**Interfaces:**
- Consumes: everything in `install.ts` and `push.ts` (Task 7).
- Produces: `<InstallPrompt location={LocationState} onAlertsChanged={() => void} />`; `<IosInstallSheet open onClose />`.

- [ ] **Step 1: iOS guide sheet**

`frontend/src/components/IosInstallSheet.tsx`:
```tsx
"use client";

import { useEffect } from "react";
import { PlusSquare, Share, X } from "lucide-react";

interface Props {
  open: boolean;
  onClose: () => void;
}

/** Safari has no install prompt, so we show the two taps it takes. */
export default function IosInstallSheet({ open, onClose }: Props) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const steps = [
    { icon: <Share size={18} />, text: <>Tap <strong>Share</strong> in Safari&apos;s toolbar</> },
    { icon: <PlusSquare size={18} />, text: <>Choose <strong>Add to Home Screen</strong></> },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center">
      <button
        aria-label="Close"
        onClick={onClose}
        className="absolute inset-0 bg-slate-900/30 dark:bg-slate-950/60"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Add Afterglow to your Home Screen"
        className="relative w-full max-w-2xl bg-white dark:bg-slate-900 rounded-t-3xl border-t border-x border-gray-200 dark:border-slate-700/50 px-5 pt-4 flex flex-col gap-4 shadow-2xl animate-slide-up"
        style={{ paddingBottom: "max(2rem, env(safe-area-inset-bottom))" }}
      >
        <div className="flex items-center gap-3">
          <h2 className="flex-1 text-lg font-bold tracking-tight text-gray-900 dark:text-white">
            Add to Home Screen
          </h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="w-11 h-11 rounded-full flex items-center justify-center text-gray-600 dark:text-slate-400 hover:bg-gray-100 dark:hover:bg-slate-800"
          >
            <X size={16} />
          </button>
        </div>
        <ol className="flex flex-col gap-3">
          {steps.map((s, i) => (
            <li key={i} className="flex items-center gap-3 text-sm text-gray-800 dark:text-slate-200">
              <span className="w-9 h-9 flex-shrink-0 rounded-xl flex items-center justify-center bg-orange-50 dark:bg-orange-500/10 text-orange-600 dark:text-orange-400">
                {s.icon}
              </span>
              <span>
                <span className="text-gray-500 dark:text-slate-500 mr-1">{i + 1}.</span>
                {s.text}
              </span>
            </li>
          ))}
        </ol>
        <p className="text-xs text-gray-600 dark:text-slate-400">
          Then open Afterglow from your Home Screen and turn on Epic sunset alerts.
        </p>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: The card**

`frontend/src/components/InstallPrompt.tsx`:
```tsx
"use client";

import { useEffect, useState } from "react";
import { Bell, Smartphone, X } from "lucide-react";
import type { LocationState } from "@/lib/types";
import {
  canShowPrompt,
  captureInstallEvent,
  detectPlatform,
  dismissPrompt,
  hasInstallEvent,
  isStandalone,
  onInstallEvent,
  recordVisit,
  triggerInstall,
  type PromptKind,
} from "@/lib/install";
import { alertsAvailable, enableAlerts, loadAlertPlaces, permission } from "@/lib/push";
import IosInstallSheet from "./IosInstallSheet";

interface Props {
  location: LocationState;
  onAlertsChanged?: () => void;
}

/**
 * One quiet card under the verdict. Browser tab → "add to Home Screen";
 * installed app → "turn on Epic alerts". Never on a first visit, never above
 * the answer, snoozed for two weeks when dismissed.
 */
export default function InstallPrompt({ location, onAlertsChanged }: Props) {
  const [kind, setKind] = useState<PromptKind | null>(null);
  const [iosSheet, setIosSheet] = useState(false);
  const [busy, setBusy] = useState(false);
  const [denied, setDenied] = useState(false);

  useEffect(() => {
    captureInstallEvent();
    const visits = recordVisit();
    const platform = detectPlatform();
    let cancelled = false;

    const decide = async () => {
      if (platform === "unsupported") return setKind(null);

      if (isStandalone()) {
        const wantsAlerts =
          permission() === "default" &&
          loadAlertPlaces().length === 0 &&
          canShowPrompt("alerts") &&
          (await alertsAvailable());
        if (!cancelled) setKind(wantsAlerts ? "alerts" : null);
        return;
      }

      const installable = platform === "ios" || hasInstallEvent();
      if (!cancelled) setKind(visits >= 2 && installable && canShowPrompt("install") ? "install" : null);
    };

    void decide();
    // Chromium may fire beforeinstallprompt after mount.
    const off = onInstallEvent(() => void decide());
    return () => {
      cancelled = true;
      off();
    };
  }, []);

  if (!kind) return null;

  const dismiss = () => {
    dismissPrompt(kind);
    setKind(null);
  };

  const act = async () => {
    if (kind === "install") {
      if (detectPlatform() === "ios") return setIosSheet(true);
      const accepted = await triggerInstall();
      if (accepted) setKind(null);
      return;
    }
    setBusy(true);
    try {
      const ok = await enableAlerts(location);
      if (ok) {
        setKind(null);
        onAlertsChanged?.();
      } else {
        setDenied(permission() === "denied");
      }
    } catch {
      setDenied(false);
    } finally {
      setBusy(false);
    }
  };

  const isInstall = kind === "install";

  return (
    <>
      <div className="relative flex items-start gap-3 p-4 rounded-2xl bg-gradient-to-br from-orange-50 to-rose-50 dark:from-orange-500/10 dark:to-rose-500/10 border border-orange-200/70 dark:border-orange-500/20">
        <span className="w-10 h-10 flex-shrink-0 rounded-xl flex items-center justify-center bg-white/80 dark:bg-slate-900/60 text-orange-600 dark:text-orange-400">
          {isInstall ? <Smartphone size={18} /> : <Bell size={18} />}
        </span>
        <div className="flex-1 min-w-0 pr-6">
          <p className="text-sm font-semibold text-gray-900 dark:text-white">
            {isInstall ? "Never miss an epic sunset" : `Epic sunset alerts for ${location.name}`}
          </p>
          <p className="mt-0.5 text-xs text-gray-700 dark:text-slate-300">
            {isInstall
              ? "Add Afterglow to your Home Screen and we'll ping you when one's coming."
              : denied
                ? "Notifications are blocked — enable them for Afterglow in Settings."
                : "One ping, about 4 hours before sunset, only when it's going to be Epic."}
          </p>
          {!denied && (
            <button
              onClick={act}
              disabled={busy}
              className="mt-3 inline-flex items-center min-h-[40px] px-4 rounded-full text-sm font-semibold text-white bg-orange-600 hover:bg-orange-700 disabled:opacity-60 transition-colors"
            >
              {isInstall ? "Add to Home Screen" : busy ? "Turning on…" : "Turn on alerts"}
            </button>
          )}
        </div>
        <button
          onClick={dismiss}
          aria-label="Not now"
          className="absolute top-2 right-2 w-9 h-9 rounded-full flex items-center justify-center text-gray-500 dark:text-slate-400 hover:bg-white/60 dark:hover:bg-slate-800/60"
        >
          <X size={15} />
        </button>
      </div>
      <IosInstallSheet open={iosSheet} onClose={() => setIosSheet(false)} />
    </>
  );
}
```

- [ ] **Step 3: Place it on the home page**

In `page.tsx`, import:
```tsx
import InstallPrompt from "@/components/InstallPrompt";
```
Add state for bell refresh (used by Task 9):
```tsx
  const [alertsVersion, setAlertsVersion] = useState(0);
```
Directly after `<VerdictCard prediction={prediction} targetDate={selectedDate} />`:
```tsx
          <InstallPrompt
            location={location}
            onAlertsChanged={() => setAlertsVersion((v) => v + 1)}
          />
```

- [ ] **Step 4: Verify in the browser**

1. `env -u NODE_ENV npx tsc --noEmit` (in `frontend/`) → clean.
2. Start preview (`frontend`) and the backend (`cd backend && /Users/yotamtsabari/sunset-predictor/backend/.venv/bin/uvicorn app.main:app --reload --port 8000`, run in background). First load: card absent (`find "Never miss"` → no match).
3. Simulate a second visit: `javascript_tool` → `sessionStorage.clear(); location.reload()`. With desktop Chrome, the install card appears only if `beforeinstallprompt` fired; to exercise the iOS path use `resize_window {preset: "mobile"}` (Android UA) — or override UA isn't possible, so verify the iOS sheet by temporarily calling in console: `localStorage.setItem('afterglow:visits','5')` and checking `find "Add to Home Screen"` on an iPhone later in Task 10.
4. Click the X → reload with fresh session → card gone; `localStorage.getItem('afterglow:prompt:install')` shows `dismissals: 1`.
5. Screenshot the card in light and dark (`resize_window {colorScheme: "dark"}`).

- [ ] **Step 5: Commit**
```bash
git add frontend/src/components/InstallPrompt.tsx frontend/src/components/IosInstallSheet.tsx frontend/src/app/page.tsx
git commit -m "feat(pwa): install and alerts opt-in card under the verdict"
```

---

### Task 9: Per-place bells in the location sheet

**Files:**
- Modify: `frontend/src/components/LocationSheet.tsx`
- Modify: `frontend/src/app/page.tsx` (pass `alertsVersion`)

**Interfaces:**
- Consumes: `isAlertOn`, `setAlert`, `enableAlerts`, `permission`, `alertsAvailable` (Task 7); `isStandalone` (Task 7); `alertsVersion` state (Task 8).
- Produces: `LocationSheet` prop `alertsVersion?: number`.

- [ ] **Step 1: Implement the bells**

In `LocationSheet.tsx`:
- imports: `import { useEffect, useState } from "react";`, add `Bell, BellOff` to the lucide import, and
```tsx
import { isStandalone } from "@/lib/install";
import { alertsAvailable, enableAlerts, isAlertOn, permission, setAlert } from "@/lib/push";
```
- add prop `alertsVersion?: number` to `LocationSheetProps` and destructure it.
- inside the component, before `if (!open) return null;`:
```tsx
  const [bellsVisible, setBellsVisible] = useState(false);
  const [blocked, setBlocked] = useState(false);
  const [, force] = useState(0);
  const [pending, setPending] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    const perm = permission();
    setBlocked(perm === "denied");
    // Bells only make sense where a push can actually arrive: installed app,
    // or a browser that already granted permission (desktop/Android).
    if (perm === "unsupported" || (!isStandalone() && perm !== "granted")) {
      setBellsVisible(false);
      return;
    }
    let cancelled = false;
    alertsAvailable().then((ok) => !cancelled && setBellsVisible(ok));
    return () => {
      cancelled = true;
    };
  }, [open, alertsVersion]);

  const toggleBell = async (place: LocationState) => {
    const key = `${place.latitude},${place.longitude}`;
    setPending(key);
    try {
      if (permission() === "default") {
        await enableAlerts(place);
      } else {
        await setAlert(place, !isAlertOn(place));
      }
    } catch {
      // network failure — bell simply stays as it was
    } finally {
      setBlocked(permission() === "denied");
      setPending(null);
      force((n) => n + 1);
    }
  };
```
- Change each place row from a single `<button>` to a flex row containing the select button and a bell button. Replace the `places.map(...)` body with:
```tsx
            {places.map((place) => {
              const isCurrent = sameLocation(place, current);
              const key = `${place.latitude},${place.longitude}`;
              const on = isAlertOn(place);
              return (
                <div key={key} className="flex items-center gap-2">
                  <button
                    onClick={() => handleSelect(place)}
                    className={`flex-1 min-w-0 flex items-center gap-3 px-3.5 py-3 rounded-xl border text-left transition-colors ${
                      isCurrent
                        ? "bg-orange-50 dark:bg-orange-500/10 border-orange-500/50"
                        : "bg-white dark:bg-slate-800/40 border-gray-200 dark:border-slate-700/50 hover:border-orange-500/40"
                    }`}
                  >
                    <span className="flex-1 min-w-0 truncate text-sm font-medium text-gray-900 dark:text-white">
                      {place.name}
                    </span>
                    {isCurrent && (
                      <Check size={15} className="flex-shrink-0 text-orange-600 dark:text-orange-400" />
                    )}
                  </button>
                  {bellsVisible && (
                    <button
                      onClick={() => toggleBell(place)}
                      disabled={blocked || pending === key}
                      aria-pressed={on}
                      aria-label={on ? `Turn off Epic alerts for ${place.name}` : `Turn on Epic alerts for ${place.name}`}
                      className={`w-11 h-11 flex-shrink-0 rounded-xl border flex items-center justify-center transition-colors disabled:opacity-50 ${
                        on
                          ? "bg-orange-600 border-orange-600 text-white"
                          : "bg-white dark:bg-slate-800/40 border-gray-200 dark:border-slate-700/50 text-gray-500 dark:text-slate-400 hover:text-orange-600 dark:hover:text-orange-400"
                      }`}
                    >
                      {on ? <Bell size={16} /> : <BellOff size={16} />}
                    </button>
                  )}
                </div>
              );
            })}
```
- Under the "Recent" heading row, when `bellsVisible`, show one line of help:
```tsx
            {bellsVisible && (
              <p className="text-xs text-gray-600 dark:text-slate-400">
                {blocked
                  ? "Notifications are blocked — enable them for Afterglow in Settings."
                  : "Tap a bell to get pinged ~4 h before an Epic sunset there."}
              </p>
            )}
```
Hooks must be declared before the `if (!open) return null;` early return (they are, per the placement above).

In `page.tsx`, pass the prop:
```tsx
      <LocationSheet
        open={sheetOpen}
        onClose={() => setSheetOpen(false)}
        current={location}
        places={places}
        onSelect={handleLocationSelect}
        alertsVersion={alertsVersion}
      />
```

- [ ] **Step 2: Verify**

1. `env -u NODE_ENV npx tsc --noEmit` → clean.
2. In the preview (desktop Chrome, `localhost` counts as a secure context), with backend running and **local** VAPID keys in `backend/.env` plus `DATABASE_URL` pointed at a Neon branch/local Postgres (or skip bells verification until Task 10 if no DB is available locally): open the location sheet, confirm no bells before permission is granted; grant via the card path (desktop never shows the install card when already standalone-less — use console `Notification.requestPermission()` from a click if needed), reopen sheet, bells visible; toggle one and check `read_network_requests` shows `POST /push/subscribe` 204; toggle it off → `DELETE /push/subscribe` 204.
3. Screenshot the sheet with bells in light and dark.

- [ ] **Step 3: Commit**
```bash
git add frontend/src/components/LocationSheet.tsx frontend/src/app/page.tsx
git commit -m "feat(pwa): per-place Epic alert bells in the location sheet"
```

---

### Task 10: End-to-end check, docs, PR

**Files:**
- Modify: `README.md` (short "Sunset alerts" section with the setup checklist below)

- [ ] **Step 1: Full backend suite**

Run: `/Users/yotamtsabari/sunset-predictor/backend/.venv/bin/python -m pytest -q` (from `backend/`) → only the known pre-existing failure.

- [ ] **Step 2: Frontend build**

Run: `cd frontend && env -u NODE_ENV npx next build` → succeeds.

- [ ] **Step 3: Local push round-trip (desktop Chrome in the preview)**

With backend `.env` containing a Postgres `DATABASE_URL`, fresh VAPID keys (Task 5), and `ALERTS_SECRET=dev`:
1. In the preview, grant notifications and bell the current place (Task 9 flow).
2. `curl -s -X POST -H "X-Alerts-Secret: dev" "http://localhost:8000/internal/alerts/run?force=1"` → JSON with `notifications_sent >= 1`.
3. Confirm the OS notification appeared (screenshot) and that clicking it focuses the app on that place.

- [ ] **Step 4: README section**

Append to `README.md`:
```markdown
## Sunset alerts (Home Screen + push)

Afterglow can be added to the Home Screen and ping users ~4 h before an Epic
(≥ 80) sunset at places they belled. iOS requires the app to be installed to
the Home Screen (iOS 16.4+).

Setup:
1. Create a Neon Postgres database → set `DATABASE_URL` on Render.
2. `python backend/scripts/generate_vapid_keys.py` → set `VAPID_PUBLIC_KEY`,
   `VAPID_PRIVATE_KEY` (and optionally `VAPID_SUBJECT`) on Render.
3. Pick a long random `ALERTS_SECRET` → set it on Render.
4. GitHub repo secrets: `ALERTS_API_URL` (the Render URL) and `ALERTS_SECRET`.
5. Test: Actions → "Sunset alerts" → Run workflow with *force* ticked.

Cost: one prediction per 0.1° cell per day, only for belled places, at ~4 h
before local sunset — independent of subscriber count.
```

- [ ] **Step 5: Commit, push, PR** (direct push to main is blocked — feature branch + PR)
```bash
git add README.md
git commit -m "docs: sunset alerts setup"
git push -u origin HEAD
gh pr create --title "PWA install prompt + Epic sunset push alerts" --body "..."
```
PR body: summary of the feature, the manual setup checklist from the README, the cost model, and the test plan (backend tests; local push round-trip; iPhone check pending deploy). End with the Claude Code attribution line.

- [ ] **Step 6: iPhone check after deploy (with the user)**

After merge + Render/Vercel deploy + the user's manual setup: on an iPhone, open the site twice (second visit shows the card) → "Add to Home Screen" sheet → install → open from Home Screen → "Turn on alerts" → run the workflow with *force* → notification arrives → tap opens the place.
