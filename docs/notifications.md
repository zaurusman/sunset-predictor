# Sunset alerts (Web Push)

A subscriber gets at most one notification per evening, sent a chosen number of
minutes before their local sunset, and only when the predicted score reaches
their threshold.

## How it fits together

```
GitHub Actions cron  ──POST /notifications/dispatch──>  FastAPI backend
   (every 20 min)         (X-Dispatch-Secret)                │
                                                             │ for each subscriber
                                                             │ inside their window,
                                                             │ not yet checked today
                                                             ▼
                                                        score tonight
                                                             │
                                              score >= threshold?
                                                             │ yes
                                                             ▼
                              browser push service (FCM / Mozilla / …)
                                                             │
                                                             ▼
                                              frontend/public/sw.js
                                                  showNotification()
```

### Why an external cron

Render's free tier puts an idle service to sleep, so an in-process scheduler
would not fire — and the evenings nobody has opened the app are exactly the
evenings the alert matters. The GitHub Actions cron both wakes the service and
triggers the run, at no cost.

### Why every 20 minutes, all day

Subscribers can be anywhere, so sunset happens at every hour of the UTC day.
The backend stamps `last_checked_date` per subscriber, so each is scored **at
most once per local day** — extra ticks cost a store scan and nothing more.

## Setup

### 1. Generate keys

```bash
cd backend && python scripts/generate_vapid_keys.py
```

This prints a VAPID keypair and a dispatch secret. The **private key is a
credential** — it goes in the environment, never in the repo.

Rotating the keypair invalidates every existing subscription, so generate once
and keep it.

### 2. Backend environment

Local (`backend/.env`) and production (Render dashboard → Environment):

| Variable | Purpose |
| --- | --- |
| `VAPID_PUBLIC_KEY` | Served to browsers; identifies this server |
| `VAPID_PRIVATE_KEY` | Signs each push. Secret. |
| `VAPID_SUBJECT` | `mailto:` or `https:` contact for the push service |
| `NOTIFY_DISPATCH_SECRET` | Required by `POST /notifications/dispatch` |
| `SUBSCRIPTIONS_PATH` | Where subscriptions are stored (see the caveat below) |

Leaving `VAPID_PRIVATE_KEY` blank disables the whole feature: `/notifications/config`
reports `enabled: false` and the frontend hides the toggle rather than offering
one that cannot work.

### 3. Repository secrets

For the cron in `.github/workflows/sunset-notifications.yml`:

- `BACKEND_URL` — e.g. `https://sunset-predictor-b8ig.onrender.com`
- `NOTIFY_DISPATCH_SECRET` — the same value as the backend's

Trigger a run by hand from the Actions tab (`workflow_dispatch`) to check the
wiring before waiting on the schedule.

## ⚠️ Before this is a real feature: persistent storage

`SUBSCRIPTIONS_PATH` defaults to `data/subscriptions.json`, and **Render's free
tier has an ephemeral filesystem**. Every redeploy — and every restart after
the free tier's idle sleep — wipes that file. Everyone silently stops receiving
alerts.

The frontend detects this (`GET /notifications/status` returns 404 for a
browser the server has forgotten) and shows "Alerts stopped — tap to switch
them back on", so it fails visibly rather than silently. But that is damage
control, not a fix.

The fix is one of:

- a Render persistent disk (paid) mounted at, say, `/var/data`, with
  `SUBSCRIPTIONS_PATH=/var/data/subscriptions.json`; or
- swapping `SubscriptionStore` for a hosted store — its interface is four
  methods (`all`, `get`, `upsert`, `update_fields`, `delete`), so a Postgres or
  Redis implementation is a drop-in.

## Platform notes

**iOS**: Safari supports Web Push **only for a site installed to the Home
Screen**. In a normal Safari tab there is no `PushManager` at all. The UI
detects iOS specifically and says to install first, rather than reporting the
browser as unsupported. `manifest.json` and the `appleWebApp` metadata exist to
make that install work.

**Permission**: once a user denies notifications, the browser will not prompt
again — only they can undo it in site settings. The UI says so instead of
retrying.

## Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/notifications/config` | Public. Whether alerts are on, and the VAPID public key. |
| `POST` | `/notifications/subscribe` | Upsert by endpoint — one browser, one subscription. |
| `POST` | `/notifications/unsubscribe` | 204 whether or not it existed. |
| `GET` | `/notifications/status?endpoint=…` | 404 when the server has forgotten this browser. |
| `POST` | `/notifications/dispatch` | Requires `X-Dispatch-Secret`. Returns per-run counts. |

## Testing

The suite in `backend/tests/test_notifications.py` stubs the push service and
the prediction service, so it asserts on the dispatcher's decisions — who is in
the window, who gets scored, who gets an alert — without network access:

```bash
cd backend && pytest tests/test_notifications.py
```
