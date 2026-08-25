# Sunset alerts (Web Push)

A subscriber gets at most one notification per evening, sent a chosen number of
minutes before their local sunset, and only when the predicted score reaches
their threshold.

## How it fits together

Two workflows. The planner works out *when* alerts are needed; the dispatcher
only wakes the backend during those hours.

```
      ┌─ sunset-schedule.yml (daily, 03:00 UTC) ─────────────────────┐
      │  GET /notifications/schedule  ──>  backend computes, from    │
      │                                    each subscriber's sunset, │
      │                                    which UTC hours matter    │
      │  commits .github/dispatch-schedule.json                      │
      └──────────────────────────────────────────────────────────────┘
                                   │
                                   │ read locally — no network
                                   ▼
      ┌─ sunset-notifications.yml (hourly) ──────────────────────────┐
      │  is this hour in cron_hours?                                 │
      │      no  ──> exit. The backend is never contacted.           │
      │      yes ──> POST /notifications/dispatch                    │
      └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                      for each subscriber inside their window,
                      not yet checked today: score tonight
                                   │
                        score >= threshold?  ── yes ──>
                                   │
                    browser push service (FCM / Mozilla / …)
                                   │
                                   ▼
                    frontend/public/sw.js → showNotification()
```

### Why an external cron

Render's free tier puts an idle service to sleep, so an in-process scheduler
would not fire — and the evenings nobody has opened the app are exactly the
evenings the alert matters. The GitHub Actions cron both wakes the service and
triggers the run. Actions minutes are free on a public repository.

### Why the schedule file exists

This is the part that keeps the free tier free.

A [Render free service spins down after 15 minutes of inactivity][render-free],
and **spun-down services do not consume instance hours**. So every wake-up
costs roughly 15 instance-minutes whether or not there was anything to send.

Polling every 20 minutes is therefore far more expensive than it looks — the
service is woken again just as it goes to sleep:

| Approach | Wakes/day | Render hours/month | Share of the free 750 |
| --- | --- | --- | --- |
| Poll every 20 min | 72 | ~540 | **72%** |
| Scheduled (3 continents) | 9 | ~68 | 9% |
| Scheduled (one timezone) | 3 | ~23 | 3% |

Spending 72% of the monthly allowance on polling — before a single real
visitor arrives — is what the schedule file avoids. The dispatch job reads it
from the checkout, so deciding *not* to run costs nothing at all.

### Why hourly, and why leads must be ≥ 60 minutes

A subscriber's window is `[sunset - lead, sunset]`, so it is `lead_minutes`
wide. An hourly check cannot fall through a window at least 60 minutes wide —
which is why `lead_minutes` has a floor of 60 (`MIN_LEAD_MINUTES` in
`dispatch_schedule.py`). Lower it and alerts start silently not firing.

The backend also stamps `last_checked_date` per subscriber, so each is scored
**at most once per local day** regardless of how often dispatch is called.

[render-free]: https://render.com/docs/free

### Keeping the scheduled workflows alive

GitHub [disables scheduled workflows in a public repository after 60 days with
no repository activity][gh-disable]. The planner's commit counts as activity,
but it only commits when the hour band actually moves — which for a
single-timezone user base is a handful of times a year, as sunset drifts.

For an actively developed project, ordinary commits cover this. If Afterglow
goes quiet for two months, re-enable the workflows from the Actions tab.

[gh-disable]: https://docs.github.com/actions/managing-workflow-runs/disabling-and-enabling-a-workflow

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
| `GET` | `/notifications/schedule` | Requires `X-Dispatch-Secret`. The UTC hours that need a dispatch. |
| `POST` | `/notifications/dispatch` | Requires `X-Dispatch-Secret`. Returns per-run counts. |

`/schedule` sits behind the secret because it reveals how many subscribers
exist and roughly where they are.

## Testing

The suite in `backend/tests/test_notifications.py` stubs the push service and
the prediction service, so it asserts on the dispatcher's decisions — who is in
the window, who gets scored, who gets an alert — without network access:

```bash
cd backend && pytest tests/test_notifications.py
```
