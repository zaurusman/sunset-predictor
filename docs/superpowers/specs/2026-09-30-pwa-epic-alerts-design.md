# PWA install + Epic sunset alerts — design

Date: 2026-09-30
Status: approved (pending spec review)

## Goal

Make Afterglow feel like an iPhone app: installable to the Home Screen with its
icon, users gently prompted to install, and — once installed — a push
notification when an **Epic** sunset (score ≥ 80) is forecast for a place they
opted into. Keep Open-Meteo usage independent of subscriber count.

## Platform constraints that drive the design

- iOS delivers web push **only to Home-Screen-installed web apps** (iOS 16.4+),
  and `Notification.requestPermission()` must run from a user gesture inside the
  installed app. So the install pitch *is* the alerts pitch.
- iOS Safari has no `beforeinstallprompt`; we must show our own
  "Share → Add to Home Screen" guide. Android/Chrome fires `beforeinstallprompt`.
- Render free tier has an ephemeral filesystem → subscriptions go to hosted
  Postgres (Neon free tier).
- Render free tier has no cron → a GitHub Actions scheduled workflow triggers
  the alert run hourly.

## 1. Install + opt-in UX

### Phase A — install pitch (browser tab, not installed)

- Never on the first visit. Shown on the **2nd+ visit**, only **after the
  verdict has rendered**, as a dismissible card below the verdict. Never blocks
  or covers the answer (see daily-glance principle).
- Copy: "🌅 Never miss an epic sunset — add Afterglow to your Home Screen and
  we'll ping you when one's coming."
- iOS Safari: tapping opens a bottom sheet with a 2-step illustrated guide
  (Share icon → "Add to Home Screen").
- Android/Chromium: tapping calls the stashed `beforeinstallprompt` event's
  `prompt()`. If the event never fired (unsupported browser), the card is not shown.
- Dismiss → snooze 14 days. After 3 dismissals, never show again.
- Hidden when: already standalone; in-app browsers (Instagram, FB, WhatsApp,
  etc. — detected by UA) where install is impossible; iOS versions below 16.4
  (no web push, so nothing to pitch).

### Phase B — alerts opt-in (installed, standalone)

- On standalone launch, if permission is `default` and the user hasn't dismissed
  it, the same card slot shows "Turn on Epic sunset alerts for {place}" with an
  **Enable** button.
- Tap → `Notification.requestPermission()` → on `granted`, subscribe via the
  service worker's `pushManager` and register the current place with its bell on.
- Dismiss → same snooze rules as Phase A (separate counters).

### Ongoing control

- `LocationSheet` shows a bell toggle next to each saved place (only when push
  is supported and the app is standalone or permission is already granted).
- Toggling syncs the full list of belled places to the backend.
- If permission is `denied`, bells are disabled with a hint:
  "Notifications are blocked — enable them in Settings".
- Turning off all bells unsubscribes (DELETE) on the backend.

### Notification

- Title: "🔥 Epic sunset tonight"
- Body: "{place} — {score}/100. Best around {HH:MM}."
- Click → focus an existing client or open `/?lat={lat}&lon={lon}&name={name}`;
  the home page selects that place.

## 2. Backend

### Storage — Postgres (Neon), table `push_subscriptions`

| column          | type        | notes                                       |
|-----------------|-------------|---------------------------------------------|
| endpoint        | text PK     | push service URL                            |
| p256dh          | text        | subscription key                            |
| auth            | text        | subscription key                            |
| places          | jsonb       | `[{lat, lon, name}]`, max 5                 |
| last_notified   | jsonb       | `{"<cellKey>": "YYYY-MM-DD"}`               |
| created_at      | timestamptz |                                             |
| updated_at      | timestamptz |                                             |

Plus table `alert_cell_checks (cell_key text, local_date date, score int,
checked_at timestamptz, PK(cell_key, local_date))` so each cell is predicted at
most once per local day even across hourly runs.

Table creation is idempotent (`CREATE TABLE IF NOT EXISTS`) at startup when
`DATABASE_URL` is set. If `DATABASE_URL` is unset, push endpoints return 503
and the rest of the app is unaffected. Driver: `asyncpg`.

### Endpoints

- `GET /push/vapid-key` → `{ "public_key": "..." }`.
- `POST /push/subscribe` body `{ subscription: {endpoint, keys:{p256dh, auth}}, places: [...] }`
  → upsert. Validates ≤ 5 places, lat/lon ranges.
- `DELETE /push/subscribe` body `{ endpoint }` → delete.
- `POST /internal/alerts/run` — requires header `X-Alerts-Secret` equal to
  `ALERTS_SECRET`; returns a JSON summary `{cells_checked, notifications_sent,
  pruned}`. Optional query `force=1` (still requires the secret) ignores the
  timing window, the score threshold, and both per-day dedupes, for manual
  end-to-end testing.

### Alert run algorithm

1. Load all subscriptions. Expand to (subscription, place) pairs.
2. Group places by **cell** = lat/lon rounded to 0.1° (same as
   `CACHE_COORD_DECIMALS`, so it shares the weather cache).
3. For each cell, compute today's local sunset (existing `astronomy_service`,
   timezone from the cell's coordinates as the predict path already does).
   The cell is **due** if `now` is within [sunset − 4.5h, sunset − 3.5h) and
   there is no `alert_cell_checks` row for (cell, local date).
4. For each due cell: run one prediction through `prediction_service` for the
   cell center and today (same code + cache as `/predict`). Record the check row
   (even when the score is low, so the cell isn't re-predicted today).
   On `WeatherUnavailableError`, don't record — the next hourly run retries while
   still in the window.
5. If score ≥ 80: for each subscription with a place in that cell whose
   `last_notified[cell]` ≠ today, send one push (payload: title, body, url),
   then set `last_notified[cell] = today`. One notification per subscription
   per cell per day, even if several of its places share the cell.
6. A push response of 404/410 deletes the subscription. Other failures are
   logged and skipped.

Open-Meteo cost: ≤ one prediction per distinct cell per day, independent of
subscriber count, spread through the day by sunset time, often served from cache.

### Scheduler

`.github/workflows/sunset-alerts.yml`: `schedule: cron "7 * * * *"` plus
`workflow_dispatch`; one `curl -fsS -X POST` to
`${{ secrets.ALERTS_API_URL }}/internal/alerts/run` with the secret header.
The ±1h window tolerates GitHub cron delays.

### Config (env)

`DATABASE_URL`, `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`, `VAPID_SUBJECT`
(`mailto:` address), `ALERTS_SECRET`. Also document in `backend/.env.example`.
New deps: `asyncpg`, `pywebpush`.

## 3. Frontend

- `public/sw.js` (hand-written): `push` → `showNotification`;
  `notificationclick` → focus/open URL; no fetch caching (offline is out of scope).
  Registered once from a small client component in the layout.
- `src/lib/install.ts`: platform detection (iOS, Android, standalone, in-app
  browser, push support), visit counter, snooze/dismiss state in localStorage,
  stash of `beforeinstallprompt`.
- `src/lib/push.ts`: fetch VAPID key, subscribe/unsubscribe, get/set belled
  places (localStorage mirror) and sync to backend via `lib/api.ts`.
- `src/components/InstallPrompt.tsx`: the card (Phase A / Phase B variants) and
  the iOS instructions sheet.
- `LocationSheet.tsx`: bell toggles per saved place.
- `page.tsx`: honour `?lat&lon&name` from notification clicks; render
  `InstallPrompt` under the verdict.
- `manifest.ts`: add `id: "/"`, `scope: "/"`, a `purpose: "maskable"` icon entry,
  and `orientation: "portrait"`.

## 4. Error handling

- Missing DB/VAPID config → push endpoints 503; frontend hides bells and the
  Phase B card when `/push/vapid-key` fails.
- Alert run is idempotent per (cell, day) and per (subscription, cell, day):
  sending nothing on failure is acceptable, sending twice is not.

## 5. Testing

- pytest: cell grouping; due-window selection across time zones; per-day check
  dedupe; per-subscription notify dedupe; 410 pruning; secret enforcement;
  subscribe validation. DB layer and webpush mocked behind small interfaces.
- Manual E2E: deploy, install on iPhone, enable alerts, run the workflow with
  `force=1` against a place, confirm notification + click-through.

## Out of scope

Offline mode, day-before heads-up, user-selectable threshold, email alerts,
migrating ratings.jsonl to Postgres.

## Manual setup (by the user)

1. Create a Neon Postgres DB; set `DATABASE_URL` on Render.
2. Set VAPID keys (generated during implementation), `VAPID_SUBJECT`,
   `ALERTS_SECRET` on Render.
3. GitHub repo secrets: `ALERTS_API_URL`, `ALERTS_SECRET`.
