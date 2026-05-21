# Afterglow — Agent Handover Note

## What This Project Is

**Afterglow** is a sunset beauty predictor web app. Given a location and date it returns a 0–100 score, category, reasons, and the best viewing window. It has two parts:

- **Backend** — FastAPI + Python at `backend/` — runs on port 8000
- **Frontend** — Next.js 15 (App Router) at `frontend/` — runs on port 3000

---

## How to Start the Servers

**Backend:**
```bash
cd /Users/yotamtsabari/sunset-predictor/backend
.venv/bin/uvicorn app.main:app --reload --port 8000
```

**Frontend** — IMPORTANT: the shell has `NODE_ENV=production` set globally which breaks Next.js. Always strip the environment:
```bash
cd /Users/yotamtsabari/sunset-predictor
env -i HOME="$HOME" PATH="$PATH" LANG=en_US.UTF-8 bash -c 'cd frontend && npx next dev'
```
Frontend log is written to `/tmp/frontend.log` when running in background.

---

## Project Structure

```
sunset-predictor/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI factory + lifespan wiring
│   │   ├── core/config.py             # Settings (env vars)
│   │   ├── api/
│   │   │   ├── predict.py             # POST /predict
│   │   │   ├── forecast.py            # POST /forecast
│   │   │   ├── health.py              # GET /health
│   │   │   ├── geocode.py             # GET /geocode
│   │   │   ├── model_info.py          # GET /model/info
│   │   │   └── submit.py             # POST /submit-photo
│   │   ├── services/
│   │   │   ├── weather_service.py     # Open-Meteo fetching + caching
│   │   │   ├── astronomy_service.py   # Sunset times, solar elevation
│   │   │   ├── scoring_engine.py      # Physics-based 4-component scoring
│   │   │   ├── prediction_service.py  # Orchestration layer
│   │   │   └── explanation_engine.py  # Natural-language reasons
│   │   ├── schemas/
│   │   │   ├── weather.py             # WeatherSnapshot, WeatherOverride
│   │   │   ├── prediction.py          # PredictRequest/Response
│   │   │   └── forecast.py            # ForecastRequest/Response, DayForecast
│   │   ├── models/
│   │   │   ├── ml_model.py            # ML calibration model (optional)
│   │   │   └── model_registry.py      # Loads .joblib from trained_models/
│   │   └── utils/
│   │       ├── cache.py               # TTLCache (in-memory, thread-safe)
│   │       └── math_utils.py          # clamp, bell_curve, etc.
│   └── .venv/                         # Python venv — use .venv/bin/python etc.
│
└── frontend/
    ├── src/
    │   ├── app/
    │   │   ├── layout.tsx             # Root layout + ThemeProvider
    │   │   ├── globals.css            # Tailwind base + scrollbar + glass helpers
    │   │   ├── page.tsx               # Main page (single-day prediction)
    │   │   └── forecast/page.tsx      # 7-day forecast page
    │   ├── components/
    │   │   ├── ScoreDial.tsx          # SVG arc score dial (uses useTheme)
    │   │   ├── ForecastChart.tsx      # Recharts bar chart (uses useTheme)
    │   │   ├── SunsetCard.tsx         # Collapsible forecast day card
    │   │   ├── LocationSearch.tsx     # Geocode search input + dropdown
    │   │   ├── DatePicker.tsx         # Calendar date picker popover
    │   │   ├── ComponentBreakdown.tsx # Physics score bars
    │   │   ├── ReasonsList.tsx        # "Why" bullet list
    │   │   ├── ViewingWindow.tsx      # Timeline showing best window
    │   │   ├── SubmitPhotoModal.tsx   # Photo upload modal
    │   │   ├── ModelInfoPanel.tsx     # Debug panel showing model info
    │   │   ├── ThemeProvider.tsx      # next-themes wrapper ("use client")
    │   │   ├── ThemeToggle.tsx        # Sun/Moon toggle button
    │   │   ├── LoadingState.tsx       # Spinner + message
    │   │   └── ErrorAlert.tsx         # Error box + retry button
    │   └── lib/
    │       ├── api.ts                 # Typed fetch wrappers for all endpoints
    │       └── types.ts               # TypeScript types mirroring Pydantic schemas
    ├── public/
    │   └── logo.png                   # App logo (360×60 px)
    └── .next/                         # Build cache — delete if stale image issues
```

---

## Backend Architecture (Key Details)

### Scoring System (4 components, physics-based)
| Component | Weight | What it measures |
|---|---|---|
| Cloud Quality | 42% | High/mid/low cloud distribution for colour potential |
| Atmosphere | 28% | Visibility, aerosol optical depth, humidity |
| Moisture | 20% | Precipitation, clearing trends, pressure rising |
| Horizon | 10% | Obstruction from terrain/buildings (default 2°) |

Score categories: Poor (0–29), Decent (30–49), Good (50–64), Great (65–79), Epic (80+)

### Weather Data Sources (Open-Meteo — free)
- **Forecast API** (`api.open-meteo.com/v1/forecast`) — today + future + up to 7 days past via `past_days`
- **Archive API** (`archive-api.open-meteo.com/v1/archive`) — dates >7 days ago; supports `start_date`/`end_date` range (one call for months of data)
- **Air Quality API** (`air-quality-api.open-meteo.com/v1/air-quality`) — aerosol optical depth (AOD); gracefully falls back to proxy estimate if unavailable

### Caching
`TTLCache` in `app/utils/cache.py` — in-memory, thread-safe, TTL=900s by default. Keys are MD5 hashes of args via `TTLCache.make_key(*args)`. Historical data never changes, so the TTL can be much longer for past scores.

### Adding a New Endpoint (pattern to follow)
1. Create `app/schemas/newfeature.py` with Pydantic models
2. Create `app/api/newfeature.py` with a FastAPI router
3. Register router in `app/main.py`: `app.include_router(newfeature.router)`
4. Inject `prediction_service` from `request.app.state.prediction_service` in the handler

---

## Frontend Architecture (Key Details)

### Theme System
- **Light mode by default**, dark mode toggle via sun/moon button
- Uses `next-themes` v0.4.6 with `attribute="class"`, `defaultTheme="light"`, `enableSystem={false}`
- Tailwind `darkMode: "class"` in `tailwind.config.ts`
- Pattern for all new components: always write light default + `dark:` variant
  - e.g. `bg-gray-100/60 dark:bg-slate-900/60`, `text-gray-900 dark:text-white`
- `ThemeToggle` has a `mounted` guard to avoid hydration mismatch
- `ScoreDial` and `ForecastChart` use `useTheme()` hook for dynamic SVG/canvas colors

### Pages
- `/` — main page: location search, date picker, score dial, breakdown, weather stats, links to forecast
- `/forecast` — 7-day forecast: chart overview + collapsible day cards
- Both pages import `ThemeToggle` and have `bg-gray-50 dark:bg-slate-950` on `<main>`

### API Client (`lib/api.ts`)
All backend calls go through `lib/api.ts`. `API_BASE` reads from `NEXT_PUBLIC_API_URL` env var, falls back to `http://localhost:8000`.

### Known Gotchas
- **`NODE_ENV=production` in shell** — breaks npm devDependencies + Next.js SWC. Always launch frontend with `env -i` (see above).
- **Stale logo** — Next.js caches optimized images in `.next/cache/images/`. If the user replaces `public/logo.png`, run `rm -rf frontend/.next/cache/images` then hard-refresh (`Cmd+Shift+R`).
- **Port 8000 in use** — `lsof -ti:8000 | xargs kill -9`

---

## Implemented Features

### Historical Heatmap ✓
GitHub-style contribution calendar showing past sunset scores by location. Fully implemented.
- Backend: `GET /heatmap?lat=&lon=&months=` in `backend/app/api/heatmap.py`
- Frontend: `/heatmap` page with 6m/12m/24m selector and best-months summary
- Scores match `/predict` exactly: same 4-point window pipeline + ML blend
- Data source split: forecast API (last 7 days) / archive API (older) — mirrors `/predict`

---

## Known Limitations / Future Improvements

### Atmosphere accuracy for historical dates (archive API)
The Open-Meteo **archive API never returns visibility data** — the field is always `null` for all locations and all dates. As a result:
- Visibility defaults to **24 km** (changed from 10 km which was causing systematic under-scoring)
- AOD is always proxy-estimated from humidity alone (no real aerosol data)
- The atmosphere component (28% weight) is approximate for all dates > 7 days old

**Impact:** The model accurately captures cloud cover patterns (42% weight, ERA5 data) but cannot detect unusual atmosphere events like sandstorms, heavy pollution, or sea fog in the historical record. For the heatmap use case (identifying good months/seasons) this is acceptable. For day-level accuracy in the past it's a known gap.

**Future fix:** Fetch real AOD from the CAMS/air-quality archive for dates 8–92 days ago (Open-Meteo AQ API supports `past_days` up to 92). Implement in `weather_service.get_historical_range_windows()` — add an archive AQ fetch alongside the archive weather fetch for that date range.

---

## Environment Variables
Backend reads from `.env` in `backend/`:
- `RESEND_API_KEY` — email provider for photo submissions
- `RESEND_FROM_EMAIL` / `DEVELOPER_EMAIL` — photo submission email routing
- `APP_ENV` — "development" | "production"
- `ALGORITHM_VERSION` — shown in UI footer

Frontend reads from `frontend/.env.local`:
- `NEXT_PUBLIC_API_URL` — backend URL (default: `http://localhost:8000`)

---

## Deployment Plan (Saved in `.claude/plans/`)
A Vercel (frontend) + Render (backend) deployment plan exists at:
`/Users/yotamtsabari/.claude/plans/compressed-toasting-naur.md`

Key steps: create `render.yaml` in repo root, update `.gitignore` to allow `.joblib` model files, push to GitHub, connect to Render + Vercel. This has NOT been done yet.
