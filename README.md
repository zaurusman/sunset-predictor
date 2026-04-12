# Sunset Predictor

**How good will tonight's sunset be?**

Sunset Predictor combines a physics-informed scoring engine with an optional ML calibration layer to predict how beautiful a sunset will be for any location and date. It returns a beauty score (0–100), a category (Poor → Epic), a confidence level, natural-language explanations, the best viewing window, and a 7-day forecast.

---

## Architecture

```
sunset-predictor/
├── backend/         Python 3.11 + FastAPI — scoring engine, ML, REST API
│   ├── app/
│   │   ├── api/             Endpoints: /health, /predict, /forecast, /model/info
│   │   ├── core/            Config (pydantic-settings) + logging
│   │   ├── models/          ML model wrapper + artifact registry
│   │   ├── services/        astronomy, weather, scoring, explanation, prediction
│   │   ├── schemas/         Pydantic request/response models
│   │   └── utils/           Math helpers, TTL cache, time utils
│   ├── scripts/
│   │   ├── build_reddit_dataset.py   Fetch r/sunset + join weather → labeled CSV
│   │   └── train_model.py            Train HistGradientBoostingRegressor
│   └── tests/               pytest suite
└── frontend/        Next.js 15 + TypeScript + Tailwind CSS
    └── src/
        ├── app/             Home page + /forecast route
        ├── components/      ScoreDial, ForecastChart, LocationSearch, …
        └── lib/             Typed API client + utilities
```

### Scoring algorithm

The beauty score is a **daily quality metric** — it answers "will today's sunset be beautiful?", not "are you watching at the exact right moment?".

Weather is sampled at the hour nearest to local sunset. Four components are weighted:

| Component | Weight | What it measures |
|---|---|---|
| Cloud Quality | 40% | High/mid cloud distribution (catching light) vs low clouds (blocking sun) |
| Atmosphere | 30% | Visibility, aerosol optical depth, clarity |
| Moisture | 20% | Precipitation and humidity penalties |
| Horizon | 10% | Permanent obstruction (buildings, mountains) |

The physics score is optionally calibrated by a trained ML model (HistGradientBoostingRegressor) using the blending formula:

```
final_score = 0.4 × physics_score + 0.6 × ml_prediction
```

The ML model is trained on Reddit engagement data (r/sunset upvotes as a proxy label) joined with historical weather. It is completely optional — the app runs in physics-only mode if no model is trained.

---

## Local Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- (Optional) Docker + Docker Compose

---

## Backend Setup

```bash
cd sunset-predictor/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env file
cp .env.example .env
# Edit .env if needed (all defaults work without changes)

# Start the API server
uvicorn app.main:app --reload --port 8000
```

The API is now running at http://localhost:8000.
Auto-generated docs: http://localhost:8000/docs

---

## Frontend Setup

```bash
cd sunset-predictor/frontend

# Install dependencies
npm install

# Copy env file
cp .env.example .env.local
# NEXT_PUBLIC_API_URL defaults to http://localhost:8000

# Start dev server
npm run dev
```

Open http://localhost:3000.

---

## Run with Docker Compose

```bash
cd sunset-predictor

# First run: copy env files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# Build and start
docker-compose up --build

# Stop
docker-compose down
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

---

## Build Reddit Dataset

Fetches top posts from r/sunset, joins each post's timestamp with historical weather from Open-Meteo, and saves a labeled CSV for training.

```bash
cd sunset-predictor/backend

# Basic (uses public Reddit JSON API, no credentials needed)
python scripts/build_reddit_dataset.py \
    --latitude 37.7749 \
    --longitude -122.4194 \
    --limit 300 \
    --out data/reddit_dataset.csv

# More options
python scripts/build_reddit_dataset.py \
    --latitude 51.5074 \
    --longitude -0.1278 \
    --subreddit sunset \
    --limit 500 \
    --time-filter month \     # hour|day|week|month|year|all
    --label-method percentile \ # percentile|log1p|zscore
    --image-only \
    --out data/reddit_dataset.csv

# With Reddit API credentials (higher rate limits)
# Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env
```

> **Note**: Reddit upvotes are a proxy for engagement, not objective beauty. The label represents "posts that the community engaged with most", which correlates but is not identical to beauty.

---

## Train the ML Model

```bash
cd sunset-predictor/backend

python scripts/train_model.py \
    --input data/reddit_dataset.csv \
    --output-dir trained_models/ \
    --blend-alpha 0.4
```

Saves:
- `trained_models/calibration_model.joblib` — the trained model
- `trained_models/model_metadata.json` — RMSE, Spearman r, feature importances

Restart the backend after training to load the new model, or the running server will pick it up on the next restart.

---

## API Examples

### Health check

```bash
curl http://localhost:8000/health
```

### Predict tonight's sunset

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 37.7749, "longitude": -122.4194}'
```

### Predict with weather override (for testing)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 37.7749,
    "longitude": -122.4194,
    "weather_override": {
      "cloud_high": 45,
      "cloud_low": 5,
      "cloud_total": 52,
      "visibility_m": 25000,
      "precipitation_mm": 0,
      "relative_humidity": 52
    }
  }'
```

### 7-day forecast

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"latitude": 37.7749, "longitude": -122.4194, "days": 7}'
```

### Model info

```bash
curl http://localhost:8000/model/info
```

### Sample response (predict)

```json
{
  "beauty_score_0_100": 72.4,
  "category": "Great",
  "confidence_0_100": 68.2,
  "reasons": [
    "High clouds look very promising for catching vivid colour.",
    "Clear air and good visibility will help colours pop.",
    "No rain near sunset — conditions are clean."
  ],
  "sunset_time": "2024-06-21T19:45:00Z",
  "best_viewing_window_start": "2024-06-21T19:35:00Z",
  "best_viewing_window_end": "2024-06-21T20:10:00Z",
  "algorithm_version": "1.0.0",
  "ml_model_used": false,
  "physics_component_breakdown": {
    "cloud_quality_score": 78.3,
    "atmosphere_score": 74.1,
    "moisture_score": 96.0,
    "horizon_score": 89.5,
    "weighted_physics_score": 72.4
  }
}
```

---

## Run Tests

```bash
cd sunset-predictor/backend
pytest tests/ -v
```

Tests cover:
- Scoring engine sanity (clear sky, overcast, rain, high clouds, etc.)
- API endpoints with deterministic weather overrides
- Explanation engine (reasons count, content, icons)
- ML smoke test (train on synthetic data → load → predict)

---

## Current Limitations

1. **Reddit labels** are engagement proxies, not direct beauty ratings. The ML model improves with better labels.
2. **Location for Reddit dataset** is manually specified. Posts may come from anywhere in the world, but weather is fetched for your specified location — this is a known mismatch. Future work: per-post geocoding from title/flair.
3. **Aerosol optical depth** is estimated from visibility when the Open-Meteo Air Quality API is unavailable. The `aerosol_is_estimated` flag in the response indicates this.
4. **Horizon obstruction** is manually specified (degrees). Future work: auto-derive from DEM elevation data.
5. **No user rating system yet**. The architecture is designed to add it.

---

## Roadmap

- [ ] Per-post geocoding from Reddit title/flair
- [ ] User rating system (POST /rate) to build ground-truth labels
- [ ] Image aesthetic scoring via a vision model (e.g. CLIP embeddings)
- [ ] Webcam / live image integration
- [ ] Terrain-based horizon obstruction auto-detection
- [ ] Personalisation (user preferences for cloud type, urban vs coastal)
- [ ] Push notifications for high-score upcoming sunsets
- [ ] Share card UI (Open Graph preview image)

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Backend | Python 3.11, FastAPI, Pydantic v2, Uvicorn, httpx |
| ML / Data | scikit-learn, XGBoost, pandas, numpy, joblib |
| Astronomy | astral |
| Weather | Open-Meteo (free, no key required) |
| Frontend | Next.js 15, TypeScript, Tailwind CSS, Recharts, Lucide |
| Infra | Docker, docker-compose |
