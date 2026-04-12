"""
Multi-location Dataset Builder + ML Trainer
============================================

Fetches r/sunset posts ONCE, joins each post's timestamp with historical
weather from Open-Meteo at MULTIPLE geographic locations, computes the
ACTUAL physics score for every row, then trains the ML calibration model.

Why multiple locations?
  Reddit posts come from all over the world. We don't know each post's exact
  location, so we use several representative cities. This gives the model
  more diverse weather features for the same set of human-judged sunsets,
  reducing overfitting to one climate.

Run from backend/:
    python scripts/build_and_train.py

Options (see --help):
    --limit        Max Reddit posts to fetch  (default: 1000)
    --out-dir      Where to save datasets and model  (default: data/ + trained_models/)
    --min-score    Minimum post upvotes  (default: 5)
    --blend-alpha  Physics weight in final blend (default: 0.4)
    --image-only   Keep only image posts
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.ml_model import FEATURE_NAMES
from app.services.scoring_engine import ScoringEngine
from app.schemas.weather import WeatherSnapshot
from app.utils.math_utils import clamp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locations: geographically diverse sample covering common sunset photography
# regions and a range of climates.
# ---------------------------------------------------------------------------
LOCATIONS = [
    {"name": "San Francisco",  "lat": 37.7749,  "lon": -122.4194},
    {"name": "Tel Aviv",       "lat": 32.0853,  "lon":   34.7818},
    {"name": "London",         "lat": 51.5074,  "lon":   -0.1278},
    {"name": "Sydney",         "lat": -33.8688, "lon":  151.2093},
    {"name": "Cape Town",      "lat": -33.9249, "lon":   18.4241},
]

REDDIT_HEADERS = {
    "User-Agent": os.getenv("REDDIT_USER_AGENT", "sunset-predictor-dataset/1.0"),
}
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_VARS = ",".join([
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "visibility", "relative_humidity_2m", "dew_point_2m", "temperature_2m",
    "precipitation", "wind_speed_10m", "surface_pressure",
])
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

_scoring_engine = ScoringEngine()


# ---------------------------------------------------------------------------
# Reddit fetching
# ---------------------------------------------------------------------------

def fetch_reddit_posts(limit: int, min_score: int, image_only: bool) -> list[dict]:
    """Fetch top all-time r/sunset posts via the public JSON API."""
    posts: list[dict] = []
    after: Optional[str] = None

    while len(posts) < limit:
        batch = min(100, limit - len(posts))
        params: dict[str, Any] = {"t": "all", "limit": batch}
        if after:
            params["after"] = after

        try:
            resp = requests.get(
                "https://www.reddit.com/r/sunset/top.json",
                params=params,
                headers=REDDIT_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Reddit API error: %s — stopping at %d posts", exc, len(posts))
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            p = child.get("data", {})
            if p.get("score", 0) < min_score:
                continue
            url = p.get("url", "")
            if image_only and Path(url).suffix.lower() not in IMAGE_EXTS:
                continue
            posts.append({
                "post_id":     p.get("id", ""),
                "score":       int(p.get("score", 0)),
                "created_utc": int(p.get("created_utc", 0)),
            })

        after = data.get("data", {}).get("after")
        if not after:
            break
        time.sleep(2)

    logger.info("Fetched %d Reddit posts", len(posts))
    return posts


# ---------------------------------------------------------------------------
# Weather fetching
# ---------------------------------------------------------------------------

def fetch_weather_for_date(lat: float, lon: float, d: str) -> Optional[dict]:
    """
    Fetch hourly weather from the Open-Meteo archive for a single date.
    Returns a dict of hourly arrays keyed by variable name, or None on failure.
    """
    try:
        resp = requests.get(
            OPEN_METEO_ARCHIVE,
            params={
                "latitude": lat, "longitude": lon,
                "hourly": ARCHIVE_VARS,
                "start_date": d, "end_date": d,
                "timezone": "UTC",
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json().get("hourly", {})
    except Exception as exc:
        logger.debug("Weather API error for %s at (%.2f,%.2f): %s", d, lat, lon, exc)
        return None


def extract_hour(hourly: dict, target_hour: int) -> Optional[dict]:
    """Pick the row nearest to target_hour from hourly data."""
    times: list[str] = hourly.get("time", [])
    if not times:
        return None
    idx = min(range(len(times)), key=lambda i: abs(
        datetime.fromisoformat(times[i]).hour - target_hour
    ))
    def get(key: str, default: float = 0.0) -> float:
        v = hourly.get(key, [])
        return float(v[idx]) if idx < len(v) and v[idx] is not None else default

    return {
        "cloud_cover":     get("cloud_cover"),
        "cloud_cover_low": get("cloud_cover_low"),
        "cloud_cover_mid": get("cloud_cover_mid"),
        "cloud_cover_high":get("cloud_cover_high"),
        "visibility_m":    get("visibility", 10000.0),
        "relative_humidity": get("relative_humidity_2m", 50.0),
        "dewpoint_c":      get("dew_point_2m"),
        "temperature_c":   get("temperature_2m"),
        "precipitation_mm":get("precipitation"),
        "wind_speed_kmh":  get("wind_speed_10m"),
        "pressure_hpa":    get("surface_pressure", 1013.0),
    }


# ---------------------------------------------------------------------------
# Physics score computation
# ---------------------------------------------------------------------------

def compute_physics_score(row: dict) -> float:
    """Run the real ScoringEngine on a row dict."""
    snap = WeatherSnapshot(
        cloud_low=row["cloud_cover_low"],
        cloud_mid=row["cloud_cover_mid"],
        cloud_high=row["cloud_cover_high"],
        cloud_total=row["cloud_cover"],
        visibility_m=row["visibility_m"],
        relative_humidity=row["relative_humidity"],
        dewpoint_c=row["dewpoint_c"],
        temperature_c=row["temperature_c"],
        precipitation_mm=row["precipitation_mm"],
        wind_speed_kmh=row["wind_speed_kmh"],
        pressure_hpa=row["pressure_hpa"],
        aerosol_optical_depth=None,
        sun_elevation_deg=5.0,  # approximate at sunset
        data_source="archive",
        aerosol_is_estimated=True,
    )
    result = _scoring_engine.score(snap, horizon_obstruction_deg=2.0)
    return round(result.physics_score, 2)


# ---------------------------------------------------------------------------
# Feature engineering (matches train_model.py + ml_model.py exactly)
# ---------------------------------------------------------------------------

def build_feature_vector(row: dict) -> np.ndarray:
    month = datetime.fromtimestamp(row["created_utc"], tz=timezone.utc).month
    aod = row.get("aerosol_optical_depth", 0.15) or 0.15
    return np.array([
        row["cloud_cover_low"],
        row["cloud_cover_mid"],
        row["cloud_cover_high"],
        row["cloud_cover"],
        math.log(row["visibility_m"] + 1.0),
        row["relative_humidity"],
        row["dewpoint_c"],
        math.log1p(max(0, row["precipitation_mm"])),
        row["wind_speed_kmh"],
        row["pressure_hpa"],
        aod,
        math.sin(month * 2.0 * math.pi / 12.0),
        math.cos(month * 2.0 * math.pi / 12.0),
        5.0,   # sun_elevation_deg approximation at sunset
        row["physics_score"],
        2.0,   # horizon_obstruction_deg default
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_dataset(
    posts: list[dict],
    locations: list[dict],
) -> pd.DataFrame:
    """Join each post × each location with real weather and physics score."""
    rows = []
    total = len(posts) * len(locations)
    done = 0

    for loc in locations:
        lat, lon = loc["lat"], loc["lon"]
        logger.info("Fetching weather for %s (%.4f, %.4f)…", loc["name"], lat, lon)

        # Cache weather by date for this location (avoid duplicate API calls)
        weather_cache: dict[str, Optional[dict]] = {}

        for post in posts:
            dt = datetime.fromtimestamp(post["created_utc"], tz=timezone.utc)
            d_str = dt.date().isoformat()

            if d_str not in weather_cache:
                weather_cache[d_str] = fetch_weather_for_date(lat, lon, d_str)
                time.sleep(0.4)  # polite rate limit

            hourly = weather_cache[d_str]
            if hourly is None:
                done += 1
                continue

            # Sunset is roughly 18:00 local — use UTC hour 16 as approximation
            sunset_utc_hour = 16
            weather = extract_hour(hourly, sunset_utc_hour)
            if weather is None:
                done += 1
                continue

            physics = compute_physics_score(weather)

            rows.append({
                **post,
                "location": loc["name"],
                "location_lat": lat,
                "location_lon": lon,
                "post_date": d_str,
                **weather,
                "physics_score": physics,
            })
            done += 1
            if done % 200 == 0:
                logger.info("  Progress: %d / %d rows", done, total)

    df = pd.DataFrame(rows)
    logger.info("Total rows with weather: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile-rank upvotes across the ENTIRE merged dataset so labels are
    consistent across locations.
    """
    df = df.copy()
    df["beauty_label"] = df["score"].rank(pct=True) * 100.0
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame, output_dir: Path, blend_alpha: float) -> dict:
    """Train HistGradientBoostingRegressor and return metrics."""
    df = df.sort_values("created_utc")
    split = int(len(df) * 0.80)
    train_df, val_df = df.iloc[:split], df.iloc[split:]

    logger.info("Train: %d  |  Validation: %d", len(train_df), len(val_df))

    X_train = np.stack([build_feature_vector(r) for _, r in train_df.iterrows()])
    y_train = train_df["beauty_label"].values.astype(float)
    X_val   = np.stack([build_feature_vector(r) for _, r in val_df.iterrows()])
    y_val   = val_df["beauty_label"].values.astype(float)

    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.04,
        max_depth=4,
        min_samples_leaf=15,
        l2_regularization=0.1,
        random_state=42,
    )
    logger.info("Training HistGradientBoostingRegressor (max_iter=500)…")
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, preds))
    mae  = mean_absolute_error(y_val, preds)
    sr, sp = spearmanr(y_val, preds)

    logger.info("── Validation metrics ──────────────────────────────")
    logger.info("  RMSE:       %.2f", rmse)
    logger.info("  MAE:        %.2f", mae)
    logger.info("  Spearman r: %.3f  (p=%.4f)", sr, sp)

    importances = {}
    if hasattr(model, "feature_importances_"):
        for name, imp in zip(FEATURE_NAMES, model.feature_importances_):
            importances[name] = round(float(imp), 4)
        logger.info("── Top feature importances ─────────────────────────")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1])[:8]:
            logger.info("  %-35s %.4f", name, imp)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "calibration_model.joblib"
    joblib.dump(model, model_path)
    logger.info("Model saved → %s", model_path)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_locations": df["location"].nunique(),
        "locations": df["location"].unique().tolist(),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "spearman_r": round(float(sr), 4),
        "spearman_p": round(float(sp), 6),
        "feature_names": FEATURE_NAMES,
        "feature_importances": importances,
        "blend_alpha": blend_alpha,
        "algorithm_version": "1.0.0",
    }
    meta_path = output_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Metadata saved → %s", meta_path)

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset + train ML calibration model.")
    parser.add_argument("--limit",       type=int,   default=1000)
    parser.add_argument("--min-score",   type=int,   default=5)
    parser.add_argument("--out-dir",     default="data")
    parser.add_argument("--model-dir",   default="trained_models")
    parser.add_argument("--blend-alpha", type=float, default=0.4)
    parser.add_argument("--image-only",  action="store_true")
    parser.add_argument("--locations",   type=int,   default=5,
                        help="How many of the built-in locations to use (1–5, default 5)")
    args = parser.parse_args()

    locations = LOCATIONS[: args.locations]
    logger.info("Using %d locations: %s", len(locations), [l["name"] for l in locations])

    # Step 1: fetch Reddit posts once
    logger.info("── Step 1: Fetching Reddit posts ───────────────────")
    posts = fetch_reddit_posts(args.limit, args.min_score, args.image_only)
    if not posts:
        logger.error("No posts fetched. Exiting.")
        sys.exit(1)

    # Step 2: join with weather at each location
    logger.info("── Step 2: Joining with weather (%d locations) ─────", len(locations))
    df = build_dataset(posts, locations)
    if df.empty:
        logger.error("No rows with weather data. Exiting.")
        sys.exit(1)

    # Step 3: label
    logger.info("── Step 3: Building labels ─────────────────────────")
    df = build_labels(df)

    # Step 4: save raw dataset
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "multi_location_dataset.csv"
    df.to_csv(dataset_path, index=False)
    logger.info("Dataset saved → %s  (%d rows)", dataset_path, len(df))

    # Step 5: train
    logger.info("── Step 4: Training ────────────────────────────────")
    metrics = train(df, Path(args.model_dir), args.blend_alpha)

    # Decision
    sr = metrics["spearman_r"]
    threshold = 0.35
    logger.info("")
    if sr >= threshold:
        logger.info("✓ Spearman r=%.3f ≥ %.2f — model is useful. Restart the backend to load it.", sr, threshold)
    else:
        logger.info("✗ Spearman r=%.3f < %.2f — model quality is low; physics-only mode is better.", sr, threshold)


if __name__ == "__main__":
    main()
