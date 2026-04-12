"""
Sunset Predictor — ML Calibration Model Trainer
================================================

Trains a HistGradientBoostingRegressor on the labeled Reddit dataset to
produce a calibrated beauty score that supplements the physics engine.

Usage
-----
    python scripts/train_model.py \\
        --input data/reddit_dataset.csv \\
        --output-dir trained_models/ \\
        --blend-alpha 0.4

The trained model is saved as:
    trained_models/calibration_model.joblib
    trained_models/model_metadata.json

The metadata JSON can be inspected directly or fetched via GET /model/info.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# ── Bootstrap path ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.ml_model import FEATURE_NAMES
from app.utils.math_utils import clamp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive ML features from the raw dataset columns.

    Feature list (must match FEATURE_NAMES in ml_model.py):
        cloud_cover_low, cloud_cover_mid, cloud_cover_high, cloud_cover,
        log_visibility_m, relative_humidity, dewpoint_c,
        log1p_precipitation, wind_speed_kmh, pressure_hpa,
        aerosol_optical_depth (imputed), sin_month, cos_month,
        sun_elevation_deg, physics_score, horizon_obstruction_deg
    """
    out = df.copy()

    # Log-transform skewed variables
    out["log_visibility_m"] = np.log(out["visibility_m"].clip(lower=1) + 1)
    out["log1p_precipitation"] = np.log1p(out["precipitation_mm"].clip(lower=0))

    # Seasonality encoding from post_date
    if "post_date" in out.columns:
        months = pd.to_datetime(out["post_date"]).dt.month
    else:
        months = pd.Series([6] * len(out))  # fallback: assume June

    out["sin_month"] = np.sin(months * 2 * math.pi / 12)
    out["cos_month"] = np.cos(months * 2 * math.pi / 12)

    # AOD: impute with median if missing
    if "aerosol_optical_depth" not in out.columns:
        out["aerosol_optical_depth"] = 0.15  # global median approximation
    else:
        median_aod = out["aerosol_optical_depth"].median()
        out["aerosol_optical_depth"] = out["aerosol_optical_depth"].fillna(
            median_aod if not pd.isna(median_aod) else 0.15
        )

    # Physics score (if not already present, estimate from cloud cover heuristic)
    if "physics_score" not in out.columns:
        # Rough proxy: penalise heavy low clouds and total overcast
        out["physics_score"] = (
            50
            + (out.get("cloud_cover_high", out.get("cloud_cover", 30)) - 30) * 0.3
            - out.get("cloud_cover_low", 15) * 0.4
            - (out.get("precipitation_mm", 0) * 10)
        ).clip(0, 100)

    # Horizon obstruction (not in Reddit dataset — use default)
    if "horizon_obstruction_deg" not in out.columns:
        out["horizon_obstruction_deg"] = 2.0

    # Sun elevation (not in Reddit dataset — estimate from month/latitude)
    if "sun_elevation_deg" not in out.columns:
        out["sun_elevation_deg"] = 10.0  # approximate sunset elevation

    # Rename to canonical names
    rename_map = {
        "cloud_cover": "cloud_cover",  # already named
        "cloud_cover_low": "cloud_cover_low",
        "cloud_cover_mid": "cloud_cover_mid",
        "cloud_cover_high": "cloud_cover_high",
        "relative_humidity": "relative_humidity",
        "dewpoint_c": "dewpoint_c",
        "temperature_c": "temperature_c",
        "wind_speed_kmh": "wind_speed_kmh",
        "pressure_hpa": "pressure_hpa",
    }

    return out


def select_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the feature matrix in the correct column order."""
    # Map dataset column names → FEATURE_NAMES
    col_map = {
        "cloud_cover_low": "cloud_cover_low",
        "cloud_cover_mid": "cloud_cover_mid",
        "cloud_cover_high": "cloud_cover_high",
        "cloud_cover": "cloud_cover",
        "log_visibility_m": "log_visibility_m",
        "relative_humidity": "relative_humidity",
        "dewpoint_c": "dewpoint_c",
        "log1p_precipitation": "log1p_precipitation",
        "wind_speed_kmh": "wind_speed_kmh",
        "pressure_hpa": "pressure_hpa",
        "aerosol_optical_depth": "aerosol_optical_depth",
        "sin_month": "sin_month",
        "cos_month": "cos_month",
        "sun_elevation_deg": "sun_elevation_deg",
        "physics_score": "physics_score",
        "horizon_obstruction_deg": "horizon_obstruction_deg",
    }

    rows = []
    for fname in FEATURE_NAMES:
        col = col_map.get(fname, fname)
        if col in df.columns:
            rows.append(df[col].values)
        else:
            logger.warning("Feature %s not found in dataset — filling with 0", fname)
            rows.append(np.zeros(len(df)))

    return np.column_stack(rows)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    input_path: str,
    output_dir: str,
    blend_alpha: float = 0.4,
    label_col: str = "beauty_label",
) -> None:
    # ── Load data ─────────────────────────────────────────────────────────
    path = Path(input_path)
    if not path.exists():
        logger.error("Input file not found: %s", path)
        sys.exit(1)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    logger.info("Loaded %d rows from %s", len(df), path)

    if label_col not in df.columns:
        logger.error("Label column %r not found. Available: %s", label_col, list(df.columns))
        sys.exit(1)

    # Drop rows with missing labels
    df = df.dropna(subset=[label_col])
    logger.info("Rows with valid labels: %d", len(df))

    # ── Feature engineering ───────────────────────────────────────────────
    df = engineer_features(df)

    # ── Time-aware train / validation split ───────────────────────────────
    # Sort by post date, use last 20% as validation
    if "created_utc" in df.columns:
        df = df.sort_values("created_utc")
    elif "post_date" in df.columns:
        df = df.sort_values("post_date")

    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    logger.info("Train: %d rows | Validation: %d rows", len(train_df), len(val_df))

    X_train = select_feature_matrix(train_df)
    y_train = train_df[label_col].values.astype(float)

    X_val = select_feature_matrix(val_df)
    y_val = val_df[label_col].values.astype(float)

    # ── Model training ────────────────────────────────────────────────────
    # HistGradientBoostingRegressor handles NaN natively and is fast.
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )

    logger.info("Training HistGradientBoostingRegressor…")
    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────
    preds_val = model.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, preds_val))
    mae = mean_absolute_error(y_val, preds_val)
    spearman_r, spearman_p = spearmanr(y_val, preds_val)

    logger.info("Validation metrics — RMSE: %.2f | MAE: %.2f | Spearman r: %.3f (p=%.4f)",
                rmse, mae, spearman_r, spearman_p)

    # ── Feature importance ────────────────────────────────────────────────
    importances = {}
    if hasattr(model, "feature_importances_"):
        for name, imp in zip(FEATURE_NAMES, model.feature_importances_):
            importances[name] = round(float(imp), 4)
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top feature importances:")
        for name, imp in sorted_imp[:8]:
            logger.info("  %-35s %.4f", name, imp)

    # ── Save model ────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "calibration_model.joblib"
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    # ── Save metadata ─────────────────────────────────────────────────────
    label_method = df.get("label_method", pd.Series(["unknown"]))[0] if "label_method" in df.columns else "unknown"

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 6),
        "feature_names": FEATURE_NAMES,
        "feature_importances": importances,
        "blend_alpha": blend_alpha,
        "label_method": str(label_method),
        "algorithm_version": "1.0.0",
        "input_file": str(path),
    }

    metadata_path = out_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Metadata saved to %s", metadata_path)

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML calibration model.")
    parser.add_argument("--input", required=True, help="Path to labeled CSV/Parquet dataset")
    parser.add_argument("--output-dir", default="trained_models/")
    parser.add_argument("--blend-alpha", type=float, default=0.4,
                        help="Physics weight in final blend (0=pure ML, 1=pure physics)")
    parser.add_argument("--label-col", default="beauty_label")
    args = parser.parse_args()

    train(
        input_path=args.input,
        output_dir=args.output_dir,
        blend_alpha=args.blend_alpha,
        label_col=args.label_col,
    )


if __name__ == "__main__":
    main()
