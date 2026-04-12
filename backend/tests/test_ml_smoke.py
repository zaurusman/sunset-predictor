"""
Smoke tests for the ML training pipeline.

Trains on a tiny synthetic dataset and verifies that the saved model
can be loaded and produces valid predictions.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.config import Settings
from app.models.ml_model import MLModel, FEATURE_NAMES
from app.models.model_registry import ModelRegistry
from app.schemas.weather import WeatherSnapshot


def _make_synthetic_dataset(n: int = 120) -> pd.DataFrame:
    """Create a minimal synthetic training dataset."""
    rng = np.random.default_rng(42)

    rows = {
        "post_id": [f"abc{i}" for i in range(n)],
        "title": ["sunset photo"] * n,
        "subreddit": ["sunset"] * n,
        "created_utc": [1_700_000_000 + i * 3600 for i in range(n)],
        "score": rng.integers(50, 5000, size=n).tolist(),
        "num_comments": rng.integers(5, 200, size=n).tolist(),
        "is_image": [True] * n,
        "post_date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d").tolist(),
        "location_lat": [37.77] * n,
        "location_lon": [-122.41] * n,
        "cloud_cover_low": rng.uniform(0, 80, n).tolist(),
        "cloud_cover_mid": rng.uniform(0, 60, n).tolist(),
        "cloud_cover_high": rng.uniform(0, 100, n).tolist(),
        "cloud_cover": rng.uniform(0, 100, n).tolist(),
        "visibility_m": rng.uniform(5000, 30000, n).tolist(),
        "relative_humidity": rng.uniform(30, 95, n).tolist(),
        "dewpoint_c": rng.uniform(0, 20, n).tolist(),
        "temperature_c": rng.uniform(10, 30, n).tolist(),
        "precipitation_mm": rng.exponential(0.3, n).tolist(),
        "wind_speed_kmh": rng.uniform(0, 40, n).tolist(),
        "pressure_hpa": rng.uniform(990, 1025, n).tolist(),
    }
    df = pd.DataFrame(rows)
    # Build label from score (same as build_reddit_dataset.py)
    df["beauty_label"] = df["score"].rank(pct=True) * 100.0
    df["label_method"] = "percentile"
    return df


def test_train_and_predict_on_synthetic_data(tmp_path: Path) -> None:
    """
    Full smoke test: write synthetic CSV → run train → load model → predict.
    """
    from scripts.train_model import train

    # Write dataset
    df = _make_synthetic_dataset()
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)

    # Train
    out_dir = tmp_path / "models"
    train(
        input_path=str(csv_path),
        output_dir=str(out_dir),
        blend_alpha=0.4,
    )

    # Verify artifacts
    model_path = out_dir / "calibration_model.joblib"
    metadata_path = out_dir / "model_metadata.json"
    assert model_path.exists(), "calibration_model.joblib not found"
    assert metadata_path.exists(), "model_metadata.json not found"

    # Load and predict
    settings = Settings(
        MODEL_PATH=str(model_path),
        MODEL_METADATA_PATH=str(metadata_path),
    )
    registry = ModelRegistry(settings=settings)
    ml_model = MLModel(registry=registry, settings=settings)
    loaded = ml_model.load()
    assert loaded, "Model should load successfully"

    # Build a test weather snapshot
    weather = WeatherSnapshot(
        cloud_low=10.0, cloud_mid=25.0, cloud_high=45.0, cloud_total=60.0,
        visibility_m=18_000.0, relative_humidity=55.0, dewpoint_c=8.0,
        temperature_c=18.0, precipitation_mm=0.0, wind_speed_kmh=10.0,
        pressure_hpa=1013.0, aerosol_optical_depth=0.15,
        sun_elevation_deg=2.0, data_source="override", aerosol_is_estimated=False,
    )
    score = ml_model.predict_calibrated_score(
        weather=weather,
        physics_score=65.0,
        target_date_or_month=6,
        horizon_obstruction_deg=2.0,
    )
    assert score is not None, "predict_calibrated_score returned None"
    assert 0 <= score <= 100, f"Score {score} out of [0, 100]"


def test_model_metadata_structure(tmp_path: Path) -> None:
    """Metadata JSON should contain required keys."""
    from scripts.train_model import train

    df = _make_synthetic_dataset(n=60)
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "models"
    train(str(csv_path), str(out_dir))

    with open(out_dir / "model_metadata.json") as f:
        meta = json.load(f)

    required_keys = ["trained_at", "n_train", "n_val", "rmse", "mae", "feature_names"]
    for key in required_keys:
        assert key in meta, f"Missing key in metadata: {key}"

    assert meta["n_train"] > 0
    assert meta["n_val"] > 0
    assert meta["rmse"] >= 0


def test_ml_blend_is_in_range(tmp_path: Path) -> None:
    """Blended score must remain in [0, 100]."""
    from scripts.train_model import train

    df = _make_synthetic_dataset(n=60)
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "models"
    train(str(csv_path), str(out_dir))

    settings = Settings(
        MODEL_PATH=str(out_dir / "calibration_model.joblib"),
        MODEL_METADATA_PATH=str(out_dir / "model_metadata.json"),
        ML_BLEND_ALPHA=0.4,
    )
    registry = ModelRegistry(settings=settings)
    ml = MLModel(registry=registry, settings=settings)
    ml.load()

    for physics_score in [0.0, 25.0, 50.0, 75.0, 100.0]:
        blended = ml.blend(physics_score, 55.0)
        assert 0 <= blended <= 100, f"Blend out of range for physics={physics_score}: {blended}"
