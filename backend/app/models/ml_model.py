"""
ML calibration model wrapper.

The model learns a calibrated beauty score from labeled data (Reddit-derived
engagement scores joined with historical weather).  It is OPTIONAL — the app
works fully in physics-only mode when no trained model is available.

Blending formula:
    final_score = alpha * physics_score + (1 - alpha) * ml_prediction

where alpha is configured via ML_BLEND_ALPHA (default 0.4).

The feature vector has 16 dimensions (no time-of-day features — the score is
per-day, not per-moment).
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.model_registry import ModelRegistry
from app.schemas.weather import WeatherSnapshot
from app.utils.math_utils import clamp

logger = get_logger(__name__)

# Median aerosol imputation value used when AOD is unavailable at inference time
_AOD_MEDIAN_IMPUTE = 0.15

# Feature names in order — must match training script
FEATURE_NAMES = [
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "cloud_cover",
    "log_visibility_m",
    "relative_humidity",
    "dewpoint_c",
    "log1p_precipitation",
    "wind_speed_kmh",
    "pressure_hpa",
    "aerosol_optical_depth",
    "sin_month",
    "cos_month",
    "sun_elevation_deg",
    "physics_score",
    "horizon_obstruction_deg",
]


class MLModel:
    """
    Wraps a trained HistGradientBoostingRegressor (or any sklearn-compatible model)
    for sunset beauty score calibration.

    All public methods are safe to call even when no model is loaded:
    `predict_calibrated_score` returns None in that case.
    """

    def __init__(self, registry: ModelRegistry, settings: Settings) -> None:
        self._registry = registry
        self._settings = settings
        self._model: Any = None
        self._metadata: dict = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Attempt to load the model from disk.

        Returns True if successful, False if no model file exists.
        Logs a warning (not an error) when no model is present — this is
        expected before the first training run.
        """
        if not self._registry.model_exists():
            logger.info(
                "No trained model found at %s — running in physics-only mode.",
                self._registry.model_path,
            )
            self._loaded = False
            return False

        try:
            import joblib

            self._model = joblib.load(self._registry.model_path)
            self._metadata = self._registry.load_metadata()
            self._loaded = True
            logger.info(
                "Loaded ML calibration model from %s (trained %s)",
                self._registry.model_path,
                self._metadata.get("trained_at", "unknown"),
            )
            return True
        except Exception as exc:
            logger.error("Failed to load ML model: %s — physics-only mode active.", exc)
            self._loaded = False
            return False

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_calibrated_score(
        self,
        weather: WeatherSnapshot,
        physics_score: float,
        target_date_or_month: int,
        horizon_obstruction_deg: float,
    ) -> Optional[float]:
        """
        Return a calibrated beauty score in [0, 100], or None if not loaded.

        Args:
            weather: Normalised weather snapshot at sunset.
            physics_score: Raw physics score before calibration.
            target_date_or_month: Calendar month (1–12) of the prediction date.
            horizon_obstruction_deg: Horizon obstruction in degrees.
        """
        if not self._loaded or self._model is None:
            return None

        try:
            features = self.build_feature_vector(
                weather=weather,
                physics_score=physics_score,
                month=target_date_or_month,
                horizon_obstruction_deg=horizon_obstruction_deg,
            )
            prediction = float(self._model.predict(features.reshape(1, -1))[0])
            return clamp(prediction, lo=0.0, hi=100.0)
        except Exception as exc:
            logger.warning("ML inference failed: %s — using physics score.", exc)
            return None

    def blend(self, physics_score: float, ml_score: Optional[float]) -> float:
        """
        Blend physics and ML scores.

        final = alpha * physics + (1 - alpha) * ml
        Falls back to physics_score when ml_score is None.
        """
        if ml_score is None:
            return physics_score
        alpha = self._settings.ML_BLEND_ALPHA
        return clamp(alpha * physics_score + (1.0 - alpha) * ml_score)

    def get_metadata(self) -> dict:
        """Return model metadata dict (empty dict when no model is loaded)."""
        if not self._loaded:
            return {
                "loaded": False,
                "message": "No trained model available — running physics-only mode.",
            }
        meta = dict(self._metadata)
        meta["loaded"] = True
        return meta

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def build_feature_vector(
        self,
        weather: WeatherSnapshot,
        physics_score: float,
        month: int,
        horizon_obstruction_deg: float,
    ) -> np.ndarray:
        """
        Build the 16-element feature vector used during training.

        Feature order must be identical to the training script.
        """
        aod = weather.aerosol_optical_depth
        if aod is None:
            aod = _AOD_MEDIAN_IMPUTE

        return np.array([
            weather.cloud_low,
            weather.cloud_mid,
            weather.cloud_high,
            weather.cloud_total,
            math.log(weather.visibility_m + 1.0),
            weather.relative_humidity,
            weather.dewpoint_c,
            math.log1p(weather.precipitation_mm),
            weather.wind_speed_kmh,
            weather.pressure_hpa,
            aod,
            math.sin(month * 2.0 * math.pi / 12.0),   # seasonality (sine)
            math.cos(month * 2.0 * math.pi / 12.0),   # seasonality (cosine)
            weather.sun_elevation_deg,
            physics_score,
            horizon_obstruction_deg,
        ], dtype=np.float64)
