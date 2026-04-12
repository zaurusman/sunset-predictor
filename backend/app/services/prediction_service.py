"""
Prediction orchestration service.

Coordinates astronomy → weather → scoring → ML calibration → explanation
to produce a complete PredictResponse or ForecastResponse.
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Optional

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.ml_model import MLModel
from app.schemas.forecast import DayForecast, ForecastRequest, ForecastResponse
from app.schemas.prediction import (
    PredictRequest,
    PredictResponse,
    WeatherSummary,
)
from app.schemas.weather import WeatherSnapshot
from app.services.astronomy_service import AstronomyService
from app.services.explanation_engine import ExplanationEngine
from app.services.scoring_engine import ScoringEngine
from app.services.weather_service import WeatherService
from app.utils.math_utils import clamp
from app.utils.time_utils import local_sunset_date, utcnow

logger = get_logger(__name__)


class PredictionService:
    """
    Top-level orchestrator for sunset predictions.

    Injected with all required services at startup.
    """

    def __init__(
        self,
        weather_service: WeatherService,
        astro_service: AstronomyService,
        scoring_engine: ScoringEngine,
        explanation_engine: ExplanationEngine,
        ml_model: MLModel,
        settings: Settings,
    ) -> None:
        self._weather = weather_service
        self._astro = astro_service
        self._scoring = scoring_engine
        self._explanation = explanation_engine
        self._ml = ml_model
        self._settings = settings

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    async def predict(self, request: PredictRequest) -> PredictResponse:
        """Run a full prediction for one location + date."""
        lat, lon = request.latitude, request.longitude
        horizon_deg = request.horizon_obstruction_deg

        # Resolve the target date (defaults to today in the local timezone)
        target_date: date = request.target_date or local_sunset_date(lat, lon)

        # Get sunset time and viewing window
        sunset_time = self._astro.get_sunset_time(lat, lon, target_date)
        window_start, window_end = self._astro.get_best_viewing_window(sunset_time)

        # Fetch (or override) weather
        weather = await self._weather.get_snapshot_at_sunset(
            lat, lon, target_date, override=request.weather_override
        )

        # Physics scoring
        result = self._scoring.score(weather, horizon_deg)

        # ML calibration
        ml_score: Optional[float] = None
        ml_adjustment: Optional[float] = None
        if self._ml.is_loaded():
            ml_score = self._ml.predict_calibrated_score(
                weather=weather,
                physics_score=result.physics_score,
                target_date_or_month=target_date.month,
                horizon_obstruction_deg=horizon_deg,
            )

        final_score = self._ml.blend(result.physics_score, ml_score)
        if ml_score is not None:
            ml_adjustment = round(final_score - result.physics_score, 2)

        # Category
        category = self._scoring.score_to_category(final_score)

        # Confidence (updated with ML knowledge)
        confidence = self._scoring.compute_confidence(
            weather=weather,
            component_scores={
                "cloud_quality": result.cloud_quality,
                "atmosphere": result.atmosphere,
                "moisture": result.moisture,
                "horizon": result.horizon,
            },
            physics_score=final_score,
            has_ml=self._ml.is_loaded(),
        )

        # Explanations
        reasons = self._explanation.generate(
            weather=weather,
            breakdown=result.to_physics_breakdown(),
            category=category,
        )

        return PredictResponse(
            beauty_score_0_100=round(final_score, 1),
            category=category,
            confidence_0_100=round(confidence, 1),
            reasons=reasons,
            sunset_time=sunset_time,
            best_viewing_window_start=window_start,
            best_viewing_window_end=window_end,
            algorithm_version=self._settings.ALGORITHM_VERSION,
            ml_model_used=self._ml.is_loaded(),
            ml_adjustment=ml_adjustment,
            physics_component_breakdown=result.to_physics_breakdown(),
            weather_summary=_build_weather_summary(weather),
            location={"latitude": lat, "longitude": lon},
            requested_at=utcnow(),
        )

    # ------------------------------------------------------------------
    # Multi-day forecast
    # ------------------------------------------------------------------

    async def forecast(self, request: ForecastRequest) -> ForecastResponse:
        """Run predictions for *days* consecutive days starting from today."""
        lat, lon = request.latitude, request.longitude
        horizon_deg = request.horizon_obstruction_deg

        # Fetch weather snapshots for all days in one batch
        daily_snapshots = await self._weather.get_forecast_range(
            lat, lon, days=request.days
        )

        # Score each day concurrently
        tasks = [
            self._score_day(lat, lon, d, snap, horizon_deg)
            for d, snap in daily_snapshots
        ]
        day_forecasts = await asyncio.gather(*tasks, return_exceptions=True)

        valid_days: list[DayForecast] = []
        for item in day_forecasts:
            if isinstance(item, Exception):
                logger.warning("Day forecast failed: %s", item)
            else:
                valid_days.append(item)

        return ForecastResponse(
            days=valid_days,
            location={"latitude": lat, "longitude": lon},
            algorithm_version=self._settings.ALGORITHM_VERSION,
            generated_at=utcnow(),
        )

    # ------------------------------------------------------------------
    # Internal: score a single day (used by forecast)
    # ------------------------------------------------------------------

    async def _score_day(
        self,
        lat: float,
        lon: float,
        target_date: date,
        weather: WeatherSnapshot,
        horizon_deg: float,
    ) -> DayForecast:
        sunset_time = self._astro.get_sunset_time(lat, lon, target_date)
        window_start, window_end = self._astro.get_best_viewing_window(sunset_time)

        result = self._scoring.score(weather, horizon_deg)

        ml_score: Optional[float] = None
        if self._ml.is_loaded():
            ml_score = self._ml.predict_calibrated_score(
                weather=weather,
                physics_score=result.physics_score,
                target_date_or_month=target_date.month,
                horizon_obstruction_deg=horizon_deg,
            )

        final_score = self._ml.blend(result.physics_score, ml_score)
        category = self._scoring.score_to_category(final_score)
        confidence = self._scoring.compute_confidence(
            weather=weather,
            component_scores={
                "cloud_quality": result.cloud_quality,
                "atmosphere": result.atmosphere,
                "moisture": result.moisture,
                "horizon": result.horizon,
            },
            physics_score=final_score,
            has_ml=self._ml.is_loaded(),
        )
        reasons = self._explanation.generate(
            weather=weather,
            breakdown=result.to_physics_breakdown(),
            category=category,
        )

        return DayForecast(
            date=target_date,
            beauty_score_0_100=round(final_score, 1),
            category=category,
            confidence_0_100=round(confidence, 1),
            sunset_time=sunset_time,
            best_viewing_window_start=window_start,
            best_viewing_window_end=window_end,
            reasons=reasons,
            physics_component_breakdown=result.to_physics_breakdown(),
            ml_model_used=self._ml.is_loaded(),
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_weather_summary(weather: WeatherSnapshot) -> WeatherSummary:
    return WeatherSummary(
        cloud_low_pct=round(weather.cloud_low, 1),
        cloud_mid_pct=round(weather.cloud_mid, 1),
        cloud_high_pct=round(weather.cloud_high, 1),
        cloud_total_pct=round(weather.cloud_total, 1),
        visibility_km=round(weather.visibility_m / 1000.0, 1),
        precipitation_mm=round(weather.precipitation_mm, 2),
        aerosol_optical_depth=(
            round(weather.aerosol_optical_depth, 3)
            if weather.aerosol_optical_depth is not None
            else None
        ),
        aerosol_is_estimated=weather.aerosol_is_estimated,
        temperature_c=round(weather.temperature_c, 1),
        humidity_pct=round(weather.relative_humidity, 1),
        wind_speed_kmh=round(weather.wind_speed_kmh, 1),
    )
