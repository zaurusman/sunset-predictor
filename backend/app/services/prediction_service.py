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
        """
        Run a full prediction for one location + date.

        When no weather_override is supplied the prediction uses four
        window snapshots (-15m, sunset, +15m, +30m) so the daily score
        reflects the best likely viewing moment rather than a single instant.
        The physics breakdown and weather summary are taken from the best
        window point.  When a weather_override is supplied the system falls
        back to single-snapshot mode (the override fully specifies conditions
        so there is nothing to vary across the window).
        """
        lat, lon = request.latitude, request.longitude
        horizon_deg = request.horizon_obstruction_deg

        target_date: date = request.target_date or local_sunset_date(lat, lon)
        sunset_time = self._astro.get_sunset_time(lat, lon, target_date)
        window_start, window_end = self._astro.get_best_viewing_window(sunset_time)

        # ------------------------------------------------------------------
        # Weather + scoring
        # ------------------------------------------------------------------
        if request.weather_override is not None:
            # Single-snapshot path (override controls the conditions)
            weather = await self._weather.get_snapshot_at_sunset(
                lat, lon, target_date, override=request.weather_override
            )
            single_result = self._scoring.score(weather, horizon_deg)
            window_result = self._scoring.score_window(
                [("sunset", single_result.physics_score)]
            )
            primary_result = single_result
            primary_weather = weather
        else:
            # Window path: score four snapshots, pick the best
            window_snaps = await self._weather.get_window_snapshots(
                lat, lon, target_date, sunset_time
            )
            scored: list[tuple[str, float]] = []
            snap_results: dict[str, tuple] = {}  # label → (ScoringResult, WeatherSnapshot)
            for snap in window_snaps:
                r = self._scoring.score(snap, horizon_deg)
                label = snap.timestamp_label or "sunset"
                scored.append((label, r.physics_score))
                snap_results[label] = (r, snap)

            window_result = self._scoring.score_window(scored)
            best_label = window_result.best_label
            primary_result, primary_weather = snap_results[best_label]

        # ------------------------------------------------------------------
        # ML calibration (applied to the window final score)
        # ------------------------------------------------------------------
        ml_score: Optional[float] = None
        ml_adjustment: Optional[float] = None
        if self._ml.is_loaded():
            ml_score = self._ml.predict_calibrated_score(
                weather=primary_weather,
                physics_score=window_result.final_score,
                target_date_or_month=target_date.month,
                horizon_obstruction_deg=horizon_deg,
            )

        final_score = self._ml.blend(window_result.final_score, ml_score)
        if ml_score is not None:
           ml_adjustment = round(final_score - window_result.final_score, 2)

        category = self._scoring.score_to_category(final_score)

        confidence = self._scoring.compute_confidence(
            weather=primary_weather,
            component_scores={
                "cloud_quality": primary_result.cloud_quality,
                "atmosphere": primary_result.atmosphere,
                "moisture": primary_result.moisture,
                "horizon": primary_result.horizon,
            },
            physics_score=final_score,
            has_ml=self._ml.is_loaded(),
            window_scores=list(window_result.window_scores.values()),
        )

        reasons = self._explanation.generate(
            weather=primary_weather,
            breakdown=primary_result.to_physics_breakdown(),
            category=category,
            window_result=window_result,
        )

        # Build a breakdown whose weighted_physics_score matches the displayed
        # score (final_score) rather than the raw single-snapshot score of the
        # best window point.  Component sub-scores still come from the best
        # window point so they correctly explain WHY the sky looks the way it does.
        breakdown = primary_result.to_physics_breakdown()
        breakdown.weighted_physics_score = round(final_score, 1)

        return PredictResponse(
            beauty_score_0_100=round(final_score, 1),
            category=category,
            confidence_0_100=round(confidence, 1),
            reasons=reasons,
            sunset_time=sunset_time,
            best_viewing_window_start=window_start,
            best_viewing_window_end=window_end,
            best_window_point=window_result.best_label,
            window_scores={k: round(v, 1) for k, v in window_result.window_scores.items()},
            go_outside_recommendation=window_result.go_outside,
            algorithm_version=self._settings.ALGORITHM_VERSION,
            ml_model_used=self._ml.is_loaded(),
            ml_adjustment=ml_adjustment,
            physics_component_breakdown=breakdown,
            weather_summary=_build_weather_summary(primary_weather),
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

        # Fetch window snapshots for all days in one batch API call
        daily_window_snaps = await self._weather.get_forecast_range_windows(
            lat, lon, days=request.days
        )

        # Score each day concurrently using the same window algorithm as predict()
        tasks = [
            self._score_day(lat, lon, d, window_snaps, horizon_deg)
            for d, window_snaps in daily_window_snaps
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
        window_snaps: list[WeatherSnapshot],
        horizon_deg: float,
    ) -> DayForecast:
        sunset_time = self._astro.get_sunset_time(lat, lon, target_date)
        window_start, window_end = self._astro.get_best_viewing_window(sunset_time)

        # Score all four window snapshots — mirrors predict() window path
        scored: list[tuple[str, float]] = []
        snap_results: dict[str, tuple] = {}
        for snap in window_snaps:
            r = self._scoring.score(snap, horizon_deg)
            label = snap.timestamp_label or "sunset"
            scored.append((label, r.physics_score))
            snap_results[label] = (r, snap)

        window_result = self._scoring.score_window(scored)
        best_label = window_result.best_label
        primary_result, primary_weather = snap_results[best_label]

        ml_score: Optional[float] = None
        if self._ml.is_loaded():
            ml_score = self._ml.predict_calibrated_score(
                weather=primary_weather,
                physics_score=window_result.final_score,
                target_date_or_month=target_date.month,
                horizon_obstruction_deg=horizon_deg,
            )

        final_score = self._ml.blend(window_result.final_score, ml_score)
        category = self._scoring.score_to_category(final_score)
        confidence = self._scoring.compute_confidence(
            weather=primary_weather,
            component_scores={
                "cloud_quality": primary_result.cloud_quality,
                "atmosphere": primary_result.atmosphere,
                "moisture": primary_result.moisture,
                "horizon": primary_result.horizon,
            },
            physics_score=final_score,
            has_ml=self._ml.is_loaded(),
            window_scores=list(window_result.window_scores.values()),
        )
        reasons = self._explanation.generate(
            weather=primary_weather,
            breakdown=primary_result.to_physics_breakdown(),
            category=category,
            window_result=window_result,
        )

        breakdown = primary_result.to_physics_breakdown()
        breakdown.weighted_physics_score = round(final_score, 1)

        return DayForecast(
            date=target_date,
            beauty_score_0_100=round(final_score, 1),
            category=category,
            confidence_0_100=round(confidence, 1),
            sunset_time=sunset_time,
            best_viewing_window_start=window_start,
            best_viewing_window_end=window_end,
            best_window_point=window_result.best_label,
            window_scores={k: round(v, 1) for k, v in window_result.window_scores.items()},
            go_outside_recommendation=window_result.go_outside,
            reasons=reasons,
            physics_component_breakdown=breakdown,
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
