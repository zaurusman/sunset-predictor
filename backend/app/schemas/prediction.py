"""Request / response schemas for the /predict endpoint."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.weather import WeatherOverride


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Single-day sunset prediction request."""

    latitude: float = Field(..., ge=-90, le=90, description="Decimal latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Decimal longitude")
    target_date: Optional[date] = Field(
        default=None,
        description=(
            "Date to predict. Defaults to today in the location's approximate timezone. "
            "Dates up to 16 days in the future use forecast data; "
            "dates in the past use historical archive data."
        ),
    )
    horizon_obstruction_deg: float = Field(
        default=2.0,
        ge=0,
        le=90,
        description=(
            "Estimated horizon obstruction in degrees. "
            "0 = open ocean or flat field; 5 = gentle hills; 15+ = urban buildings / mountains."
        ),
    )
    weather_override: Optional[WeatherOverride] = Field(
        default=None,
        description="Manually override weather fields (useful for testing).",
    )


# ---------------------------------------------------------------------------
# Response building blocks
# ---------------------------------------------------------------------------


class PhysicsBreakdown(BaseModel):
    """Scores for each physics component before blending with ML."""

    cloud_quality_score: float = Field(ge=0, le=100)
    atmosphere_score: float = Field(ge=0, le=100)
    moisture_score: float = Field(ge=0, le=100)
    horizon_score: float = Field(ge=0, le=100)
    weighted_physics_score: float = Field(ge=0, le=100)
    component_weights: dict[str, float]

    # Afterglow potential (0–100). Non-zero only when sun is below the horizon
    # and conditions support afterglow (high clouds present, not overcast).
    # This is an explanatory field — the effect is already baked into
    # cloud_quality_score; it is NOT added to weighted_physics_score again.
    afterglow_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Afterglow illumination potential when sun is below the horizon. "
            "Non-zero only at window points after sunset (sun < 0°)."
        ),
    )


class WeatherSummary(BaseModel):
    """Human-readable weather snapshot for display."""

    cloud_low_pct: float
    cloud_mid_pct: float
    cloud_high_pct: float
    cloud_total_pct: float
    visibility_km: float
    precipitation_mm: float
    aerosol_optical_depth: Optional[float]
    aerosol_is_estimated: bool
    temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

SunsetCategory = Literal["Poor", "Decent", "Good", "Great", "Epic"]


class PredictResponse(BaseModel):
    """Full sunset prediction response."""

    beauty_score_0_100: float = Field(ge=0, le=100)
    category: SunsetCategory
    confidence_0_100: float = Field(ge=0, le=100)

    # Natural-language explanations ordered by importance (3–6 items)
    reasons: list[str]

    # Informational timing (does NOT affect the score)
    sunset_time: datetime
    best_viewing_window_start: datetime
    best_viewing_window_end: datetime

    # Window-based scoring results
    best_window_point: str = Field(
        default="sunset",
        description="The window position that scored highest: '-15m' | 'sunset' | '+15m' | '+30m'",
    )
    window_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-window-point physics scores, keyed by label",
    )
    go_outside_recommendation: bool = Field(
        default=False,
        description="True when conditions are worth going outside for",
    )

    # Metadata
    algorithm_version: str
    ml_model_used: bool
    ml_adjustment: Optional[float] = Field(
        default=None,
        description="Raw adjustment applied by the ML model (positive = boosted, negative = reduced)",
    )

    physics_component_breakdown: PhysicsBreakdown
    weather_summary: WeatherSummary

    location: dict[str, float] = Field(description='{"latitude": …, "longitude": …}')
    requested_at: datetime
