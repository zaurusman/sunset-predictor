"""Request / response schemas for the /forecast endpoint."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.prediction import PhysicsBreakdown, SunsetCategory


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ForecastRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    days: int = Field(default=7, ge=1, le=16, description="Number of days to forecast")
    horizon_obstruction_deg: float = Field(default=2.0, ge=0, le=90)


# ---------------------------------------------------------------------------
# Per-day result
# ---------------------------------------------------------------------------


class DayForecast(BaseModel):
    """Prediction for a single day in the forecast window."""

    date: date
    beauty_score_0_100: float = Field(ge=0, le=100)
    category: SunsetCategory
    confidence_0_100: float = Field(ge=0, le=100)

    sunset_time: datetime
    best_viewing_window_start: datetime
    best_viewing_window_end: datetime

    reasons: list[str]
    physics_component_breakdown: PhysicsBreakdown
    ml_model_used: bool


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class ForecastResponse(BaseModel):
    days: list[DayForecast]
    location: dict[str, float]
    algorithm_version: str
    generated_at: datetime
