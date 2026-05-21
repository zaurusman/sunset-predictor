"""Request / response schemas for the /heatmap endpoint."""
from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from app.schemas.prediction import SunsetCategory


class HeatmapDay(BaseModel):
    date: date
    score: float = Field(ge=0, le=100)
    category: SunsetCategory


class HeatmapResponse(BaseModel):
    days: list[HeatmapDay]
    location: dict[str, float]
    generated_at: datetime
