"""POST /forecast endpoint — multi-day sunset forecast."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.forecast import ForecastRequest, ForecastResponse
from app.services.weather_service import WeatherUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["forecast"])


@router.post("/forecast", response_model=ForecastResponse, summary="Multi-day sunset forecast")
async def forecast_sunset(
    body: ForecastRequest,
    request: Request,
) -> ForecastResponse:
    """
    Predict sunset beauty scores for the next N days (default 7, max 16).

    Returns one `DayForecast` entry per day, each with a score, category,
    confidence, sunset time, best viewing window, and explanations.
    """
    svc = request.app.state.prediction_service
    try:
        return await svc.forecast(body)
    except WeatherUnavailableError as exc:
        logger.warning("Weather provider unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Weather data provider is temporarily rate-limited or unavailable. Please try again shortly.",
            headers={"Retry-After": "30"},
        ) from exc
    except Exception as exc:
        logger.error("Forecast failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecast error: {exc}") from exc
