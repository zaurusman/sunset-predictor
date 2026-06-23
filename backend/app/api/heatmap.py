"""GET /heatmap endpoint — historical sunset score heatmap."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.logging import get_logger
from app.schemas.heatmap import HeatmapResponse
from app.services.weather_service import WeatherUnavailableError

logger = get_logger(__name__)
router = APIRouter(tags=["heatmap"])


@router.get("/heatmap", response_model=HeatmapResponse, summary="Historical sunset score heatmap")
async def get_heatmap(
    request: Request,
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    months: int = Query(default=12, ge=1, le=24, description="How many months of history to return"),
) -> HeatmapResponse:
    """
    Return historical sunset scores for the past *months* months.

    Data is fetched from the Open-Meteo archive API in a single batch call
    and cached for 24 hours (historical data never changes).
    """
    svc = request.app.state.prediction_service
    try:
        return await svc.heatmap(lat=lat, lon=lon, months=months)
    except WeatherUnavailableError as exc:
        logger.warning("Weather provider unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Weather data provider is temporarily rate-limited or unavailable. Please try again shortly.",
            headers={"Retry-After": "30"},
        ) from exc
    except Exception as exc:
        logger.error("Heatmap failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Heatmap error: {exc}") from exc
