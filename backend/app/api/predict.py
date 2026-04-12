"""POST /predict endpoint — single-day sunset prediction."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.prediction import PredictRequest, PredictResponse
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictResponse, summary="Predict sunset beauty")
async def predict_sunset(
    body: PredictRequest,
    request: Request,
) -> PredictResponse:
    """
    Predict the beauty score for a sunset at the given location and (optional) date.

    - If `target_date` is omitted, defaults to **today** in the location's timezone.
    - Dates in the past use Open-Meteo **archive** data.
    - Future dates (up to 16 days) use Open-Meteo **forecast** data.
    - Supply `weather_override` to inject custom weather values (useful for testing).
    """
    svc = request.app.state.prediction_service
    try:
        return await svc.predict(body)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
