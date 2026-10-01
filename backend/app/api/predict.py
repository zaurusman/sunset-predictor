"""POST /predict endpoint — single-day sunset prediction."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.weather_service import WeatherUnavailableError
from app.core.logging import get_logger
from app.utils.call_budget import OTHER, TONIGHT, priority
from app.utils.time_utils import local_sunset_date

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
    tonight = body.target_date is None or body.target_date == local_sunset_date(
        body.latitude, body.longitude
    )
    if not tonight:
        budget = getattr(request.app.state, "call_budget", None)
        if budget is not None and not budget.optional_work_allowed(headroom=20.0):
            # Another date: Open-Meteo's remaining share is kept for tonight.
            raise HTTPException(
                status_code=503,
                detail="Other dates are busy right now — tonight's forecast still works. Try again later.",
                headers={"Retry-After": "600"},
            )
    try:
        # Tonight's calls go first and are never held back (see call_budget).
        with priority(TONIGHT if tonight else OTHER):
            return await svc.predict(body)
    except WeatherUnavailableError as exc:
        logger.warning("Weather provider unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Weather data provider is temporarily rate-limited or unavailable. Please try again shortly.",
            headers={"Retry-After": "30"},
        ) from exc
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
