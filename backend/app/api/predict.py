"""POST /predict endpoint — single-day sunset prediction."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.weather_service import WeatherBusyError, WeatherUnavailableError
from app.core.logging import get_logger
from app.utils.call_budget import OTHER, TONIGHT, priority
from app.utils.client_fetch import ClientFetchNeeded, browser_may_fetch, fetch_request
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
    # The location's date or the UTC date (which /forecast starts from): for a
    # few hours around midnight they differ, and both mean this evening — the
    # same rule as the corridor's (WeatherService.get_corridor_samples).
    tonight = body.target_date is None or body.target_date in (
        local_sunset_date(body.latitude, body.longitude),
        datetime.now(timezone.utc).date(),
    )
    try:
        # Tonight's calls go first and are never held back (see call_budget).
        # Tonight only: the browser may fetch what the server can't (see
        # app/utils/client_fetch.py).
        with priority(TONIGHT if tonight else OTHER), browser_may_fetch(tonight, body.client_data):
            return await svc.predict(body)
    except ClientFetchNeeded as need:
        logger.info("Tonight: asking the browser to fetch %s", [u.split("?")[0] for u in need.urls])
        return fetch_request(need)
    except WeatherBusyError as exc:
        # Only reached when it actually needed Open-Meteo calls: whatever is
        # cached (memory or the durable tier) is served even when the share
        # for non-tonight work is used up.
        logger.warning("%s held back to keep tonight working: %s", 'Prediction for another date', exc)
        raise HTTPException(status_code=503, detail="Other dates are busy right now — tonight's forecast still works. Try again in a minute.", headers={"Retry-After": "60"}) from exc
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
