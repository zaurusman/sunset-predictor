"""POST /forecast endpoint — multi-day sunset forecast."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.forecast import ForecastRequest, ForecastResponse
from app.services.weather_service import WeatherBusyError, WeatherUnavailableError
from app.core.logging import get_logger
from app.utils.client_fetch import ClientFetchNeeded, browser_may_fetch, fetch_request

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
        # The browser may fetch what the server can't (see app/utils/client_fetch.py).
        with browser_may_fetch(True, body.client_data):
            return await svc.forecast(body)
    except ClientFetchNeeded as need:
        logger.info("7-day forecast: asking the browser to fetch %s", [u.split("?")[0] for u in need.urls])
        return fetch_request(need)
    except WeatherBusyError as exc:
        # Only reached when it actually needed Open-Meteo calls: whatever is
        # cached (memory or the durable tier) is served even when the share
        # for non-tonight work is used up.
        logger.warning("%s held back to keep tonight working: %s", '7-day forecast', exc)
        raise HTTPException(status_code=503, detail="The 7-day forecast is busy right now — tonight's forecast still works. Try again in a minute.", headers={"Retry-After": "60"}) from exc
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
