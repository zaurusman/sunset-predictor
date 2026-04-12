"""
Sunset Predictor — FastAPI application entry point.

Service wiring happens in the lifespan context manager so that all
components are properly initialised before the first request and
cleanly shut down on exit.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, predict, forecast, model_info, geocode
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.models.ml_model import MLModel
from app.models.model_registry import ModelRegistry
from app.services.astronomy_service import AstronomyService
from app.services.explanation_engine import ExplanationEngine
from app.services.prediction_service import PredictionService
from app.services.scoring_engine import ScoringEngine
from app.services.weather_service import WeatherService
from app.utils.cache import TTLCache

setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise services on startup; clean up on shutdown."""
    logger.info("Starting Sunset Predictor (env=%s, version=%s)", settings.APP_ENV, settings.ALGORITHM_VERSION)

    # Shared HTTP client (connection-pooled, async)
    http_client = httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT)

    # Infrastructure
    cache = TTLCache(ttl_seconds=settings.CACHE_TTL_SECONDS)
    registry = ModelRegistry(settings=settings)

    # Services
    astro_service = AstronomyService()
    weather_service = WeatherService(
        http_client=http_client,
        astro_service=astro_service,
        cache=cache,
        settings=settings,
    )
    scoring_engine = ScoringEngine()
    explanation_engine = ExplanationEngine()

    # ML model (gracefully no-ops if not trained yet)
    ml_model = MLModel(registry=registry, settings=settings)
    ml_model.load()

    # Orchestration
    prediction_service = PredictionService(
        weather_service=weather_service,
        astro_service=astro_service,
        scoring_engine=scoring_engine,
        explanation_engine=explanation_engine,
        ml_model=ml_model,
        settings=settings,
    )

    # Attach to app state for injection via Request
    app.state.settings = settings
    app.state.prediction_service = prediction_service
    app.state.ml_model = ml_model

    logger.info("All services initialised. ML model loaded: %s", ml_model.is_loaded())

    yield  # ← application runs here

    logger.info("Shutting down…")
    await http_client.aclose()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sunset Predictor",
        description=(
            "Predicts how beautiful a sunset will be for any location and date. "
            "Uses a physics-informed scoring engine calibrated by an optional ML model "
            "trained on Reddit sunset engagement data and historical weather."
        ),
        version=settings.ALGORITHM_VERSION,
        lifespan=lifespan,
    )

    # CORS — allow all origins in dev; restrict in production via reverse proxy
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(forecast.router)
    app.include_router(model_info.router)
    app.include_router(geocode.router)

    return app


app = create_app()
