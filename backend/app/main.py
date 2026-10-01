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
from starlette.responses import JSONResponse

from app.api import health, predict, forecast, heatmap, model_info, geocode, submit, rate, push
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.models.ml_model import MLModel
from app.models.model_registry import ModelRegistry
from app.services.alert_service import AlertService, prediction_predictor
from app.services.astronomy_service import AstronomyService
from app.services.climatology_service import ClimatologyService
from app.services.model_runs import ModelRunClock
from app.services.explanation_engine import ExplanationEngine
from app.services.prediction_service import PredictionService
from app.services.push_sender import WebPushSender
from app.services.rating_store import RatingStore
from app.services.scoring_engine import ScoringEngine
from app.services.subscription_store import PostgresSubscriptionStore
from app.services.weather_service import WeatherService
from app.utils.cache import TTLCache
from app.utils.call_budget import CallBudget, client_key, current_client
from app.utils.durable_cache import PostgresCacheTier
from app.utils.time_utils import local_sunset_date

setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise services on startup; clean up on shutdown."""
    logger.info("Starting Sunset Predictor (env=%s, version=%s)", settings.APP_ENV, settings.ALGORITHM_VERSION)
    logger.info(
        "Email config — resend_key_set=%s, from=%r, dev_email=%r",
        bool(settings.RESEND_API_KEY),
        settings.RESEND_FROM_EMAIL,
        settings.DEVELOPER_EMAIL,
    )

    # Shared HTTP client (connection-pooled, async)
    http_client = httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT)

    # Infrastructure
    cache = TTLCache(
        ttl_seconds=settings.CACHE_TTL_SECONDS,
        persist_path=settings.CACHE_PERSIST_PATH or None,
        stale_grace_seconds=settings.CACHE_STALE_GRACE_SECONDS,
        memory_budget_bytes=int(settings.CACHE_MEMORY_BUDGET_MB * 1e6),
        persist_interval=settings.CACHE_PERSIST_INTERVAL_SECONDS,
    )
    logger.info(
        "Weather cache: ttl=%ss, stale_grace=%ss, budget=%.0f MB, persist=%s",
        settings.CACHE_TTL_SECONDS,
        settings.CACHE_STALE_GRACE_SECONDS,
        settings.CACHE_MEMORY_BUDGET_MB,
        settings.CACHE_PERSIST_PATH or "disabled",
    )
    budget = CallBudget(
        client_hourly_limit=settings.RATE_LIMIT_CLIENT_HOURLY_CALLS,
        daily_soft_cap=settings.OPEN_METEO_DAILY_SOFT_CAP,
    )
    registry = ModelRegistry(settings=settings)
    rating_store = RatingStore(path=settings.RATINGS_PATH)

    # Services
    astro_service = AstronomyService()
    weather_service = WeatherService(
        http_client=http_client,
        astro_service=astro_service,
        cache=cache,
        settings=settings,
        budget=budget,
        runs=ModelRunClock(http_client, settings) if settings.MODEL_RUN_TRACKING else None,
    )
    scoring_engine = ScoringEngine()
    explanation_engine = ExplanationEngine()

    # ML model (gracefully no-ops if not trained yet)
    ml_model = MLModel(registry=registry, settings=settings)
    ml_model.load()

    # Local climatology — turns the raw physics score into a rank against this
    # location's own history. Shares the persisted cache, so a warmed curve
    # survives restarts.
    climatology = ClimatologyService(
        weather_service=weather_service,
        astro_service=astro_service,
        scoring_engine=scoring_engine,
        cache=cache,
    )

    # Orchestration
    prediction_service = PredictionService(
        weather_service=weather_service,
        astro_service=astro_service,
        scoring_engine=scoring_engine,
        explanation_engine=explanation_engine,
        ml_model=ml_model,
        settings=settings,
        climatology=climatology,
    )

    # Epic-sunset push alerts — optional. Without a database or VAPID key the
    # app runs exactly as before and the /push endpoints answer 503.
    subscription_store = None
    alert_service = None
    if settings.DATABASE_URL:
        try:
            subscription_store = await PostgresSubscriptionStore.connect(settings.DATABASE_URL)
        except Exception as exc:
            logger.error("Push alerts disabled — could not connect to DATABASE_URL: %s", exc)
    if subscription_store is not None and settings.VAPID_PRIVATE_KEY:
        alert_service = AlertService(
            store=subscription_store,
            predictor=prediction_predictor(prediction_service),
            sunset_for=astro_service.get_sunset_time,
            local_date_for=local_sunset_date,
            sender=WebPushSender(settings.VAPID_PRIVATE_KEY, settings.VAPID_SUBJECT),
            lead_min_hours=settings.ALERT_LEAD_MIN_HOURS,
            lead_max_hours=settings.ALERT_LEAD_MAX_HOURS,
            decimals=settings.CACHE_COORD_DECIMALS,
        )
    # Durable tier for the long-lived weather cache (climatology curves,
    # archive months, frozen evenings). Render wipes /tmp on every deploy, and
    # rebuilding every location's climatology from Open-Meteo afterwards is a
    # suspected source of the intermittent 503s. Reuses the pool above.
    if subscription_store is not None:
        try:
            tier = PostgresCacheTier(subscription_store.pool)
            await tier.ensure_schema()
            await cache.attach_durable(tier)
        except Exception as exc:
            logger.error("Durable weather cache disabled: %s", exc)
    logger.info(
        "Push alerts: store=%s, sender=%s",
        "postgres" if subscription_store else "off",
        "on" if alert_service else "off",
    )

    # Attach to app state for injection via Request
    app.state.settings = settings
    app.state.call_budget = budget
    app.state.prediction_service = prediction_service
    app.state.ml_model = ml_model
    app.state.rating_store = rating_store
    app.state.subscription_store = subscription_store
    app.state.alert_service = alert_service

    logger.info(
        "All services initialised. ML model loaded: %s. Ratings: %d stored at %s",
        ml_model.is_loaded(), rating_store.count(), rating_store.path,
    )

    yield  # ← application runs here

    logger.info("Shutting down…")
    await http_client.aclose()
    await cache.close_durable()
    cache.close()
    if subscription_store is not None:
        await subscription_store.close()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sunset Predictor",
        description=(
            "Predicts how beautiful a sunset will be for any location and date. "
            "Scoring is physics-based. The optional ML calibration branch is disabled — "
            "see data/dead/README.md."
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

    # Added last, so it runs first; its own 429 carries the CORS header.
    app.add_middleware(CallBudgetMiddleware)

    # Routers
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(forecast.router)
    app.include_router(heatmap.router)
    app.include_router(model_info.router)
    app.include_router(geocode.router)
    app.include_router(submit.router)
    app.include_router(rate.router)
    app.include_router(push.router)

    return app


# Endpoints that can make Open-Meteo calls.
_BUDGETED_PATHS = ("/predict", "/forecast", "/heatmap", "/geocode", "/rate")


class CallBudgetMiddleware:
    """Tags each request with its client (so Open-Meteo calls it causes are
    charged to it) and refuses it up front, with 429, once that client is over
    RATE_LIMIT_CLIENT_HOURLY_CALLS. See app/utils/call_budget.py."""

    def __init__(self, app) -> None:
        self.app = app
        self._logged_source = False

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(_BUDGETED_PATHS):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        peer = scope["client"][0] if scope.get("client") else None
        client = client_key(headers, peer)
        if not self._logged_source:
            # Once per process: shows in Render's logs which header is used.
            self._logged_source = True
            source = ("cf-connecting-ip" if "cf-connecting-ip" in headers
                      else "x-forwarded-for" if "x-forwarded-for" in headers else "socket peer")
            logger.info("Call budget: client IP taken from %s", source)

        budget = getattr(scope.get("app").state, "call_budget", None) if scope.get("app") else None
        if budget is not None and scope["method"] != "OPTIONS":
            retry_after = budget.client_retry_after(client)
            if retry_after is not None:
                logger.warning("Call budget: %s over its hourly limit, refusing %s", client, scope["path"])
                minutes = max(1, round(retry_after / 60))
                response = JSONResponse(
                    {"detail": f"Too many new places in a short time — please try again in about {minutes} min."},
                    status_code=429,
                    headers={"Retry-After": str(retry_after), "Access-Control-Allow-Origin": "*"},
                )
                await response(scope, receive, send)
                return

        token = current_client.set(client)
        try:
            await self.app(scope, receive, send)
        finally:
            current_client.reset(token)


app = create_app()
