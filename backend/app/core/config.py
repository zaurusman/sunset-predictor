"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
import tempfile

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Open-Meteo API base URLs (free, no key required)
    OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1"
    OPEN_METEO_AIR_QUALITY_URL: str = "https://air-quality-api.open-meteo.com/v1"
    OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1"
    OPEN_METEO_GEOCODING_URL: str = "https://geocoding-api.open-meteo.com/v1"

    # Reddit API credentials (optional; needed only for dataset building)
    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    REDDIT_USER_AGENT: str = "sunset-predictor/1.0"

    # ML model artifact paths (relative to the backend/ working directory)
    MODEL_PATH: str = "trained_models/calibration_model.joblib"
    MODEL_METADATA_PATH: str = "trained_models/model_metadata.json"

    # Blending weight: final = alpha * physics + (1 - alpha) * ml_prediction
    # 1.0 = pure physics, 0.0 = pure ML
    ML_BLEND_ALPHA: float = 0.4

    # Default horizon obstruction in degrees (0 = open ocean/flat horizon)
    DEFAULT_HORIZON_OBSTRUCTION_DEG: float = 2.0

    # Weather cache TTL (seconds). Open-Meteo's global models refresh every ~6 h
    # (regional/rapid models as often as hourly), so a 2-hour TTL cuts redundant
    # calls while staying within one model-run of fresh data.
    CACHE_TTL_SECONDS: int = 7200

    # Persist the weather cache to disk so it survives process restarts
    # (notably `uvicorn --reload`, which otherwise wipes the in-memory cache on
    # every code change and forces a full re-fetch). Defaults to a temp-dir file;
    # set to empty string to disable persistence.
    CACHE_PERSIST_PATH: str = os.path.join(
        tempfile.gettempdir(), "afterglow_weather_cache.pkl"
    )

    # Decimal places used to round coordinates in cache keys, so nearby lookups
    # (different users, jittery geolocation) share one Open-Meteo fetch:
    #   2 → ~1 km (lossless on Open-Meteo's grid)
    #   1 → ~11 km (collapses more users onto one call; may merge distinct
    #       high-resolution grid cells in complex coastal/mountain terrain)
    CACHE_COORD_DECIMALS: int = 1

    # Displayed in /health and API responses
    ALGORITHM_VERSION: str = "1.0.0"

    APP_ENV: str = "development"

    # HTTP client timeout (seconds)
    HTTP_TIMEOUT: float = 15.0

    # Resilience to Open-Meteo rate-limits (HTTP 429) and transient 5xx errors.
    # Retries use exponential backoff; the Retry-After header (when present)
    # overrides the computed delay. Delays are capped at HTTP_MAX_RETRY_DELAY.
    HTTP_MAX_RETRIES: int = 3
    HTTP_BACKOFF_BASE: float = 0.5      # seconds; delay = BASE * 2**attempt (+jitter)
    HTTP_MAX_RETRY_DELAY: float = 8.0   # seconds; ceiling for any single backoff wait

    # ── Email / photo submission ──────────────────────────────────────────────
    # Resend API key for sending photo submissions to the developer.
    # Leave RESEND_API_KEY empty to disable the /submit-photo endpoint entirely.
    RESEND_API_KEY: str = ""       # from resend.com dashboard
    RESEND_FROM_EMAIL: str = "Afterglow <onboarding@resend.dev>"  # verified sender
    DEVELOPER_EMAIL: str = ""      # where submissions are sent


# Module-level singleton — import this everywhere
settings = Settings()
