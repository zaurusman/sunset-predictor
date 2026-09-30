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
    OPEN_METEO_ENSEMBLE_URL: str = "https://ensemble-api.open-meteo.com/v1"

    # Forecast model, set explicitly rather than left at Open-Meteo's `auto`.
    # ICON-based cloud forecasts are generally ranked above GFS/NAM-based ones,
    # and an explicit model is a prerequisite for the ensemble spread below —
    # confidence has to measure spread of the SAME model family it forecasts with.
    OPEN_METEO_MODEL: str = "icon_seamless"

    # Reddit API credentials (optional; needed only for dataset building)
    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    REDDIT_USER_AGENT: str = "sunset-predictor/1.0"

    # ML model artifact paths (relative to the backend/ working directory)
    MODEL_PATH: str = "trained_models/calibration_model.joblib"
    MODEL_METADATA_PATH: str = "trained_models/model_metadata.json"

    # Blending weight: final = alpha * physics + (1 - alpha) * ml_prediction
    # 1.0 = pure physics, 0.0 = pure ML.
    #
    # Deliberately 1.0 — the ML branch is OFF. The one model ever trained scored
    # spearman_r = -0.0266 (p = 0.55), i.e. no signal, because its labels were
    # Reddit upvotes joined to weather at five fixed cities regardless of where
    # the photo was taken. It was shelved for that reason. At the previous 0.4 a
    # model would have taken 60 % of the final score, so an accidental .joblib
    # landing in trained_models/ was one file away from silently degrading every
    # prediction. See docs/scoring-v2-plan.md (D7) and ML_MIN_SPEARMAN below.
    ML_BLEND_ALPHA: float = 1.0

    # A model must beat this rank correlation, recorded in its own metadata,
    # before MLModel.load() will accept it. Guards against re-loading a model
    # that measured no better than noise.
    ML_MIN_SPEARMAN: float = 0.15

    # Where human sunset ratings (ML training labels) are appended as JSONL.
    # NOTE: Render's free tier filesystem is EPHEMERAL — point this at a mounted
    # persistent disk before relying on it in production.
    RATINGS_PATH: str = "data/ratings.jsonl"

    # Default horizon obstruction in degrees (0 = open ocean/flat horizon)
    DEFAULT_HORIZON_OBSTRUCTION_DEG: float = 2.0

    # Weather cache TTL (seconds). Open-Meteo's global models refresh every ~6 h
    # (regional/rapid models as often as hourly), so a 2-hour TTL cuts redundant
    # calls while staying within one model-run of fresh data.
    CACHE_TTL_SECONDS: int = 7200

    # How long an EXPIRED cache entry is kept as a fallback. When Open-Meteo
    # is rate-limiting or down, a weather lookup serves the last good data
    # (up to CACHE_TTL_SECONDS + this old) instead of failing with a 503.
    # 0 disables the fallback.
    CACHE_STALE_GRACE_SECONDS: int = 43200

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

    # Most Open-Meteo requests in flight at once. More than a handful gets
    # 429 "Too many concurrent requests" on the free API.
    OPEN_METEO_MAX_CONCURRENCY: int = 3

    # ── Email / photo submission ──────────────────────────────────────────────
    # Resend API key for sending photo submissions to the developer.
    # Leave RESEND_API_KEY empty to disable the /submit-photo endpoint entirely.
    RESEND_API_KEY: str = ""       # from resend.com dashboard
    RESEND_FROM_EMAIL: str = "Afterglow <onboarding@resend.dev>"  # verified sender
    DEVELOPER_EMAIL: str = ""      # where submissions are sent

    # ── Push alerts (Epic sunset notifications) ──────────────────────────────
    # All empty → push is disabled and the app behaves exactly as without it.
    # Neon/Postgres DSN. Render's disk is ephemeral, so subscriptions can't live
    # in a local file.
    DATABASE_URL: str = ""
    # Generate with: python scripts/generate_vapid_keys.py
    VAPID_PUBLIC_KEY: str = ""
    VAPID_PRIVATE_KEY: str = ""
    # Contact claim push services require (mailto: or https:).
    VAPID_SUBJECT: str = "https://sunset-predictor-henna.vercel.app"
    # Shared secret the hourly GitHub Actions cron sends in X-Alerts-Secret.
    ALERTS_SECRET: str = ""
    # A cell is checked once, when its sunset is this many hours away.
    ALERT_LEAD_MIN_HOURS: float = 3.5
    ALERT_LEAD_MAX_HOURS: float = 4.5

# Module-level singleton — import this everywhere
settings = Settings()
