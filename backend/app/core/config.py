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

    # ── Web Push notifications ────────────────────────────────────────────────
    # VAPID keypair identifying this server to browser push services. Generate
    # with `python scripts/generate_vapid_keys.py`. Leave VAPID_PRIVATE_KEY
    # empty to disable notifications entirely — /notifications/config then
    # reports disabled and the frontend hides the toggle.
    #
    # The PRIVATE key is a credential: keep it in the environment, never in the
    # repo. The public key is served to browsers by design.
    VAPID_PUBLIC_KEY: str = ""
    VAPID_PRIVATE_KEY: str = ""
    # Contact for the push service to reach if this server misbehaves. Must be
    # a mailto: or https: URL — push services reject a bare address.
    VAPID_SUBJECT: str = ""

    # Shared secret required by POST /notifications/dispatch. Empty disables the
    # endpoint: unauthenticated, it would let anyone exhaust the Open-Meteo
    # quota and push to every subscriber.
    NOTIFY_DISPATCH_SECRET: str = ""

    # Where push subscriptions are stored as JSON.
    # NOTE: Render's free tier filesystem is EPHEMERAL — every redeploy and
    # idle-sleep restart wipes this file and silently unsubscribes everyone.
    # Point it at a mounted persistent disk before relying on it.
    SUBSCRIPTIONS_PATH: str = "data/subscriptions.json"

    # Defaults offered to a new subscriber. The threshold matches the UI's
    # "worth heading out" bar (GO_OUTSIDE_THRESHOLD) so the alert and the app
    # never disagree about what is worth leaving the house for.
    NOTIFY_DEFAULT_THRESHOLD: float = 70.0
    NOTIFY_DEFAULT_LEAD_MINUTES: int = 120


# Module-level singleton — import this everywhere
settings = Settings()
