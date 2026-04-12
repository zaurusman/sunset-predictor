"""Application configuration loaded from environment variables."""
from __future__ import annotations

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

    # In-memory weather cache TTL (seconds)
    CACHE_TTL_SECONDS: int = 900

    # Displayed in /health and API responses
    ALGORITHM_VERSION: str = "1.0.0"

    APP_ENV: str = "development"

    # HTTP client timeout (seconds)
    HTTP_TIMEOUT: float = 15.0


# Module-level singleton — import this everywhere
settings = Settings()
