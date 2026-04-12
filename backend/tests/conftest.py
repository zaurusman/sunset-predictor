"""Shared test fixtures."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.weather import WeatherSnapshot
from app.services.scoring_engine import ScoringEngine


@pytest.fixture(scope="session")
def client() -> TestClient:
    """TestClient that drives the full FastAPI app."""
    return TestClient(app)


@pytest.fixture
def scoring_engine() -> ScoringEngine:
    return ScoringEngine()


@pytest.fixture
def clear_sky_weather() -> WeatherSnapshot:
    """Clear sky — good visibility but no clouds."""
    return WeatherSnapshot(
        cloud_low=0.0,
        cloud_mid=5.0,
        cloud_high=5.0,
        cloud_total=8.0,
        visibility_m=25_000.0,
        relative_humidity=40.0,
        dewpoint_c=5.0,
        temperature_c=20.0,
        precipitation_mm=0.0,
        wind_speed_kmh=10.0,
        pressure_hpa=1013.0,
        aerosol_optical_depth=0.1,
        sun_elevation_deg=2.0,
        data_source="override",
        aerosol_is_estimated=False,
    )


@pytest.fixture
def ideal_weather() -> WeatherSnapshot:
    """Near-ideal sunset conditions: moderate high clouds, clear horizon."""
    return WeatherSnapshot(
        cloud_low=5.0,
        cloud_mid=20.0,
        cloud_high=50.0,
        cloud_total=55.0,
        visibility_m=22_000.0,
        relative_humidity=50.0,
        dewpoint_c=8.0,
        temperature_c=18.0,
        precipitation_mm=0.0,
        wind_speed_kmh=8.0,
        pressure_hpa=1015.0,
        aerosol_optical_depth=0.18,
        sun_elevation_deg=2.5,
        data_source="override",
        aerosol_is_estimated=False,
    )


@pytest.fixture
def overcast_weather() -> WeatherSnapshot:
    """Full overcast with heavy low cloud."""
    return WeatherSnapshot(
        cloud_low=80.0,
        cloud_mid=50.0,
        cloud_high=20.0,
        cloud_total=95.0,
        visibility_m=8_000.0,
        relative_humidity=85.0,
        dewpoint_c=14.0,
        temperature_c=15.0,
        precipitation_mm=0.5,
        wind_speed_kmh=20.0,
        pressure_hpa=1005.0,
        aerosol_optical_depth=0.4,
        sun_elevation_deg=1.0,
        data_source="override",
        aerosol_is_estimated=False,
    )


@pytest.fixture
def rainy_weather() -> WeatherSnapshot:
    """Raining near sunset."""
    return WeatherSnapshot(
        cloud_low=70.0,
        cloud_mid=60.0,
        cloud_high=30.0,
        cloud_total=90.0,
        visibility_m=5_000.0,
        relative_humidity=92.0,
        dewpoint_c=16.0,
        temperature_c=17.0,
        precipitation_mm=5.0,
        wind_speed_kmh=25.0,
        pressure_hpa=1002.0,
        aerosol_optical_depth=0.5,
        sun_elevation_deg=0.5,
        data_source="override",
        aerosol_is_estimated=False,
    )
