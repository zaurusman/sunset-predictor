"""Integration tests for API endpoints using TestClient."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """GET /health should return 200 with status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "algorithm_version" in data
    assert "ml_model_loaded" in data


def test_predict_with_weather_override(client):
    """
    POST /predict with a full weather_override should return a valid response
    without making any external API calls.
    """
    payload = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "horizon_obstruction_deg": 2.0,
        "weather_override": {
            "cloud_low": 5.0,
            "cloud_mid": 20.0,
            "cloud_high": 50.0,
            "cloud_total": 60.0,
            "visibility_m": 22000.0,
            "relative_humidity": 50.0,
            "dewpoint_c": 8.0,
            "temperature_c": 18.0,
            "precipitation_mm": 0.0,
            "wind_speed_kmh": 8.0,
            "pressure_hpa": 1015.0,
            "aerosol_optical_depth": 0.18,
        },
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"

    data = resp.json()
    assert 0 <= data["beauty_score_0_100"] <= 100
    assert data["category"] in ("Poor", "Decent", "Good", "Great", "Epic")
    assert 0 <= data["confidence_0_100"] <= 100
    assert isinstance(data["reasons"], list)
    assert len(data["reasons"]) >= 3
    assert "sunset_time" in data
    assert "best_viewing_window_start" in data
    assert "best_viewing_window_end" in data
    assert "physics_component_breakdown" in data
    assert "weather_summary" in data


def test_predict_clear_sky_override(client):
    """Clear sky (no high clouds) should not produce Epic or Great."""
    payload = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "weather_override": {
            "cloud_low": 0.0,
            "cloud_mid": 2.0,
            "cloud_high": 3.0,
            "cloud_total": 5.0,
            "visibility_m": 30000.0,
            "relative_humidity": 30.0,
            "dewpoint_c": 2.0,
            "temperature_c": 22.0,
            "precipitation_mm": 0.0,
            "wind_speed_kmh": 5.0,
            "pressure_hpa": 1018.0,
            "aerosol_optical_depth": 0.05,
        },
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] not in ("Epic", "Great"), (
        f"Clear sky should not be Epic/Great, got {data['category']} "
        f"(score={data['beauty_score_0_100']})"
    )


def test_predict_rainy_override_scores_poorly(client):
    """Heavy rain near sunset should produce Poor or Decent."""
    payload = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "weather_override": {
            "cloud_low": 80.0,
            "cloud_mid": 70.0,
            "cloud_high": 20.0,
            "cloud_total": 95.0,
            "visibility_m": 4000.0,
            "relative_humidity": 95.0,
            "dewpoint_c": 16.0,
            "temperature_c": 17.0,
            "precipitation_mm": 8.0,
            "wind_speed_kmh": 30.0,
            "pressure_hpa": 1000.0,
            "aerosol_optical_depth": 0.6,
        },
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["beauty_score_0_100"] < 35, (
        f"Rainy overcast should score < 35, got {data['beauty_score_0_100']}"
    )


def test_predict_invalid_latitude(client):
    """Latitude out of [-90, 90] should return 422."""
    resp = client.post("/predict", json={"latitude": 200.0, "longitude": 0.0})
    assert resp.status_code == 422


def test_predict_invalid_longitude(client):
    """Longitude out of [-180, 180] should return 422."""
    resp = client.post("/predict", json={"latitude": 0.0, "longitude": 999.0})
    assert resp.status_code == 422


def test_model_info_endpoint(client):
    """GET /model/info should return 200."""
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "blend_alpha" in data
    assert "algorithm_version" in data
