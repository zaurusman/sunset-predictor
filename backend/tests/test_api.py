"""Integration tests for API endpoints using TestClient."""
from __future__ import annotations

from datetime import date, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


_FULL_OVERRIDE = {
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
}


def test_predict_confidence_drops_for_distant_future(client):
    """Identical conditions 14 days out should be less confident than today —
    forecast skill decays with lead time."""
    base = {"latitude": 32.08, "longitude": 34.78, "weather_override": _FULL_OVERRIDE}
    r_today = client.post("/predict", json={**base, "target_date": date.today().isoformat()})
    r_far = client.post(
        "/predict",
        json={**base, "target_date": (date.today() + timedelta(days=14)).isoformat()},
    )
    assert r_today.status_code == 200 and r_far.status_code == 200, (r_today.text, r_far.text)
    c_today = r_today.json()["confidence_0_100"]
    c_far = r_far.json()["confidence_0_100"]
    assert c_far < c_today, f"14-days-out confidence {c_far} should be < today {c_today}"


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


def test_predict_returns_503_when_weather_rate_limited(client, monkeypatch):
    """A persistent Open-Meteo 429 should surface as a clean 503, not a 500."""
    weather = app.state.prediction_service._weather

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    monkeypatch.setattr(weather, "_http", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(weather._settings, "HTTP_BACKOFF_BASE", 0.0)
    monkeypatch.setattr(weather._settings, "HTTP_MAX_RETRY_DELAY", 0.0)
    weather._cache.clear()

    resp = client.post("/predict", json={"latitude": 12.34, "longitude": 56.78})

    assert resp.status_code == 503, f"Expected 503, got {resp.status_code}: {resp.text}"
    assert "Retry-After" in resp.headers
    body = resp.text.lower()
    assert "rate" in body or "unavailable" in body or "try again" in body


def test_model_info_endpoint(client):
    """GET /model/info should return 200."""
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "blend_alpha" in data
    assert "algorithm_version" in data
