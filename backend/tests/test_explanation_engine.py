"""Unit tests for the explanation engine."""
from __future__ import annotations

import pytest
from app.services.explanation_engine import ExplanationEngine
from app.schemas.prediction import PhysicsBreakdown
from app.schemas.weather import WeatherSnapshot


def _make_breakdown(**kwargs) -> PhysicsBreakdown:
    defaults = dict(
        cloud_quality_score=50.0,
        atmosphere_score=50.0,
        moisture_score=80.0,
        horizon_score=90.0,
        weighted_physics_score=60.0,
        component_weights={"cloud_quality": 0.4, "atmosphere": 0.3, "moisture": 0.2, "horizon": 0.1},
    )
    defaults.update(kwargs)
    return PhysicsBreakdown(**defaults)


def _make_weather(**kwargs) -> WeatherSnapshot:
    defaults = dict(
        cloud_low=5.0, cloud_mid=20.0, cloud_high=45.0, cloud_total=55.0,
        visibility_m=20000.0, relative_humidity=55.0, dewpoint_c=8.0,
        temperature_c=18.0, precipitation_mm=0.0, wind_speed_kmh=10.0,
        pressure_hpa=1013.0, aerosol_optical_depth=0.15,
        sun_elevation_deg=2.0, data_source="override", aerosol_is_estimated=False,
    )
    defaults.update(kwargs)
    return WeatherSnapshot(**defaults)


def test_returns_at_least_3_reasons():
    engine = ExplanationEngine()
    weather = _make_weather()
    breakdown = _make_breakdown()
    reasons = engine.generate(weather, breakdown, "Good")
    assert len(reasons) >= 3, f"Expected >= 3 reasons, got {len(reasons)}"


def test_returns_at_most_6_reasons():
    engine = ExplanationEngine()
    weather = _make_weather()
    breakdown = _make_breakdown()
    reasons = engine.generate(weather, breakdown, "Good")
    assert len(reasons) <= 6, f"Expected <= 6 reasons, got {len(reasons)}"


def test_rain_reason_present_when_raining():
    engine = ExplanationEngine()
    weather = _make_weather(precipitation_mm=5.0, cloud_low=70.0, cloud_total=90.0)
    breakdown = _make_breakdown(moisture_score=10.0, cloud_quality_score=20.0)
    reasons = engine.generate(weather, breakdown, "Poor")
    rain_reasons = [r for r in reasons if "rain" in r.lower() or "precipitation" in r.lower()]
    assert rain_reasons, f"Expected rain-related reason, got: {reasons}"


def test_high_cloud_positive_reason():
    engine = ExplanationEngine()
    weather = _make_weather(cloud_high=55.0, cloud_low=5.0, cloud_total=60.0)
    breakdown = _make_breakdown(cloud_quality_score=72.0)
    reasons = engine.generate(weather, breakdown, "Great")
    cloud_reasons = [r for r in reasons if "high cloud" in r.lower() or "cloud" in r.lower()]
    assert cloud_reasons, f"Expected cloud-positive reason, got: {reasons}"


def test_low_cloud_negative_reason():
    engine = ExplanationEngine()
    weather = _make_weather(cloud_low=60.0, cloud_high=20.0, cloud_total=75.0)
    breakdown = _make_breakdown(cloud_quality_score=25.0)
    reasons = engine.generate(weather, breakdown, "Decent")
    negative_reasons = [r for r in reasons if "low cloud" in r.lower() or "block" in r.lower()]
    assert negative_reasons, f"Expected low-cloud negative reason, got: {reasons}"


def test_aerosol_estimated_note():
    engine = ExplanationEngine()
    weather = _make_weather(aerosol_is_estimated=True)
    breakdown = _make_breakdown(atmosphere_score=55.0)
    reasons = engine.generate(weather, breakdown, "Good")
    aod_notes = [r for r in reasons if "aerosol" in r.lower() or "estimated" in r.lower()]
    assert aod_notes, f"Expected aerosol estimation note, got: {reasons}"


def test_all_reasons_are_strings():
    engine = ExplanationEngine()
    weather = _make_weather()
    breakdown = _make_breakdown()
    reasons = engine.generate(weather, breakdown, "Good")
    for r in reasons:
        assert isinstance(r, str), f"Expected string reason, got {type(r)}: {r!r}"
        assert len(r) > 10, f"Reason too short: {r!r}"
