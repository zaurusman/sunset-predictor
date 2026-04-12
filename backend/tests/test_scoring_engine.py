"""Unit tests for the physics scoring engine."""
from __future__ import annotations

import pytest
from app.services.scoring_engine import ScoringEngine
from app.schemas.weather import WeatherSnapshot


def test_clear_sky_no_high_clouds_not_top_tier(scoring_engine, clear_sky_weather):
    """
    Clear sky with minimal clouds should not score top-tier.
    Nothing is in the sky to catch and scatter the light.
    """
    result = scoring_engine.score(clear_sky_weather, horizon_obstruction_deg=2.0)
    # Cloud quality should be low (no high clouds)
    assert result.cloud_quality < 50, f"Expected cloud_quality < 50, got {result.cloud_quality}"
    # Overall score should not be Epic or Great
    category = scoring_engine.score_to_category(result.physics_score)
    assert category not in ("Epic", "Great"), f"Expected not Great/Epic for clear sky, got {category}"


def test_moderate_high_clouds_scores_well(scoring_engine, ideal_weather):
    """
    Moderate high clouds with clear horizon should score well.
    """
    result = scoring_engine.score(ideal_weather, horizon_obstruction_deg=2.0)
    assert result.physics_score >= 60, f"Expected score >= 60, got {result.physics_score}"
    category = scoring_engine.score_to_category(result.physics_score)
    assert category in ("Great", "Epic", "Good"), f"Expected Good/Great/Epic, got {category}"


def test_full_overcast_scores_poorly(scoring_engine, overcast_weather):
    """Full overcast with heavy low cloud should score poorly."""
    result = scoring_engine.score(overcast_weather, horizon_obstruction_deg=2.0)
    assert result.physics_score < 30, f"Expected score < 30, got {result.physics_score}"
    category = scoring_engine.score_to_category(result.physics_score)
    assert category in ("Poor", "Decent"), f"Expected Poor/Decent, got {category}"


def test_heavy_low_cloud_reduces_score(scoring_engine):
    """High low-cloud coverage should reduce cloud quality score."""
    high_low_cloud = WeatherSnapshot(
        cloud_low=60.0, cloud_mid=20.0, cloud_high=55.0, cloud_total=70.0,
        visibility_m=18_000.0, relative_humidity=60.0, dewpoint_c=10.0,
        temperature_c=18.0, precipitation_mm=0.0, wind_speed_kmh=10.0,
        pressure_hpa=1013.0, aerosol_optical_depth=0.15,
        sun_elevation_deg=2.0, data_source="override", aerosol_is_estimated=False,
    )
    no_low_cloud = WeatherSnapshot(
        cloud_low=5.0, cloud_mid=20.0, cloud_high=55.0, cloud_total=60.0,
        visibility_m=18_000.0, relative_humidity=60.0, dewpoint_c=10.0,
        temperature_c=18.0, precipitation_mm=0.0, wind_speed_kmh=10.0,
        pressure_hpa=1013.0, aerosol_optical_depth=0.15,
        sun_elevation_deg=2.0, data_source="override", aerosol_is_estimated=False,
    )
    engine = ScoringEngine()
    result_high = engine.score(high_low_cloud, 2.0)
    result_low = engine.score(no_low_cloud, 2.0)
    assert result_high.cloud_quality < result_low.cloud_quality, (
        "High low-cloud should produce lower cloud quality than low low-cloud"
    )
    assert result_high.physics_score < result_low.physics_score


def test_precipitation_penalizes_moisture_score(scoring_engine, rainy_weather):
    """Rain should produce a very low moisture score."""
    result = scoring_engine.score(rainy_weather, horizon_obstruction_deg=2.0)
    assert result.moisture < 20, f"Expected moisture score < 20 with rain, got {result.moisture}"


def test_horizon_obstruction_penalty(scoring_engine, ideal_weather):
    """Large horizon obstruction should significantly reduce horizon score."""
    result_open = scoring_engine.score(ideal_weather, horizon_obstruction_deg=0.0)
    result_blocked = scoring_engine.score(ideal_weather, horizon_obstruction_deg=15.0)
    assert result_blocked.horizon < 40, f"Expected horizon score < 40 at 15 deg, got {result_blocked.horizon}"
    assert result_open.horizon > 90, f"Expected horizon score > 90 at 0 deg, got {result_open.horizon}"
    assert result_open.physics_score > result_blocked.physics_score


def test_score_to_category_boundaries():
    """Category thresholds should map correctly."""
    engine = ScoringEngine()
    assert engine.score_to_category(85) == "Epic"
    assert engine.score_to_category(70) == "Great"
    assert engine.score_to_category(55) == "Good"
    assert engine.score_to_category(40) == "Decent"
    assert engine.score_to_category(15) == "Poor"
    assert engine.score_to_category(0) == "Poor"
    assert engine.score_to_category(100) == "Epic"


def test_score_output_is_in_range(scoring_engine, ideal_weather, overcast_weather, clear_sky_weather):
    """All component and total scores must stay in [0, 100]."""
    for weather in [ideal_weather, overcast_weather, clear_sky_weather]:
        result = scoring_engine.score(weather, horizon_obstruction_deg=5.0)
        for name, val in [
            ("cloud_quality", result.cloud_quality),
            ("atmosphere", result.atmosphere),
            ("moisture", result.moisture),
            ("horizon", result.horizon),
            ("physics_score", result.physics_score),
        ]:
            assert 0 <= val <= 100, f"{name} = {val} out of [0, 100]"


def test_confidence_in_range(scoring_engine, ideal_weather):
    """Confidence should always be within [15, 92]."""
    result = scoring_engine.score(ideal_weather, horizon_obstruction_deg=2.0)
    assert 15 <= result.confidence <= 92, f"Confidence {result.confidence} out of range"


def test_cloud_quality_bell_curve_peak():
    """Cloud quality should peak around 45% high coverage."""
    engine = ScoringEngine()
    score_low_high = engine.cloud_quality_score(5, 10, 10, 20)
    score_peak_high = engine.cloud_quality_score(5, 10, 45, 55)
    score_max_high = engine.cloud_quality_score(5, 10, 100, 100)
    assert score_peak_high > score_low_high, "Peak high cloud coverage should score better than low"
    assert score_peak_high > score_max_high, "Full high cloud coverage should be worse than peak"
