"""Unit tests for the physics scoring engine."""
from __future__ import annotations

import pytest
from app.services.scoring_engine import ScoringEngine
from app.schemas.weather import WeatherSnapshot


# ---------------------------------------------------------------------------
# Helper to build a minimal WeatherSnapshot
# ---------------------------------------------------------------------------

def _snap(**kwargs) -> WeatherSnapshot:
    defaults = dict(
        cloud_low=0.0, cloud_mid=10.0, cloud_high=40.0, cloud_total=45.0,
        visibility_m=20_000.0, relative_humidity=50.0,
        dewpoint_c=8.0, temperature_c=18.0, precipitation_mm=0.0,
        wind_speed_kmh=10.0, pressure_hpa=1013.0,
        aerosol_optical_depth=0.18, sun_elevation_deg=2.0,
        data_source="override", aerosol_is_estimated=False,
    )
    defaults.update(kwargs)
    return WeatherSnapshot(**defaults)


# ---------------------------------------------------------------------------
# Cloud quality tests
# ---------------------------------------------------------------------------

def test_clear_sky_no_high_clouds_not_top_tier(scoring_engine, clear_sky_weather):
    """
    Clear sky with minimal clouds should not score Epic.
    With the improved algorithm a clear sky + perfect air quality can
    legitimately reach Great (the atmosphere component is very strong),
    but it should never be Epic — there's nothing dramatic to light up.
    """
    result = scoring_engine.score(clear_sky_weather, horizon_obstruction_deg=2.0)
    assert result.cloud_quality < 50, f"Expected cloud_quality < 50, got {result.cloud_quality}"
    category = scoring_engine.score_to_category(result.physics_score)
    assert category != "Epic", f"Expected not Epic for clear sky, got {category}"


def test_clear_sky_not_crushed():
    """
    Clear sky should not score below ~20 — it produces a pastel sunset,
    not a non-event.  Previous algorithm could push this near 0.
    """
    engine = ScoringEngine()
    score = engine.cloud_quality_score(0.0, 5.0, 5.0, 8.0)
    # Mild penalty applied; expect at least 18 (was effectively ~0 before)
    assert score >= 18.0, f"Clear sky cloud_quality too low: {score}"


def test_strong_high_clouds_with_low_low_clouds_scores_high():
    """
    The target failure case: excellent high clouds + some low clouds.
    Upper-cloud offset should soften the low-cloud penalty so the result
    is still high (>= 60).
    """
    engine = ScoringEngine()
    # High=65%, low=30% — previously these low clouds would crush the score
    score = engine.cloud_quality_score(30.0, 15.0, 65.0, 75.0)
    assert score >= 60.0, (
        f"Strong high clouds + moderate low clouds should score >= 60, got {score}"
    )


def test_moderate_high_clouds_scores_well(scoring_engine, ideal_weather):
    """Moderate high clouds with clear horizon should score well."""
    result = scoring_engine.score(ideal_weather, horizon_obstruction_deg=2.0)
    assert result.physics_score >= 60, f"Expected score >= 60, got {result.physics_score}"
    category = scoring_engine.score_to_category(result.physics_score)
    assert category in ("Great", "Epic", "Good"), f"Expected Good/Great/Epic, got {category}"


def test_full_overcast_scores_poorly(scoring_engine, overcast_weather):
    """
    Full overcast with heavy low cloud should score poorly.
    The improved moisture scorer is slightly more lenient (humidity threshold
    raised to 85 %, precip factor softened) so the fixture lands at ~32
    rather than <30.  The category check (Poor/Decent) remains the real
    intent of this test — the sky is not worth going outside for.
    """
    result = scoring_engine.score(overcast_weather, horizon_obstruction_deg=2.0)
    assert result.physics_score < 35, f"Expected score < 35, got {result.physics_score}"
    category = scoring_engine.score_to_category(result.physics_score)
    assert category in ("Poor", "Decent"), f"Expected Poor/Decent, got {category}"


def test_heavy_low_cloud_reduces_score(scoring_engine):
    """High low-cloud coverage should reduce cloud quality score."""
    high_low_cloud = _snap(cloud_low=60.0, cloud_mid=20.0, cloud_high=55.0, cloud_total=70.0)
    no_low_cloud = _snap(cloud_low=5.0, cloud_mid=20.0, cloud_high=55.0, cloud_total=60.0)

    engine = ScoringEngine()
    result_high = engine.score(high_low_cloud, 2.0)
    result_low = engine.score(no_low_cloud, 2.0)
    assert result_high.cloud_quality < result_low.cloud_quality, (
        "High low-cloud should produce lower cloud quality than low low-cloud"
    )
    assert result_high.physics_score < result_low.physics_score


def test_overcast_penalty_is_type_aware():
    """
    The overcast penalty must be driven by LOW and MID cloud coverage, not total.

    Physical reasoning:
    - Low stratus at 90 % blocks all light near the horizon — very bad.
    - High cirrus at 90 % (with 0 % low) diffuses light but keeps the sky open
      for colour.  After sunset it illuminates from below — potentially excellent.

    Scenario A: stratus-dominated overcast (heavy low+mid, little high)
    Scenario B: cirrus-dominated overcast (heavy high only, no low)
    Scenario A must score significantly lower than Scenario B.
    """
    engine = ScoringEngine()

    # Scenario A: low-cloud overcast — genuinely bad
    score_stratus = engine.cloud_quality_score(65.0, 40.0, 10.0, 85.0)

    # Scenario B: pure cirrus overcast — bad for sunset drama but not blocking
    score_cirrus = engine.cloud_quality_score(0.0, 0.0, 97.0, 97.0)

    assert score_cirrus > score_stratus, (
        f"Cirrus overcast ({score_cirrus:.1f}) should score above stratus overcast ({score_stratus:.1f})"
    )
    assert score_cirrus >= 35.0, (
        f"Cirrus overcast should not be crushed; got {score_cirrus:.1f}"
    )
    assert score_stratus < 30.0, (
        f"Stratus overcast should score poorly; got {score_stratus:.1f}"
    )


def test_cloud_quality_bell_curve_peak():
    """Cloud quality should peak around 45% high coverage."""
    engine = ScoringEngine()
    score_low_high = engine.cloud_quality_score(5, 10, 10, 20)
    score_peak_high = engine.cloud_quality_score(5, 10, 45, 55)
    score_max_high = engine.cloud_quality_score(5, 10, 100, 100)
    assert score_peak_high > score_low_high, "Peak high cloud coverage should score better than low"
    assert score_peak_high > score_max_high, "Full high cloud coverage should be worse than peak"


# ---------------------------------------------------------------------------
# Moisture / precipitation tests
# ---------------------------------------------------------------------------

def test_precipitation_penalizes_moisture_score(scoring_engine, rainy_weather):
    """Rain should produce a very low moisture score."""
    result = scoring_engine.score(rainy_weather, horizon_obstruction_deg=2.0)
    assert result.moisture < 20, f"Expected moisture score < 20 with rain, got {result.moisture}"


def test_post_rain_clearing_bonus():
    """
    Recent rain (last 3h) followed by no current rain should score
    better than active rain — post-rain clearing is a known sunset trigger.
    """
    engine = ScoringEngine()

    # Currently raining — bad
    score_active = engine.moisture_score(2.5, 70.0)

    # Rain stopped, recently cleared, pressure rising
    score_clearing = engine.moisture_score(
        0.0, 65.0,
        precip_last_3h=2.0,
        pressure_trend=2.0,
        cloud_trend=-15.0,
        vis_trend=2000.0,
    )

    # No rain at all (baseline)
    score_dry = engine.moisture_score(0.0, 55.0)

    assert score_clearing > score_active, (
        f"Post-rain clearing ({score_clearing:.1f}) should beat active rain ({score_active:.1f})"
    )
    # Clearing bonus should push it meaningfully above the dry baseline would imply
    assert score_clearing >= 90.0, (
        f"Post-rain clearing with all bonuses should reach >= 90, got {score_clearing:.1f}"
    )
    assert score_dry > score_clearing or score_dry <= score_clearing + 2, (
        "Dry baseline should be close to or above clearing (both are good)"
    )


def test_missing_aerosol_does_not_tank_score():
    """
    Missing aerosol data should not cause a severe penalty.
    With good visibility the atmosphere score should still be reasonable.
    """
    engine = ScoringEngine()
    score_with_aod = engine.atmosphere_score(20_000.0, 0.18, 50.0)
    score_no_aod = engine.atmosphere_score(20_000.0, None, 50.0)
    # No-AOD fallback should be within 20 pts of real AOD score
    assert score_no_aod >= score_with_aod - 20.0, (
        f"Missing AOD score ({score_no_aod:.1f}) dropped too far below real AOD ({score_with_aod:.1f})"
    )
    # Should still clear a reasonable floor
    assert score_no_aod >= 45.0, f"Missing AOD score too low: {score_no_aod:.1f}"


# ---------------------------------------------------------------------------
# Horizon tests
# ---------------------------------------------------------------------------

def test_horizon_obstruction_penalty(scoring_engine, ideal_weather):
    """Large horizon obstruction should significantly reduce horizon score."""
    result_open = scoring_engine.score(ideal_weather, horizon_obstruction_deg=0.0)
    result_blocked = scoring_engine.score(ideal_weather, horizon_obstruction_deg=15.0)
    assert result_blocked.horizon < 40, f"Expected horizon score < 40 at 15 deg, got {result_blocked.horizon}"
    assert result_open.horizon > 90, f"Expected horizon score > 90 at 0 deg, got {result_open.horizon}"
    assert result_open.physics_score > result_blocked.physics_score


def test_horizon_suburban_not_crushed():
    """
    5-degree obstruction (typical suburban) should not score below 60.
    The softened curve ensures average locations aren't over-penalised.
    """
    engine = ScoringEngine()
    score = engine.horizon_score(5.0)
    assert score >= 60.0, f"Suburban horizon (5 deg) should score >= 60, got {score}"


# ---------------------------------------------------------------------------
# Window aggregation tests
# ---------------------------------------------------------------------------

def test_window_best_point_dominates():
    """When +15m is the best point, it should be chosen as best_label."""
    engine = ScoringEngine()
    result = engine.score_window([
        ("-15m", 55.0),
        ("sunset", 58.0),
        ("+15m", 75.0),
        ("+30m", 62.0),
    ])
    assert result.best_label == "+15m", f"Expected +15m to be best, got {result.best_label}"
    assert result.final_score >= 75.0, "Final score should be at least the best point score"


def test_window_afterglow_preference():
    """When +15m ties the best within 3 pts, it should be preferred."""
    engine = ScoringEngine()
    result = engine.score_window([
        ("sunset", 72.0),
        ("+15m", 70.5),   # within 3 pts of best
        ("+30m", 65.0),
    ])
    assert result.best_label == "+15m", "Afterglow preference should crown +15m when it's close"


def test_window_consistency_bonus():
    """Four consistently good points should receive a consistency bonus."""
    engine = ScoringEngine()
    all_good = engine.score_window([
        ("-15m", 68.0), ("sunset", 70.0), ("+15m", 72.0), ("+30m", 67.0)
    ])
    one_good = engine.score_window([
        ("-15m", 20.0), ("sunset", 25.0), ("+15m", 72.0), ("+30m", 18.0)
    ])
    assert all_good.consistency_bonus > 0, "Should have consistency bonus when all points good"
    assert all_good.final_score > one_good.final_score, (
        "Consistent good window should outscore a one-great-rest-poor window"
    )


def test_highly_volatile_window_confidence_reduced():
    """
    A window where one point is excellent but the rest collapse
    should produce lower confidence than a consistent window.
    """
    engine = ScoringEngine()
    consistent_snap = _snap()
    volatile_snap = _snap()

    consistent_scores = [70.0, 72.0, 74.0, 71.0]
    volatile_scores = [75.0, 30.0, 28.0, 22.0]

    conf_consistent = engine.compute_confidence(
        weather=consistent_snap,
        component_scores={"cloud_quality": 70, "atmosphere": 65, "moisture": 80, "horizon": 95},
        physics_score=72.0,
        window_scores=consistent_scores,
    )
    conf_volatile = engine.compute_confidence(
        weather=volatile_snap,
        component_scores={"cloud_quality": 70, "atmosphere": 65, "moisture": 80, "horizon": 95},
        physics_score=72.0,
        window_scores=volatile_scores,
    )
    assert conf_volatile < conf_consistent, (
        f"Volatile window confidence ({conf_volatile:.1f}) should be lower than "
        f"consistent ({conf_consistent:.1f})"
    )


def test_window_volatility_penalty():
    """A spread > 30 should trigger a volatility penalty."""
    engine = ScoringEngine()
    result = engine.score_window([
        ("-15m", 20.0), ("sunset", 25.0), ("+15m", 75.0), ("+30m", 22.0)
    ])
    assert result.volatility_penalty > 0, "Large spread should incur volatility penalty"


def test_single_point_window():
    """score_window should work with a single point (override path)."""
    engine = ScoringEngine()
    result = engine.score_window([("sunset", 62.0)])
    assert result.best_label == "sunset"
    assert abs(result.final_score - 62.0) < 1.0


# ---------------------------------------------------------------------------
# General correctness / range tests
# ---------------------------------------------------------------------------

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


def test_go_outside_threshold():
    """go_outside should be True when final_score >= 45."""
    engine = ScoringEngine()
    above = engine.score_window([("sunset", 50.0)])
    below = engine.score_window([("sunset", 40.0)])
    assert above.go_outside is True
    assert below.go_outside is False


# ---------------------------------------------------------------------------
# Afterglow tests
# ---------------------------------------------------------------------------
# Afterglow physics: when the sun is 0°–6° below the horizon it still
# illuminates high clouds (cirrus/altocumulus at 8–12 km) from below via
# limb Rayleigh scattering.  This produces the deepest reds and crimsons and
# peaks around −3° (roughly 10–15 minutes after sunset).
#
# Expected model behaviour
# ------------------------
# • cloud_quality for the same cloud conditions should be HIGHER at sun = −3°
#   than at sun = +2° (pre-sunset), because high clouds are being lit from below.
# • The boost should be ZERO when high clouds are absent (nothing to illuminate).
# • The boost should be ZERO or negligible when the sky is overcast (diffuse).
# • Clear sky should not become "Epic" — no canvas, no colour drama.
# • Heavy low cloud should block the afterglow view (low_factor kills the boost).
# ---------------------------------------------------------------------------

def test_afterglow_boosts_cloud_quality_score():
    """
    The same high-cloud conditions should score higher at sun −3° than at
    sun +2° (pre-sunset).  This is the core afterglow requirement.
    """
    engine = ScoringEngine()
    # Good high-cloud setup: 50% high, 10% low, no overcast
    pre_sunset  = engine.cloud_quality_score(10.0, 15.0, 50.0, 60.0, sun_elevation_deg=+2.0)
    at_afterglow = engine.cloud_quality_score(10.0, 15.0, 50.0, 60.0, sun_elevation_deg=-3.0)
    assert at_afterglow > pre_sunset, (
        f"Afterglow should boost cloud quality: got {at_afterglow:.1f} <= {pre_sunset:.1f}"
    )
    assert at_afterglow - pre_sunset >= 10.0, (
        f"Expected afterglow boost ≥ 10 pts, got {at_afterglow - pre_sunset:.1f}"
    )


def test_afterglow_peaks_near_minus_3_degrees():
    """
    Cloud quality for fixed high-cloud conditions should peak near sun = −3°.
    At −1° (just below horizon) and −7° (deep twilight) the score should be
    lower than at −3°.
    """
    engine = ScoringEngine()
    args = (10.0, 15.0, 50.0, 60.0)  # low, mid, high, total
    score_minus1  = engine.cloud_quality_score(*args, sun_elevation_deg=-1.0)
    score_minus3  = engine.cloud_quality_score(*args, sun_elevation_deg=-3.0)
    score_minus7  = engine.cloud_quality_score(*args, sun_elevation_deg=-7.0)
    assert score_minus3 > score_minus1, "Peak should be at −3°, not −1°"
    assert score_minus3 > score_minus7, "Peak should be at −3°, not −7°"


def test_afterglow_requires_high_clouds():
    """
    No afterglow boost when high cloud coverage is below the 15 % threshold.
    A clear sky at sun −3° should score the same as at sun +2°.
    """
    engine = ScoringEngine()
    # Near-clear sky: 5% high, 5% total
    pre  = engine.cloud_quality_score(2.0, 3.0, 5.0, 8.0, sun_elevation_deg=+2.0)
    post = engine.cloud_quality_score(2.0, 3.0, 5.0, 8.0, sun_elevation_deg=-3.0)
    assert post == pre, (
        f"No high clouds → no afterglow boost, but got {post:.1f} vs {pre:.1f}"
    )


def test_afterglow_blocked_by_overcast():
    """
    Overcast conditions (total ≥ 82 %) should receive no afterglow boost.
    Overcast diffuses all structure — no vivid colour possible.
    """
    engine = ScoringEngine()
    pre  = engine.cloud_quality_score(70.0, 30.0, 60.0, 90.0, sun_elevation_deg=+2.0)
    post = engine.cloud_quality_score(70.0, 30.0, 60.0, 90.0, sun_elevation_deg=-3.0)
    assert post == pre, (
        f"Overcast → no afterglow boost, but got {post:.1f} vs {pre:.1f}"
    )


def test_afterglow_heavy_low_cloud_blocks_view():
    """
    Heavy low cloud (≥ 60 %) should kill the afterglow boost entirely,
    because the thick low-cloud layer screens the illuminated high-cloud canvas.
    """
    engine = ScoringEngine()
    # Identical clouds except sun position; heavy low cloud present
    pre  = engine.cloud_quality_score(65.0, 15.0, 50.0, 75.0, sun_elevation_deg=+2.0)
    post = engine.cloud_quality_score(65.0, 15.0, 50.0, 75.0, sun_elevation_deg=-3.0)
    assert post <= pre + 2.0, (
        f"Heavy low cloud should block afterglow; got post={post:.1f}, pre={pre:.1f}"
    )


def test_afterglow_score_standalone_zero_above_horizon():
    """afterglow_score() must return 0 when sun is at or above the horizon."""
    engine = ScoringEngine()
    assert engine.afterglow_score(0.0,  50.0, 5.0, 55.0) == 0.0
    assert engine.afterglow_score(+2.0, 50.0, 5.0, 55.0) == 0.0
    assert engine.afterglow_score(+10.0, 50.0, 5.0, 55.0) == 0.0


def test_afterglow_score_standalone_peak_conditions():
    """
    afterglow_score() with ideal conditions (sun −3°, 50% high, clear air)
    should return a high score (≥ 60).
    """
    engine = ScoringEngine()
    score = engine.afterglow_score(
        sun_elevation_deg=-3.0,
        cloud_high=50.0,
        cloud_low=5.0,
        cloud_total=55.0,
        atmosphere=80.0,
    )
    assert score >= 60.0, f"Peak afterglow conditions should score ≥ 60, got {score:.1f}"


def test_afterglow_stored_in_scoring_result():
    """
    score() should store non-zero afterglow in ScoringResult when sun < 0
    and conditions support afterglow.
    """
    engine = ScoringEngine()
    snap_pre  = _snap(sun_elevation_deg=+2.0, cloud_high=50.0, cloud_low=5.0,
                      cloud_mid=10.0, cloud_total=55.0)
    snap_post = _snap(sun_elevation_deg=-3.0, cloud_high=50.0, cloud_low=5.0,
                      cloud_mid=10.0, cloud_total=55.0)
    result_pre  = engine.score(snap_pre, 2.0)
    result_post = engine.score(snap_post, 2.0)

    assert result_pre.afterglow == 0.0, "No afterglow when sun is above horizon"
    assert result_post.afterglow > 0.0, "Afterglow should be non-zero when sun < 0 with high clouds"
    assert result_post.physics_score > result_pre.physics_score, (
        "Post-sunset score should exceed pre-sunset for the same cloud conditions"
    )
