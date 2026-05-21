"""
Scoring calibration tests.

These are sanity checks for the three targeted changes made to fix
systematic score inflation, especially for archive dates (>7 days old):

  1. Archive visibility default: 24 km → 15 km
  2. Estimated AOD fallback: max(40, vis*0.80) → vis*0.75  (no floor)
  3. Window consistency bonus: +5 pts max → +3 pts max

Each test documents what the score SHOULD be for a realistic scenario and
why. Run these after any change to scoring_engine.py or the visibility
default in weather_service.py to catch regressions.
"""
from __future__ import annotations

from app.services.scoring_engine import ScoringEngine
from app.schemas.weather import WeatherSnapshot


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _snap(**kwargs) -> WeatherSnapshot:
    """Build a WeatherSnapshot with sensible defaults for testing."""
    defaults = dict(
        cloud_low=0.0, cloud_mid=10.0, cloud_high=40.0, cloud_total=45.0,
        visibility_m=20_000.0, relative_humidity=60.0,
        dewpoint_c=8.0, temperature_c=18.0, precipitation_mm=0.0,
        wind_speed_kmh=10.0, pressure_hpa=1013.0,
        aerosol_optical_depth=None,   # estimated by default
        aerosol_is_estimated=True,
        sun_elevation_deg=1.0,
        data_source="archive",
    )
    defaults.update(kwargs)
    return WeatherSnapshot(**defaults)


engine = ScoringEngine()


# ---------------------------------------------------------------------------
# Fix 1 + 2: Archive atmosphere should no longer be ~93
#
# Previously: 24 km vis → vis_score=96, AOD proxy=0.25 → aer_score≈90
#             atmosphere = 96*0.5 + 90*0.5 = 93
# Now:        15 km vis → vis_score=60, aer_score=60*0.75=45
#             atmosphere = 60*0.5 + 45*0.5 = 52.5
# ---------------------------------------------------------------------------

def test_estimated_atmosphere_15km_is_moderate():
    """
    With 15 km visibility and estimated AOD the atmosphere score should be
    moderate (~50–65), not near-perfect (~90+).
    Previously the 24 km default + max(40,…) floor gave ~93 automatically.
    """
    score = engine.atmosphere_score(15_000.0, None, 60.0)
    assert 45.0 <= score <= 68.0, (
        f"15 km / estimated AOD atmosphere should be moderate, got {score:.1f}"
    )


def test_estimated_atmosphere_scales_with_visibility():
    """
    When AOD is unknown (estimated), the atmosphere score must increase with
    visibility rather than sitting at an artificial floor.
    """
    score_10km = engine.atmosphere_score(10_000.0, None, 60.0)
    score_15km = engine.atmosphere_score(15_000.0, None, 60.0)
    score_25km = engine.atmosphere_score(25_000.0, None, 60.0)

    assert score_10km < score_15km < score_25km, (
        f"Estimated atmosphere must scale with visibility: "
        f"10km={score_10km:.1f}, 15km={score_15km:.1f}, 25km={score_25km:.1f}"
    )


def test_estimated_atmosphere_no_artificial_floor():
    """
    Poor visibility (5 km) with estimated AOD should score below 35.
    The old max(40,…) floor prevented this — now it should genuinely be low.
    """
    score = engine.atmosphere_score(5_000.0, None, 70.0)
    assert score < 35.0, (
        f"Poor visibility / estimated AOD should score < 35, got {score:.1f}"
    )


def test_real_aod_good_visibility_still_scores_high():
    """
    A day with real clear-air data (40 km vis, AOD 0.12) must still score
    near-perfect on atmosphere. Good recent days must not be penalised.
    """
    score = engine.atmosphere_score(40_000.0, 0.12, 50.0)
    assert score >= 88.0, (
        f"Real clear-air data should score >= 88, got {score:.1f}"
    )


def test_real_vs_estimated_same_visibility_gap():
    """
    Real AOD (near-optimal 0.18) at 20 km should beat estimated AOD at
    20 km by a meaningful margin (≥ 15 pts), because known-good data
    should be rewarded over unknown data.
    """
    real  = engine.atmosphere_score(20_000.0, 0.18, 60.0)
    est   = engine.atmosphere_score(20_000.0, None, 60.0)
    assert real - est >= 15.0, (
        f"Real AOD ({real:.1f}) should beat estimated ({est:.1f}) by ≥ 15 pts"
    )


# ---------------------------------------------------------------------------
# Fix 3: Consistency bonus capped at +3 (was +5)
# ---------------------------------------------------------------------------

def test_consistency_bonus_max_is_3():
    """
    When all 4 window points score ≥ 50, the consistency bonus should be
    exactly 3.0 pts — not the old 5.0.
    """
    result = engine.score_window([
        ("-15m", 65.0), ("sunset", 68.0), ("+15m", 70.0), ("+30m", 66.0)
    ])
    assert result.consistency_bonus <= 3.0, (
        f"Consistency bonus should be capped at 3, got {result.consistency_bonus}"
    )


def test_consistency_bonus_zero_when_none_good():
    """When no window point reaches 50, the consistency bonus is 0."""
    result = engine.score_window([
        ("-15m", 30.0), ("sunset", 35.0), ("+15m", 40.0), ("+30m", 28.0)
    ])
    assert result.consistency_bonus == 0.0, (
        f"No points ≥ 50 → bonus should be 0, got {result.consistency_bonus}"
    )


# ---------------------------------------------------------------------------
# End-to-end scenario tests: realistic full-pipeline scores
# ---------------------------------------------------------------------------

def test_mediocre_archive_day_not_epic():
    """
    A genuinely mediocre archive day (mid/low cloud dominated, little high
    cloud, 15 km default visibility, estimated AOD) must NOT score Epic.

    Before fixes: ~78 (approaching Epic).
    After fixes:  ~67 (low end of Great) — a meaningful ~11 pt correction.

    Note: scoring "Great" at 67 is acceptable — the cloud structure is not
    terrible. The important thing is that the score is no longer inflated to
    near-Epic by the phantom atmosphere default.
    """
    snap = _snap(
        cloud_low=25.0, cloud_mid=30.0, cloud_high=10.0, cloud_total=58.0,
        visibility_m=15_000.0,
        relative_humidity=65.0,
        aerosol_optical_depth=None, aerosol_is_estimated=True,
        sun_elevation_deg=1.5,
    )
    result = engine.score(snap, horizon_obstruction_deg=2.0)
    window = engine.score_window([
        ("-15m", result.physics_score), ("sunset", result.physics_score),
        ("+15m", result.physics_score), ("+30m", result.physics_score),
    ])
    assert window.final_score < 75.0, (
        f"Mediocre archive day should score < 75, got {window.final_score:.1f}"
    )
    assert engine.score_to_category(window.final_score) != "Epic", (
        f"Mediocre archive day must not be Epic, got {window.final_score:.1f}"
    )


def test_overcast_archive_day_is_decent_or_worse():
    """
    An overcast archive day (heavy low cloud, no real atmosphere data) should
    score Decent or Poor — definitely not Great or Epic.
    Before fixes: overcast scored ~60 (Good) due to inflated atmosphere.
    """
    snap = _snap(
        cloud_low=75.0, cloud_mid=40.0, cloud_high=10.0, cloud_total=90.0,
        visibility_m=15_000.0,
        relative_humidity=80.0,
        aerosol_optical_depth=None, aerosol_is_estimated=True,
        precipitation_mm=0.0,
        sun_elevation_deg=1.0,
    )
    result = engine.score(snap, horizon_obstruction_deg=2.0)
    window = engine.score_window([("sunset", result.physics_score)] * 4)
    assert window.final_score < 55.0, (
        f"Overcast archive day should score < 55, got {window.final_score:.1f}"
    )


def test_genuinely_good_day_real_data_still_great():
    """
    A genuinely good day with REAL data (40 km vis, real AOD, ideal clouds)
    must still score Great or Epic. The fixes should not hurt good days.
    This simulates a recent day where the forecast API returned real values.
    """
    snap = _snap(
        cloud_low=5.0, cloud_mid=15.0, cloud_high=45.0, cloud_total=52.0,
        visibility_m=40_000.0,
        relative_humidity=50.0,
        aerosol_optical_depth=0.15, aerosol_is_estimated=False,
        sun_elevation_deg=1.5,
    )
    result = engine.score(snap, horizon_obstruction_deg=2.0)
    window = engine.score_window([("sunset", result.physics_score)] * 4)
    assert window.final_score >= 72.0, (
        f"Good day with real data should score >= 72, got {window.final_score:.1f}"
    )
    category = engine.score_to_category(window.final_score)
    assert category in ("Great", "Epic"), (
        f"Good day with real data should be Great or Epic, got {category}"
    )


def test_clear_sky_archive_day_not_great():
    """
    A completely clear-sky archive day (no clouds, estimated AOD, 15 km vis)
    should score Good at most — not Great or Epic.
    Clear skies produce pastel tones, not dramatic colour.
    """
    snap = _snap(
        cloud_low=0.0, cloud_mid=0.0, cloud_high=0.0, cloud_total=0.0,
        visibility_m=15_000.0,
        relative_humidity=55.0,
        aerosol_optical_depth=None, aerosol_is_estimated=True,
        sun_elevation_deg=0.5,
    )
    result = engine.score(snap, horizon_obstruction_deg=2.0)
    window = engine.score_window([("sunset", result.physics_score)] * 4)
    assert window.final_score < 65.0, (
        f"Clear-sky archive day should score < 65, got {window.final_score:.1f}"
    )


def test_clear_sky_real_good_vis_gets_some_credit():
    """
    A clear-sky day with real 35 km visibility (recent data) should still
    be at least Decent — it's not a non-event, just not epic.
    The horizon glow bonus gives it some lift.
    """
    snap = _snap(
        cloud_low=0.0, cloud_mid=0.0, cloud_high=0.0, cloud_total=0.0,
        visibility_m=35_000.0,
        relative_humidity=45.0,
        aerosol_optical_depth=0.10, aerosol_is_estimated=False,
        sun_elevation_deg=0.5,
    )
    result = engine.score(snap, horizon_obstruction_deg=2.0)
    assert result.physics_score >= 35.0, (
        f"Clear sky with real good visibility should score >= 35, got {result.physics_score:.1f}"
    )


def test_score_ordering_makes_sense():
    """
    Core ordering sanity: great conditions beat mediocre beat overcast.
    All three use estimated AOD (archive-like) to test the fixed path.
    """
    great = _snap(
        cloud_low=5.0, cloud_mid=15.0, cloud_high=50.0, cloud_total=58.0,
        visibility_m=15_000.0, relative_humidity=55.0,
        aerosol_optical_depth=None, aerosol_is_estimated=True,
    )
    mediocre = _snap(
        cloud_low=25.0, cloud_mid=20.0, cloud_high=20.0, cloud_total=55.0,
        visibility_m=15_000.0, relative_humidity=65.0,
        aerosol_optical_depth=None, aerosol_is_estimated=True,
    )
    bad = _snap(
        cloud_low=80.0, cloud_mid=50.0, cloud_high=10.0, cloud_total=92.0,
        visibility_m=15_000.0, relative_humidity=85.0,
        precipitation_mm=0.5,
        aerosol_optical_depth=None, aerosol_is_estimated=True,
    )

    score_great    = engine.score(great, 2.0).physics_score
    score_mediocre = engine.score(mediocre, 2.0).physics_score
    score_bad      = engine.score(bad, 2.0).physics_score

    assert score_great > score_mediocre > score_bad, (
        f"Expected great > mediocre > bad: "
        f"{score_great:.1f} > {score_mediocre:.1f} > {score_bad:.1f}"
    )


def test_rainy_archive_day_scores_poor():
    """Active rain must still produce a Poor/Decent score even with archive defaults."""
    snap = _snap(
        cloud_low=70.0, cloud_mid=60.0, cloud_high=20.0, cloud_total=88.0,
        visibility_m=15_000.0,
        relative_humidity=88.0,
        precipitation_mm=4.0,
        aerosol_optical_depth=None, aerosol_is_estimated=True,
        sun_elevation_deg=0.5,
    )
    result = engine.score(snap, horizon_obstruction_deg=2.0)
    assert result.physics_score < 40.0, (
        f"Rainy archive day should score < 40, got {result.physics_score:.1f}"
    )
    assert engine.score_to_category(result.physics_score) in ("Poor", "Decent")
