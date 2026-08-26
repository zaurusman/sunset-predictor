"""
Scoring calibration tests.

These are sanity checks against systematic score inflation, especially for
archive dates (>7 days old) which feed the climatology every displayed score
is ranked against.

The atmosphere tests here changed shape in Phase 3. There is no longer a
visibility default to calibrate: the archive reports no visibility, so the
snapshot carries None and the scorer leaves it out. What is checked instead is
that clean air scores high, hazy air scores low, and a missing field neither
rewards nor punishes an evening.

Each test documents what the score SHOULD be for a realistic scenario and
why. Run these after any change to scoring_engine.py.
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
# Atmosphere: clean air is the ingredient
#
# This replaced a bell curve peaking at AOD 0.18, which scored pristine air
# (0.03) at ~56 and treated a smog layer as ideal. See aerosol_clarity().
# ---------------------------------------------------------------------------

def test_aerosol_response_is_monotone_decreasing():
    """The property the whole change rests on. If this ever fails, someone has
    reintroduced the bell curve."""
    prev = 101.0
    for aod in [i / 100.0 for i in range(0, 151)]:
        s = engine.aerosol_clarity(aod)
        assert s <= prev + 1e-9, f"clarity rose at AOD {aod:.2f}: {s:.1f} > {prev:.1f}"
        prev = s


def test_pristine_air_scores_near_perfect():
    """Under the old bell curve this scored ~56 — the single worst mis-ranking
    in the engine, because it demoted exactly the post-frontal evenings that
    produce the best colour."""
    assert engine.aerosol_clarity(0.03) >= 95.0


def test_heavy_haze_scores_low():
    assert engine.aerosol_clarity(0.8) <= 20.0
    assert engine.aerosol_clarity(0.5) <= 50.0


def test_missing_visibility_is_neither_reward_nor_penalty():
    """An archive day (no visibility) with the same air as a forecast day
    should not be systematically higher or lower — the old 15 km default made
    it a fixed offset instead."""
    clean_no_vis = engine.atmosphere_score(None, 0.05)
    hazy_no_vis = engine.atmosphere_score(None, 0.6)
    assert clean_no_vis >= 95.0
    assert hazy_no_vis <= 40.0


def test_atmosphere_falls_back_when_only_one_signal_exists():
    assert engine.atmosphere_score(25_000.0, None) == 100.0   # visibility only
    assert engine.atmosphere_score(None, 0.15) == 88.0        # aerosol only
    # Neither reported: neutral, so a data gap does not move the score.
    assert 50.0 <= engine.atmosphere_score(None, None) <= 70.0


def test_poor_visibility_pulls_a_clean_reading_down():
    """Visibility is a second look at the same physics and keeps a minority
    share, so it corrects a clean AOD reading without overturning it."""
    clear = engine.atmosphere_score(30_000.0, 0.08)
    murky = engine.atmosphere_score(3_000.0, 0.08)
    assert clear > murky
    assert murky < 80.0


def test_surface_humidity_is_not_charged_twice():
    """Atmosphere used to carry its own humidity penalty while moisture carried
    another. Moisture is now scored once, as a column, in moisture_score."""
    import inspect
    sig = inspect.signature(engine.atmosphere_score)
    assert "humidity_pct" not in sig.parameters


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
