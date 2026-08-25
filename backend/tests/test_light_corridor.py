"""Tests for the light-corridor model.

The corridor is the mechanism the engine was missing entirely: whether sunset
light can *reach* the clouds overhead, which depends on the atmosphere 100-400 km
away along the sunset azimuth, not on the observer's own grid cell.

See docs/scoring-v2-plan.md (D3) for the diagnosis and the sources.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from app.services.astronomy_service import AstronomyService
from app.services.scoring_engine import (
    CORRIDOR_FLOOR,
    LAYER_HEIGHT_KM,
    ScoringEngine,
)
from app.schemas.weather import WeatherSnapshot
from app.utils.geo import (
    EARTH_RADIUS_KM,
    destination_point,
    horizon_tangent_distance_km,
)

engine = ScoringEngine()
astro = AstronomyService()

DISTANCES = [60.0, 120.0, 180.0, 240.0, 320.0, 400.0]


def corridor(low: float, mid: float = 0.0) -> list[tuple[float, float, float]]:
    """A uniform corridor with *low*/*mid* cloud at every sample distance."""
    return [(d, low, mid) for d in DISTANCES]


def _snap(**kw) -> WeatherSnapshot:
    defaults = dict(
        cloud_low=0.0, cloud_mid=10.0, cloud_high=45.0, cloud_total=50.0,
        visibility_m=20_000.0, relative_humidity=55.0, dewpoint_c=8.0,
        temperature_c=18.0, precipitation_mm=0.0, wind_speed_kmh=10.0,
        pressure_hpa=1013.0, aerosol_optical_depth=0.15,
        aerosol_is_estimated=False, sun_elevation_deg=1.0, data_source="forecast",
    )
    defaults.update(kw)
    return WeatherSnapshot(**defaults)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_tangent_distances_match_known_layer_scales():
    """Derived tangent distances should reproduce US10459119's empirical values
    (low ~130-160 km, mid ~225-320 km, high ~400 km)."""
    assert horizon_tangent_distance_km(1.0) == pytest.approx(113, abs=3)
    assert horizon_tangent_distance_km(4.0) == pytest.approx(226, abs=3)
    assert horizon_tangent_distance_km(9.0) == pytest.approx(339, abs=3)


def test_tangent_distance_is_zero_at_ground_level():
    assert horizon_tangent_distance_km(0.0) == 0.0
    assert horizon_tangent_distance_km(-5.0) == 0.0


def test_destination_point_due_north_increases_latitude():
    lat, lon = destination_point(0.0, 0.0, bearing_deg=0.0, distance_km=111.19)
    assert lat == pytest.approx(1.0, abs=0.01)
    assert lon == pytest.approx(0.0, abs=0.01)


def test_destination_point_due_west_decreases_longitude():
    lat, lon = destination_point(0.0, 0.0, bearing_deg=270.0, distance_km=111.19)
    assert lon == pytest.approx(-1.0, abs=0.01)
    assert lat == pytest.approx(0.0, abs=0.01)


def test_destination_point_normalises_across_antimeridian():
    """A corridor running west from Fiji must not produce lon < -180."""
    _lat, lon = destination_point(-17.7, 178.0, bearing_deg=270.0, distance_km=600.0)
    assert -180.0 <= lon <= 180.0


def test_destination_point_stays_valid_over_the_pole():
    lat, lon = destination_point(89.0, 0.0, bearing_deg=0.0, distance_km=500.0)
    assert -90.0 <= lat <= 90.0
    assert -180.0 <= lon <= 180.0


def test_destination_distance_is_actually_the_requested_distance():
    """Round-trip via the haversine formula."""
    lat0, lon0, d = 32.08, 34.78, 250.0
    lat1, lon1 = destination_point(lat0, lon0, 283.0, d)
    p1, p2 = math.radians(lat0), math.radians(lat1)
    dp, dl = p2 - p1, math.radians(lon1 - lon0)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    measured = 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
    assert measured == pytest.approx(d, rel=0.001)


def test_sunset_azimuth_is_seasonal():
    """A fixed 'due west' assumption would sample the wrong atmosphere."""
    summer = astro.get_sunset_azimuth(51.51, -0.13, date(2026, 6, 21))
    winter = astro.get_sunset_azimuth(51.51, -0.13, date(2026, 12, 21))
    assert summer > 300.0, f"London midsummer should set well north of west, got {summer}"
    assert winter < 240.0, f"London midwinter should set well south of west, got {winter}"
    assert summer - winter > 60.0, "seasonal azimuth swing should be large at 51°N"


# ---------------------------------------------------------------------------
# Transmittance
# ---------------------------------------------------------------------------


def test_clear_corridor_transmits_fully():
    assert engine.corridor_transmittance(corridor(low=0.0), 9.0) == pytest.approx(1.0)


def test_solid_low_deck_blocks_the_corridor():
    assert engine.corridor_transmittance(corridor(low=100.0), 9.0) == pytest.approx(0.0)


def test_mid_cloud_blocks_only_partially():
    """Mid cloud occludes at half weight; high cirrus is not counted at all."""
    t = engine.corridor_transmittance(corridor(low=0.0, mid=100.0), 9.0)
    assert t == pytest.approx(0.5, abs=0.01)


def test_no_samples_means_no_adjustment():
    """A corridor outage must degrade to the previous behaviour, not to zero."""
    assert engine.corridor_transmittance([], 9.0) == 1.0
    assert engine.light_corridor_factor([], 10.0, 10.0, 40.0) == 1.0


def test_layers_are_sensitive_to_different_parts_of_the_corridor():
    """Blocking only the far field should hurt high cloud more than low cloud.

    This is the core geometric claim: a high cirrus deck is lit through the
    atmosphere ~340 km away, a low deck through ~113 km away.
    """
    far_block = [(d, 100.0 if d >= 320.0 else 0.0, 0.0) for d in DISTANCES]
    t_high = engine.corridor_transmittance(far_block, LAYER_HEIGHT_KM["high"])
    t_low = engine.corridor_transmittance(far_block, LAYER_HEIGHT_KM["low"])
    assert t_high < t_low, (
        f"far-field blocking should hurt high cloud most: high={t_high:.2f} low={t_low:.2f}"
    )


def test_near_field_blocking_hurts_low_cloud_most():
    """The converse of the above — the geometry must run both ways."""
    near_block = [(d, 100.0 if d <= 120.0 else 0.0, 0.0) for d in DISTANCES]
    t_high = engine.corridor_transmittance(near_block, LAYER_HEIGHT_KM["high"])
    t_low = engine.corridor_transmittance(near_block, LAYER_HEIGHT_KM["low"])
    assert t_low < t_high


def test_transmittance_is_monotone_in_blocking():
    prev = 1.1
    for low in (0.0, 25.0, 50.0, 75.0, 100.0):
        t = engine.corridor_transmittance(corridor(low), 9.0)
        assert t < prev, "more upstream cloud must never transmit more light"
        prev = t


# ---------------------------------------------------------------------------
# The multiplier
# ---------------------------------------------------------------------------


def test_factor_is_bounded_by_the_floor():
    blocked = engine.light_corridor_factor(corridor(low=100.0), 0.0, 0.0, 60.0)
    clear = engine.light_corridor_factor(corridor(low=0.0), 0.0, 0.0, 60.0)
    assert blocked == pytest.approx(CORRIDOR_FLOOR)
    assert clear == pytest.approx(1.0)


def test_clear_overhead_sky_uses_the_far_field():
    """With nothing overhead, the colour is horizon glow — far-field decides."""
    far_block = [(d, 100.0 if d >= 320.0 else 0.0, 0.0) for d in DISTANCES]
    factor = engine.light_corridor_factor(far_block, cloud_low=0.0, cloud_mid=0.0, cloud_high=0.0)
    assert factor < 0.8, f"blocked far field should dim a clear-sky glow, got {factor:.2f}"


# ---------------------------------------------------------------------------
# Effect on the score — the behaviour that actually matters
# ---------------------------------------------------------------------------


def test_blocked_corridor_demotes_a_textbook_sky():
    """The headline case: a perfect local sky with an overcast deck upstream.

    Before the corridor existed this evening scored as though it were flawless.
    """
    snap = _snap(cloud_low=0.0, cloud_mid=15.0, cloud_high=50.0, cloud_total=55.0)
    open_sky = engine.score(snap, 2.0, corridor_samples=corridor(low=0.0))
    blocked = engine.score(snap, 2.0, corridor_samples=corridor(low=100.0))

    assert blocked.physics_score < open_sky.physics_score - 10.0, (
        f"a blocked corridor must materially demote the score: "
        f"open={open_sky.physics_score:.1f} blocked={blocked.physics_score:.1f}"
    )
    assert blocked.cloud_quality < open_sky.cloud_quality


def test_corridor_factor_is_reported_in_the_breakdown():
    """Users are told why the score dropped, not just that it did."""
    snap = _snap()
    result = engine.score(snap, 2.0, corridor_samples=corridor(low=80.0))
    breakdown = result.to_physics_breakdown()
    assert breakdown.light_corridor_factor is not None
    assert 0.0 <= breakdown.light_corridor_factor <= 1.0


def test_absent_corridor_leaves_score_and_breakdown_untouched():
    """Backwards compatibility: no corridor data → identical score to before."""
    snap = _snap()
    without = engine.score(snap, 2.0)
    explicit_empty = engine.score(snap, 2.0, corridor_samples=[])
    assert without.physics_score == explicit_empty.physics_score
    assert without.to_physics_breakdown().light_corridor_factor is None


def test_corridor_never_inflates_a_score():
    """The corridor can only gate light, never manufacture it."""
    snap = _snap()
    base = engine.score(snap, 2.0).physics_score
    for low in (0.0, 30.0, 60.0, 100.0):
        assert engine.score(snap, 2.0, corridor_samples=corridor(low)).physics_score <= base + 1e-9


def test_score_stays_in_range_under_full_blocking():
    snap = _snap(cloud_low=90.0, cloud_mid=80.0, cloud_high=10.0, cloud_total=98.0)
    result = engine.score(snap, 2.0, corridor_samples=corridor(low=100.0, mid=100.0))
    assert 0.0 <= result.physics_score <= 100.0
    assert 0.0 <= result.cloud_quality <= 100.0
