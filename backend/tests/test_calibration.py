"""Tests for percentile calibration (Phase 5 of docs/scoring-v2-plan.md).

The displayed score is a rank against the location's own climatology, not the
raw physics score. What must hold:

  - the mapping is monotone (a better evening never displays lower)
  - band shares come out as designed, in ANY climate
  - a cold location degrades to a global curve instead of leaking raw scores
  - nothing here can block or break a prediction
"""
from __future__ import annotations

import pytest

from app.services.climatology_service import (
    REFERENCE_QUANTILES,
    _rank_in_quantiles,
    _rank_in_sorted,
)
from app.services.scoring_engine import CALIBRATION_ANCHORS, ScoringEngine

engine = ScoringEngine()


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_rank_in_sorted_spans_the_range():
    curve = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _rank_in_sorted(curve, 5.0) == 0.0
    assert _rank_in_sorted(curve, 100.0) == 1.0
    assert 0.4 < _rank_in_sorted(curve, 30.0) < 0.6


def test_rank_in_sorted_uses_midpoint_for_ties():
    """The raw scale still has flat spots; a value inside one should land in
    the middle of the run, not at either end."""
    curve = [10.0] + [50.0] * 8 + [90.0]
    rank = _rank_in_sorted(curve, 50.0)
    assert rank == pytest.approx(0.5, abs=0.05)


def test_rank_in_sorted_handles_empty():
    assert _rank_in_sorted([], 42.0) == 0.5


def test_rank_in_sorted_is_monotone():
    curve = sorted([3.0, 17.0, 24.0, 41.0, 47.0, 55.0, 68.0, 82.0])
    prev = -1.0
    for v in range(0, 101, 5):
        r = _rank_in_sorted(curve, float(v))
        assert r >= prev, "rank must never decrease as the score rises"
        prev = r


def test_reference_curve_ranks_sensibly():
    """The global fallback must place typical scores in plausible places."""
    assert _rank_in_quantiles(REFERENCE_QUANTILES, 0.0) == 0.0
    assert _rank_in_quantiles(REFERENCE_QUANTILES, 100.0) == 1.0
    # Probe the curve's own median rather than a hard-coded score: the raw
    # scale moves whenever a component curve changes, and a literal here would
    # quietly become a test of last quarter's scoring engine.
    median_raw = REFERENCE_QUANTILES[len(REFERENCE_QUANTILES) // 2]
    mid = _rank_in_quantiles(REFERENCE_QUANTILES, median_raw)
    assert 0.4 < mid < 0.6, f"the curve's own median should rank mid-table, got {mid}"
    assert _rank_in_quantiles(REFERENCE_QUANTILES, REFERENCE_QUANTILES[-2]) > 0.9


def test_reference_curve_is_sorted():
    """A non-monotone quantile curve would produce non-monotone display scores."""
    assert REFERENCE_QUANTILES == sorted(REFERENCE_QUANTILES)


# ---------------------------------------------------------------------------
# Percentile → display mapping
# ---------------------------------------------------------------------------


def test_mapping_is_monotone():
    prev = -1.0
    for i in range(0, 101):
        d = engine.percentile_to_display_score(i / 100.0)
        assert d >= prev, "a better evening must never display a lower number"
        prev = d


def test_mapping_hits_its_anchors():
    for percentile, expected in CALIBRATION_ANCHORS:
        assert engine.percentile_to_display_score(percentile) == pytest.approx(expected)


def test_mapping_clamps_out_of_range_input():
    assert engine.percentile_to_display_score(-1.0) == 0.0
    assert engine.percentile_to_display_score(2.0) == 100.0


def test_band_boundaries_land_on_category_edges():
    """Each anchor must sit exactly on the boundary it claims to define."""
    assert engine.score_to_category(engine.percentile_to_display_score(0.30)) == "Decent"
    assert engine.score_to_category(engine.percentile_to_display_score(0.68)) == "Good"
    assert engine.score_to_category(engine.percentile_to_display_score(0.88)) == "Great"
    assert engine.score_to_category(engine.percentile_to_display_score(0.97)) == "Epic"
    assert engine.score_to_category(engine.percentile_to_display_score(0.29)) == "Poor"


# ---------------------------------------------------------------------------
# The property that motivated the whole phase
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "climate",
    [
        pytest.param([40.0 + i * 0.02 for i in range(365)], id="degenerate-dry"),
        pytest.param([10.0 + i * 0.22 for i in range(365)], id="wide-maritime"),
        pytest.param([i * 0.05 for i in range(365)], id="uniform"),
    ],
)
def test_band_shares_are_identical_across_climates(climate):
    """This is the point of Phase 5.

    Fixed cutoffs gave "Epic" on 10-15 % of evenings in one city and "Decent" on
    75 % in another. Percentile anchoring must produce the same shares whatever
    the underlying distribution looks like — including the near-constant Tel Aviv
    case that fixed cutoffs collapsed entirely.
    """
    curve = sorted(climate)
    displayed = [
        engine.percentile_to_display_score(_rank_in_sorted(curve, v)) for v in climate
    ]
    n = len(displayed)
    shares = {}
    for d in displayed:
        c = engine.score_to_category(d)
        shares[c] = shares.get(c, 0) + 1

    # Target shares from CALIBRATION_ANCHORS: 30 / 38 / 20 / 9 / 3 per cent.
    assert shares.get("Epic", 0) / n == pytest.approx(0.03, abs=0.02)
    assert shares.get("Great", 0) / n == pytest.approx(0.09, abs=0.03)
    assert shares.get("Good", 0) / n == pytest.approx(0.20, abs=0.04)
    assert shares.get("Decent", 0) / n == pytest.approx(0.38, abs=0.05)
    assert shares.get("Poor", 0) / n == pytest.approx(0.30, abs=0.05)


def test_degenerate_climate_still_gets_a_full_spread():
    """Tel Aviv scored 61.x on >40 % of the year. Even that must rank out."""
    curve = sorted([61.0] * 200 + [61.1] * 100 + [30.0] * 30 + [85.0] * 35)
    displayed = sorted(
        engine.percentile_to_display_score(_rank_in_sorted(curve, v)) for v in curve
    )
    assert displayed[-1] - displayed[0] > 50.0, (
        "a near-constant climate must still produce a usable spread of displayed scores"
    )


# ---------------------------------------------------------------------------
# Service behaviour: never block, never break
# ---------------------------------------------------------------------------


def _service():
    from app.services.climatology_service import ClimatologyService
    from app.utils.cache import TTLCache

    return ClimatologyService(
        weather_service=None,        # unused on the read path
        astro_service=None,
        scoring_engine=engine,
        cache=TTLCache(ttl_seconds=60, persist_path=None),
    )


def test_cold_location_falls_back_to_reference_not_raw():
    svc = _service()
    percentile, is_local = svc.percentile_of(32.08, 34.78, 47.0)
    assert is_local is False
    assert 0.0 <= percentile <= 1.0
    assert svc.is_warm(32.08, 34.78) is False


def test_warm_location_uses_its_own_curve():
    svc = _service()
    svc._cache.set(svc._key(32.08, 34.78), sorted(float(i) for i in range(100)))
    percentile, is_local = svc.percentile_of(32.08, 34.78, 90.0)
    assert is_local is True
    assert percentile > 0.85


def test_nearby_coordinates_share_one_curve():
    """Climate varies over ~100 km; one curve should serve a metro area."""
    svc = _service()
    svc._cache.set(svc._key(32.08, 34.78), sorted(float(i) for i in range(100)))
    assert svc.percentile_of(32.10, 34.80, 50.0)[1] is True


def test_warm_in_background_is_a_noop_without_an_event_loop():
    """Called from sync code it must degrade quietly, not raise."""
    svc = _service()
    svc.warm_in_background(32.08, 34.78)   # no running loop
    assert svc.is_warm(32.08, 34.78) is False


def test_cache_key_is_scoped_to_the_scale_version():
    """A curve built by an older scoring engine must not be served to a newer
    one. Curves persist to disk for 30 days, so without this a scoring change
    silently ranks new scores against an old distribution."""
    from app.services import climatology_service as cs

    svc = _service()
    svc._cache.set(svc._key(32.08, 34.78), sorted(float(i) for i in range(100)))
    assert svc.percentile_of(32.08, 34.78, 50.0)[1] is True

    original = cs.SCALE_VERSION
    try:
        cs.SCALE_VERSION = original + 1
        assert svc.percentile_of(32.08, 34.78, 50.0)[1] is False, (
            "a scale bump must orphan curves built by the previous engine"
        )
    finally:
        cs.SCALE_VERSION = original
