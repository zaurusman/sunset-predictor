"""Tests for climatological ranking.

The displayed score is now ABSOLUTE — the raw physics score — because a rank
cannot improve when the model improves (see PredictionService._calibrate). The
rank survives as CONTEXT: "better than 31 % of evenings here", taken against a
seasonal window rather than the whole year.

What must hold:

  - ranking is monotone, and seasonal
  - a cold location degrades to a global curve instead of leaking nonsense
  - a curve built by an older engine is never served to a newer one
  - nothing here can block or break a prediction
"""
from __future__ import annotations

from datetime import date

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


# NOTE: there was a test here asserting that the CALIBRATION_ANCHORS percentiles
# land exactly on category boundaries. It was correct while the display score
# WAS the percentile. The bands are now absolute thresholds on the physics
# score, so the two are deliberately no longer coupled — coupling them again
# would silently reintroduce the self-normalising behaviour this replaced.


def test_absolute_bands_let_climates_differ():
    """The opposite of the property this file used to assert.

    Under percentile display, every climate produced identical band shares by
    construction. On an absolute scale a location with genuinely better skies
    must be ABLE to earn more good evenings — that is what makes the number
    mean "how good will the sky look" rather than "how does it rank here".
    """
    sunny = [engine.score_to_category(v) for v in [72.0] * 50 + [60.0] * 50]
    dull = [engine.score_to_category(v) for v in [40.0] * 50 + [30.0] * 50]
    assert sunny.count("Great") > dull.count("Great")
    assert dull.count("Poor") > sunny.count("Poor")


def test_absolute_bands_are_ordered_and_cover_the_range():
    assert engine.score_to_category(0.0) == "Poor"
    assert engine.score_to_category(100.0) == "Epic"
    seen = [engine.score_to_category(float(v)) for v in range(0, 101)]
    assert set(seen) == {"Poor", "Decent", "Good", "Great", "Epic"}


def test_the_rated_evening_reads_as_good():
    """Tel Aviv, 2026-08-23. The user rated it 4/5 from a photo and judged it
    around 75; the model's raw physics score is 57.4. Under the old rank
    display it read 30.6 and "Poor" — a beautiful but typical evening is
    mid-table by construction. Absolute display puts it in the right band even
    though the number is still lower than the user's own estimate."""
    assert engine.score_to_category(57.4) == "Good"


# ---------------------------------------------------------------------------
# Service behaviour: never block, never break
# ---------------------------------------------------------------------------


def _year_of(values) -> list[tuple[int, float]]:
    """Spread *values* evenly across the calendar, as build() stores them."""
    vals = list(values)
    return [
        (1 + int(i * 365 / len(vals)), float(v)) for i, v in enumerate(vals)
    ]


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
    svc._cache.set(svc._key(32.08, 34.78), _year_of(range(100)))
    percentile, is_local = svc.percentile_of(32.08, 34.78, 90.0, on_date=date(2026, 3, 1))
    assert is_local is True
    assert percentile > 0.85


def test_nearby_coordinates_share_one_curve():
    """Climate varies over ~100 km; one curve should serve a metro area."""
    svc = _service()
    svc._cache.set(svc._key(32.08, 34.78), _year_of(range(100)))
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
    svc._cache.set(svc._key(32.08, 34.78), _year_of(range(100)))
    assert svc.percentile_of(32.08, 34.78, 50.0)[1] is True

    original = cs.SCALE_VERSION
    try:
        cs.SCALE_VERSION = original + 1
        assert svc.percentile_of(32.08, 34.78, 50.0)[1] is False, (
            "a scale bump must orphan curves built by the previous engine"
        )
    finally:
        cs.SCALE_VERSION = original


# ---------------------------------------------------------------------------
# Seasonal ranking
# ---------------------------------------------------------------------------


def test_rank_is_taken_against_the_same_time_of_year():
    """The whole point of the seasonal window. In a climate whose summers are
    dull and winters dramatic, a middling summer evening should rank WELL for
    summer even though it ranks poorly against the full year."""
    from app.services.climatology_service import _seasonal_window

    summer = [(doy, 40.0 + (doy % 5)) for doy in range(150, 240)]
    winter = [(doy, 80.0 + (doy % 5)) for doy in range(1, 90)]
    entries = summer + winter

    svc = _service()
    svc._cache.set(svc._key(32.08, 34.78), entries)

    # 15 July, a 44 in a summer of 40-44s.
    seasonal, is_local = svc.percentile_of(32.08, 34.78, 44.0, on_date=date(2026, 7, 15))
    assert is_local is True
    assert seasonal > 0.6, f"a good summer evening should rank well for summer, got {seasonal}"

    # Against the whole year it would be near the bottom.
    whole_year = _rank_in_sorted(sorted(v for _, v in entries), 44.0)
    assert whole_year < 0.55
    assert seasonal > whole_year


def test_seasonal_window_wraps_the_new_year():
    from app.services.climatology_service import _seasonal_window

    entries = [(360, 1.0), (5, 2.0), (180, 3.0)]
    picked = _seasonal_window(entries, day_of_year=2)
    assert 1.0 in picked and 2.0 in picked, "late December must count as near 2 January"
    assert 3.0 not in picked, "midsummer must not"


def test_thin_season_falls_back_to_the_full_year():
    """A partial year of data must not be ranked against a handful of days."""
    svc = _service()
    svc._cache.set(svc._key(32.08, 34.78), [(200 + i, float(i)) for i in range(10)])
    percentile, is_local = svc.percentile_of(32.08, 34.78, 5.0, on_date=date(2026, 7, 20))
    assert is_local is True
    assert 0.0 <= percentile <= 1.0
