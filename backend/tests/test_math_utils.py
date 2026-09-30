"""Tests for app.utils.math_utils."""
from app.utils.math_utils import round_half_up


def test_round_half_up_matches_js_math_round_on_ties():
    # Python's built-in round() ties to even (round(54.5) == 54); the frontend
    # uses Math.round (Math.round(54.5) == 55). round_half_up must match JS,
    # since it feeds score_to_category() and the frontend displays
    # Math.round(score) right next to the category badge.
    assert round_half_up(54.5) == 55
    assert round_half_up(37.5) == 38
    assert round_half_up(0.5) == 1


def test_round_half_up_non_ties():
    assert round_half_up(54.4) == 54
    assert round_half_up(54.6) == 55
    assert round_half_up(0.0) == 0
    assert round_half_up(100.0) == 100
