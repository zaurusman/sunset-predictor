"""Mathematical helpers for the scoring engine."""
from __future__ import annotations

import math
from typing import Sequence


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def bell_curve(x: float, peak: float, sigma: float) -> float:
    """
    Gaussian bell function returning a value in [0, 1].

    Returns 1.0 when x == peak, and falls off symmetrically with width
    controlled by *sigma* (standard deviation).

    Args:
        x: Input value.
        peak: The x-coordinate of the maximum (returns 1.0 here).
        sigma: Controls the width; larger = broader bell.
    """
    if sigma <= 0:
        return 1.0 if x == peak else 0.0
    return math.exp(-0.5 * ((x - peak) / sigma) ** 2)


def linear_ramp(
    x: float,
    x_start: float,
    x_end: float,
    y_start: float = 0.0,
    y_end: float = 100.0,
) -> float:
    """
    Linearly interpolate between y_start and y_end over [x_start, x_end].

    Values outside the range are clamped to y_start / y_end.
    """
    if x_end == x_start:
        return y_start
    t = (x - x_start) / (x_end - x_start)
    t = clamp(t, 0.0, 1.0)
    return y_start + t * (y_end - y_start)


def weighted_average(
    scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """
    Compute a weighted average of *scores* using *weights*.

    Weights need not sum to 1; they are normalised internally.
    Keys present in *scores* but missing from *weights* are ignored.
    """
    total_weight = 0.0
    total_value = 0.0
    for key, score in scores.items():
        w = weights.get(key, 0.0)
        total_value += score * w
        total_weight += w
    if total_weight == 0:
        return 0.0
    return total_value / total_weight


def normalize_to_100(values: Sequence[float]) -> list[float]:
    """Min-max normalise a sequence of floats to [0, 100]."""
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span == 0:
        return [50.0] * len(values)
    return [(v - lo) / span * 100.0 for v in values]
