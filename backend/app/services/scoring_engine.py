"""
Physics-informed sunset beauty scoring engine.

DESIGN PRINCIPLE
----------------
The score answers "should I go outside to watch the sunset today?"

Two layers:
  1. Single-snapshot scoring — deterministic, unit-testable. Evaluates
     atmospheric conditions at one moment in time.
  2. Window aggregation — takes four snapshots around sunset (-15m, sunset,
     +15m, +30m) and derives a daily score that reflects the best likely
     viewing moment, with bonuses for consistency and penalties for volatility.

COMPONENTS (4 total, weights configurable)
------------------------------------------
1. Cloud Quality  (42 %) — cloud distribution at sunset
2. Atmosphere     (28 %) — visibility, aerosol, humidity
3. Moisture       (20 %) — rain, clearing trend, humidity
4. Horizon        (10 %) — permanent obstruction (buildings, mountains)

Each component returns a score in [0, 100].  The final beauty score is a
weighted average, clamped to [0, 100].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from app.schemas.prediction import PhysicsBreakdown
from app.schemas.weather import WeatherSnapshot
from app.utils.math_utils import bell_curve, clamp, weighted_average

# ---------------------------------------------------------------------------
# Default weights — override via ScoringEngine(weights={…})
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "cloud_quality": 0.42,
    "atmosphere": 0.28,
    "moisture": 0.20,
    "horizon": 0.10,
}

# ---------------------------------------------------------------------------
# Score → category thresholds
# ---------------------------------------------------------------------------

SCORE_THRESHOLDS: list[tuple[float, str]] = [
    (80, "Epic"),
    (65, "Great"),
    (50, "Good"),
    (30, "Decent"),
    (0, "Poor"),
]

# Score at which we recommend going outside
GO_OUTSIDE_THRESHOLD = 45.0


@dataclass
class ScoringResult:
    """Raw scoring output before ML calibration."""

    cloud_quality: float
    atmosphere: float
    moisture: float
    horizon: float
    physics_score: float  # weighted average
    confidence: float
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def to_physics_breakdown(self) -> PhysicsBreakdown:
        return PhysicsBreakdown(
            cloud_quality_score=round(self.cloud_quality, 1),
            atmosphere_score=round(self.atmosphere, 1),
            moisture_score=round(self.moisture, 1),
            horizon_score=round(self.horizon, 1),
            weighted_physics_score=round(self.physics_score, 1),
            component_weights=self.weights,
        )


@dataclass
class WindowResult:
    """
    Result from scoring the full 45-minute sunset viewing window.

    The final_score is derived from the best single-point score, adjusted
    for consistency and volatility across the four window positions.
    """

    final_score: float
    best_label: str          # e.g. "+15m" — the window point that scored highest
    best_score: float
    window_scores: dict[str, float]   # label → snapshot physics score
    go_outside: bool
    consistency_bonus: float
    volatility_penalty: float


class ScoringEngine:
    """
    Computes physics-informed sunset beauty scores from WeatherSnapshot(s).

    All scoring methods are deterministic and unit-testable in isolation.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self._weights = weights or dict(DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Top-level: single-snapshot
    # ------------------------------------------------------------------

    def score(
        self, weather: WeatherSnapshot, horizon_obstruction_deg: float
    ) -> ScoringResult:
        """
        Compute the full scoring breakdown for *weather* at *horizon_obstruction_deg*.

        Returns a ScoringResult containing per-component scores, the weighted
        physics score, and a confidence estimate.
        """
        cq = self.cloud_quality_score(
            weather.cloud_low,
            weather.cloud_mid,
            weather.cloud_high,
            weather.cloud_total,
        )
        atm = self.atmosphere_score(
            weather.visibility_m,
            weather.aerosol_optical_depth,
            weather.relative_humidity,
        )
        mst = self.moisture_score(
            weather.precipitation_mm,
            weather.relative_humidity,
            precip_last_3h=weather.precipitation_last_3h_mm,
            pressure_trend=weather.pressure_trend_hpa_3h,
            cloud_trend=weather.cloud_total_trend_3h,
            vis_trend=weather.visibility_trend_3h_m,
        )
        hor = self.horizon_score(horizon_obstruction_deg)

        component_scores = {
            "cloud_quality": cq,
            "atmosphere": atm,
            "moisture": mst,
            "horizon": hor,
        }
        physics_score = clamp(weighted_average(component_scores, self._weights))

        confidence = self.compute_confidence(
            weather=weather,
            component_scores=component_scores,
            physics_score=physics_score,
        )

        return ScoringResult(
            cloud_quality=cq,
            atmosphere=atm,
            moisture=mst,
            horizon=hor,
            physics_score=physics_score,
            confidence=confidence,
            weights=dict(self._weights),
        )

    # ------------------------------------------------------------------
    # Top-level: window aggregation
    # ------------------------------------------------------------------

    def score_window(
        self, scored_points: list[tuple[str, float]]
    ) -> WindowResult:
        """
        Aggregate per-snapshot scores across the sunset viewing window.

        Parameters
        ----------
        scored_points : list of (label, score) pairs, e.g.
            [("-15m", 62.0), ("sunset", 71.0), ("+15m", 75.0), ("+30m", 68.0)]

        Aggregation strategy
        --------------------
        - The best single point is the dominant signal (max-first).
        - Slight (+2 pt) preference for afterglow (+15m) when it ties the best.
        - Consistency bonus: up to +5 pts when ≥3 of 4 points clear 50.
        - Volatility penalty: up to −8 pts when spread exceeds 30 points.
        - Result is clamped to [0, 100].
        """
        if not scored_points:
            raise ValueError("score_window requires at least one scored point")

        scores = {label: score for label, score in scored_points}
        vals = list(scores.values())

        # Best point
        best_label = max(scores, key=lambda k: scores[k])
        best_score = scores[best_label]

        # Small afterglow preference: +15m is the afterglow peak.
        # If +15m is within 3 pts of the best, crown it instead.
        afterglow_label = "+15m"
        if (
            afterglow_label in scores
            and afterglow_label != best_label
            and scores[afterglow_label] >= best_score - 3.0
        ):
            best_label = afterglow_label
            best_score = scores[afterglow_label]

        # Consistency bonus: fraction of points ≥ 50, scaled to +5 pts max
        good_count = sum(1 for v in vals if v >= 50.0)
        consistency_bonus = (good_count / len(vals)) * 5.0 if len(vals) > 1 else 0.0

        # Volatility penalty: spread > 30 → scale up to −8 pts
        spread = max(vals) - min(vals)
        volatility_penalty = clamp((spread - 30.0) / 40.0 * 8.0) if spread > 30.0 else 0.0

        final_score = clamp(best_score + consistency_bonus - volatility_penalty)

        return WindowResult(
            final_score=final_score,
            best_label=best_label,
            best_score=best_score,
            window_scores=scores,
            go_outside=final_score >= GO_OUTSIDE_THRESHOLD,
            consistency_bonus=round(consistency_bonus, 2),
            volatility_penalty=round(volatility_penalty, 2),
        )

    # ------------------------------------------------------------------
    # Component 1: Cloud Quality (weight 0.42)
    # ------------------------------------------------------------------

    def cloud_quality_score(
        self,
        low_pct: float,
        mid_pct: float,
        high_pct: float,
        total_pct: float,
    ) -> float:
        """
        Score the cloud distribution for sunset colour potential.

        Design intent
        -------------
        High clouds (cirrus, altocumulus) are the strongest positive: they
        scatter low-angle sunlight into vivid pinks and oranges.  Mid-level
        clouds (altostratus) add texture and some colour.  Low clouds are
        negative — they block the sun near the horizon — but their penalty is
        softened when strong upper clouds are present.

        Key calibration points
        ----------------------
        - High 45%, low < 20%  →  peak colour potential (~85–90)
        - Full overcast (≥90%) →  heavily penalised (<15)
        - Completely clear sky →  mild penalty (not crushed); ~35–40
        - High high + heavy low → upper-cloud offset softens low penalty
        """
        # --- High clouds: Gaussian peak at 45%, sigma 28 (broad) ---
        high_s = bell_curve(high_pct, peak=45.0, sigma=28.0) * 100.0

        # --- Mid clouds: peak at 20%, sigma 22, contributes up to 62 pts ---
        mid_s = bell_curve(mid_pct, peak=20.0, sigma=22.0) * 62.0

        # --- Low clouds: no penalty until 20%, ramps to full penalty at 85% ---
        if low_pct <= 20.0:
            low_penalty = 0.0
        else:
            low_penalty = clamp((low_pct - 20.0) / 65.0 * 100.0)

        # Upper-cloud offset: strong high + mid clouds partially absorb the
        # low-cloud penalty (models the case where upper drama dominates).
        upper_strength = clamp((high_pct + mid_pct * 0.5) / 60.0)
        effective_low_penalty = low_penalty * (1.0 - upper_strength * 0.35)

        # --- Overcast penalty: kicks in at 82%, saturates at 100% ---
        if total_pct <= 82.0:
            overcast_penalty = 0.0
        else:
            overcast_penalty = clamp((total_pct - 82.0) / 18.0 * 78.0)

        # Combine upper-layer colour potential, suppressed by effective low penalty
        base = (high_s * 0.60 + mid_s * 0.40) * (1.0 - effective_low_penalty / 175.0)
        base = clamp(base - overcast_penalty)

        # --- Clear-sky penalty: mild scaling from 0.62 at 0% to 1.0 at 15% ---
        # A cloudless sky produces pastel colours at most — good but not epic.
        if total_pct < 15.0:
            base *= 0.62 + 0.38 * (total_pct / 15.0)

        return clamp(base)

    # ------------------------------------------------------------------
    # Component 2: Atmosphere / Clarity (weight 0.28)
    # ------------------------------------------------------------------

    def atmosphere_score(
        self,
        visibility_m: float,
        aerosol_od: Optional[float],
        humidity_pct: float,
    ) -> float:
        """
        Score atmospheric clarity for sunset colour intensity.

        - High visibility = clean air = vivid colours
        - Moderate aerosol (AOD 0.1–0.3) scatters blue light and intensifies
          warm tones — the "pink hour" effect.  Too much (AOD > 0.6) creates
          milky haze that dulls colours.
        - Missing AOD falls back to a visibility-derived proxy with a gentler
          floor (40 pts) — missing data should not tank the score.
        - High humidity is a mild penalty only (above 75 %, max −18 pts).
        """
        # Visibility: 25 km = excellent; linear below that
        vis_score = clamp(visibility_m / 25_000.0 * 100.0)

        if aerosol_od is not None:
            # Peak at AOD ≈ 0.18 (light haze for warm tones)
            aer_score = bell_curve(aerosol_od, peak=0.18, sigma=0.15) * 100.0
        else:
            # Gentle proxy — never below 40 regardless of visibility
            aer_score = max(40.0, vis_score * 0.80)

        # Humidity: mild penalty above 75 % (max −18 pts at 100 %)
        hum_penalty = max(0.0, (humidity_pct - 75.0) / 25.0 * 18.0)

        combined = vis_score * 0.50 + aer_score * 0.50 - hum_penalty
        return clamp(combined)

    # ------------------------------------------------------------------
    # Component 3: Moisture / Precipitation (weight 0.20)
    # ------------------------------------------------------------------

    def moisture_score(
        self,
        precip_mm: float,
        humidity_pct: float,
        *,
        precip_last_3h: Optional[float] = None,
        pressure_trend: Optional[float] = None,
        cloud_trend: Optional[float] = None,
        vis_trend: Optional[float] = None,
    ) -> float:
        """
        Score moisture and precipitation conditions.

        Separate treatment for:
        - Active precipitation now  → strong penalty
        - Recent rain + current clearing → clearing bonus (post-rain glow)
        - Rising pressure / improving visibility / clearing clouds → bonus
        - High humidity without rain → mild penalty only

        Clearing bonus: up to +15 pts when rain stopped recently and
        atmospheric signals show improvement.
        """
        # Active rain: 0 mm = 0 penalty; ~2 mm = ~90 penalty (near knockout)
        precip_penalty = clamp(precip_mm * 45.0)

        # Clearing bonus — only applicable when it is NOT currently raining
        clearing_bonus = 0.0
        if precip_mm < 0.1:
            if precip_last_3h is not None and precip_last_3h > 0.5:
                # Rain in recent hours, now dry → classic post-rain glow potential
                clearing_bonus += 8.0
            if pressure_trend is not None and pressure_trend > 1.0:
                # Rising pressure signals improving conditions
                clearing_bonus += 4.0
            if cloud_trend is not None and cloud_trend < -10.0:
                # Cloud cover decreasing → clearing
                clearing_bonus += 3.0
            if vis_trend is not None and vis_trend > 1_000.0:
                # Visibility improving
                clearing_bonus += 3.0
            clearing_bonus = min(clearing_bonus, 15.0)

        # Humidity: penalty only above 85 % (max −25 pts at 100 %)
        hum_penalty = max(0.0, (humidity_pct - 85.0) / 15.0 * 25.0)

        return clamp(100.0 - precip_penalty - hum_penalty + clearing_bonus)

    # ------------------------------------------------------------------
    # Component 4: Horizon (weight 0.10)
    # ------------------------------------------------------------------

    def horizon_score(self, obstruction_deg: float) -> float:
        """
        Score the unobstructed horizon.

        0 degrees = open ocean / flat field = 100.
        5 degrees = gentle hills / low suburbs = ~70.
        15+ degrees = dense urban / deep valley = ~15.

        Uses a softened power curve (exponent 1.2, coefficient 3.8) so that
        typical urban/suburban locations are not over-penalised.
        """
        return clamp(100.0 - (obstruction_deg ** 1.2) * 3.8)

    # ------------------------------------------------------------------
    # Category mapping
    # ------------------------------------------------------------------

    @staticmethod
    def score_to_category(score: float) -> str:
        """Map a 0–100 beauty score to a descriptive category."""
        for threshold, label in SCORE_THRESHOLDS:
            if score >= threshold:
                return label
        return "Poor"

    # ------------------------------------------------------------------
    # Confidence estimation
    # ------------------------------------------------------------------

    def compute_confidence(
        self,
        weather: WeatherSnapshot,
        component_scores: dict[str, float],
        physics_score: float,
        has_ml: bool = False,
        window_scores: Optional[list[float]] = None,
    ) -> float:
        """
        Estimate prediction confidence in [15, 92].

        Confidence is higher when:
        - The score is far from the ambiguous middle (40–60)
        - Multiple window points agree
        - Aerosol data is real (not proxy-estimated)

        Confidence is lower when:
        - Signals conflict (good clouds + active rain)
        - The window is highly volatile (one great point, rest collapse)
        - Aerosol is estimated
        - Active rain conflicts with otherwise strong sky structure
        """
        base = 60.0

        # Boost for score extremity (far from 50 = clearer prediction)
        extremity = abs(physics_score - 50.0) / 50.0  # 0..1
        base += extremity * 18.0

        # Window consistency boost / penalty
        if window_scores and len(window_scores) > 1:
            spread = max(window_scores) - min(window_scores)
            if spread < 15.0:
                base += 8.0   # all window points agree
            elif spread > 35.0:
                base -= 10.0  # highly volatile — hard to predict

        # Penalty for estimated aerosol
        if weather.aerosol_is_estimated:
            base -= 6.0

        # Conflicting signals: nice clouds but it's raining
        if weather.precipitation_mm > 0.5 and component_scores.get("cloud_quality", 0) > 55:
            base -= 12.0

        # Active rain with otherwise strong final score = extra uncertainty
        if weather.precipitation_mm > 1.0 and physics_score > 50.0:
            base -= 8.0

        # Near-zero cloud ambiguity (clear sky is decent but unpredictable)
        if weather.cloud_total < 5.0 and weather.cloud_high < 5.0:
            base -= 5.0

        if has_ml:
            base += 4.0

        return clamp(base, lo=15.0, hi=92.0)
