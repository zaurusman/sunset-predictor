"""
Physics-informed sunset beauty scoring engine.

DESIGN PRINCIPLE
----------------
The score represents **daily sunset quality** for a given location — i.e., how
good the sunset conditions are expected to be on that day. The weather is
sampled at the sunset hour (the most relevant atmospheric window), but there is
NO time-of-day penalty. The score is not about whether you are standing outside
at the exact right moment; it answers "will today's sunset be beautiful?"

COMPONENTS (4 total, weights configurable)
------------------------------------------
1. Cloud Quality  (40 %) — cloud distribution at sunset
2. Atmosphere     (30 %) — visibility, aerosol, humidity
3. Moisture       (20 %) — rain and high-humidity penalties
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
    "cloud_quality": 0.40,
    "atmosphere": 0.30,
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


class ScoringEngine:
    """
    Computes the physics-informed sunset beauty score from a WeatherSnapshot.

    All scoring methods are deterministic and unit-testable in isolation.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self._weights = weights or dict(DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Top-level entry point
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
        mst = self.moisture_score(weather.precipitation_mm, weather.relative_humidity)
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
    # Component 1: Cloud Quality (weight 0.40)
    # ------------------------------------------------------------------

    def cloud_quality_score(
        self,
        low_pct: float,
        mid_pct: float,
        high_pct: float,
        total_pct: float,
    ) -> float:
        """
        Score the cloud distribution for sunset color potential.

        High clouds (cirrus, altocumulus) scatter and reflect sunlight with vivid
        colour.  Mid-level clouds (altostratus) add texture.  Low clouds near
        the horizon block the sun and reduce colour.  A totally overcast sky
        diffuses light too much.

        Best condition: high clouds 30–60%, mid clouds 10–40%, low clouds < 15%.
        Worst condition: total overcast > 85% or heavy low cloud > 50%.
        """
        # High clouds: Gaussian peak at 45% coverage, broad width
        high_s = bell_curve(high_pct, peak=45.0, sigma=25.0) * 100.0

        # Mid clouds: Gaussian peak at 25%, narrower
        mid_s = bell_curve(mid_pct, peak=25.0, sigma=20.0) * 70.0

        # Low clouds: reward near zero; penalise above 15%
        if low_pct <= 15.0:
            low_penalty = 0.0
        else:
            # Scales from 0 at 15% to 100 at 80%
            low_penalty = clamp((low_pct - 15.0) / 65.0 * 100.0)

        # Overcast penalty: kicks in above 75%, maxes out at 100%
        if total_pct <= 75.0:
            overcast_penalty = 0.0
        else:
            overcast_penalty = clamp((total_pct - 75.0) / 25.0 * 80.0)

        # Combine: high + mid provide colour potential, low clouds suppress it,
        # total overcast kills it.
        base = (high_s * 0.55 + mid_s * 0.45) * (1.0 - low_penalty / 150.0)
        base = clamp(base - overcast_penalty)

        # Clear-sky penalty: nearly empty sky means nothing to catch and scatter
        # sunlight.  Scale linearly from 0 at 0 % to no penalty at 20 %.
        if total_pct < 20.0:
            base *= total_pct / 20.0

        return clamp(base)

    # ------------------------------------------------------------------
    # Component 2: Atmosphere / Clarity (weight 0.30)
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
          warm tones — the "pink hour" effect.  Too much aerosol (AOD > 0.6)
          creates milky haze and dulls colours.
        - High humidity alone is a mild negative due to moisture diffusion.
        """
        # Visibility: 20 km+ is excellent; below 5 km poor
        vis_score = clamp(visibility_m / 20_000.0 * 100.0)

        # Aerosol optical depth
        if aerosol_od is not None:
            # Peak at AOD ≈ 0.18 (light haze for warm tones); penalise extremes
            aer_score = bell_curve(aerosol_od, peak=0.18, sigma=0.14) * 100.0
        else:
            # Proxy: estimate from visibility score
            aer_score = max(30.0, vis_score * 0.75)

        # Humidity: mild penalty above 70%
        hum_penalty = max(0.0, (humidity_pct - 70.0) / 30.0 * 20.0)

        combined = vis_score * 0.50 + aer_score * 0.50 - hum_penalty
        return clamp(combined)

    # ------------------------------------------------------------------
    # Component 3: Moisture / Precipitation (weight 0.20)
    # ------------------------------------------------------------------

    def moisture_score(self, precip_mm: float, humidity_pct: float) -> float:
        """
        Score moisture and precipitation conditions.

        Rain near sunset washes out colours and blocks light dramatically.
        Very high humidity alone causes light diffusion.
        """
        # Rain penalty: 0 mm = no penalty; 2 mm = heavy penalty; 5 mm = near 100
        precip_penalty = clamp(precip_mm * 50.0)

        # Humidity: only penalise above 80%
        hum_penalty = max(0.0, (humidity_pct - 80.0) / 20.0 * 30.0)

        return clamp(100.0 - precip_penalty - hum_penalty)

    # ------------------------------------------------------------------
    # Component 4: Horizon (weight 0.10)
    # ------------------------------------------------------------------

    def horizon_score(self, obstruction_deg: float) -> float:
        """
        Score the unobstructed horizon.

        0 degrees = open ocean / flat field = 100.
        5 degrees = gentle hills = ~65.
        15+ degrees = dense urban / deep mountain valley = ~10.

        Uses a power curve to model realistic visual impact.
        """
        # Power curve: harsher than linear to penalise significant obstruction
        return clamp(100.0 - (obstruction_deg ** 1.3) * 4.0)

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
    ) -> float:
        """
        Estimate prediction confidence in [15, 92].

        Confidence is higher when:
        - All signals clearly agree (all high or all low)
        - Aerosol data is from the actual API (not estimated)
        - The final score is far from the 50-point ambiguous middle

        Confidence is lower when:
        - Signals conflict (good clouds but rain)
        - Aerosol is estimated from proxy
        - Score sits in the ambiguous 40–60 range
        """
        base = 60.0

        # Boost for score extremity (far from 50 = clearer prediction)
        extremity = abs(physics_score - 50.0) / 50.0  # 0..1
        base += extremity * 18.0

        # Penalty for estimated aerosol
        if weather.aerosol_is_estimated:
            base -= 8.0

        # Penalty for conflicting signals: good cloud + rain
        if weather.precipitation_mm > 0.5 and component_scores.get("cloud_quality", 0) > 55:
            base -= 12.0

        # Penalty for missing or trace values creating ambiguity
        if weather.cloud_total < 5 and weather.cloud_high < 5:
            # Virtually no clouds — clear sky, which is decent but not epic
            base -= 5.0

        # Small boost if ML calibration is also running
        if has_ml:
            base += 4.0

        return clamp(base, lo=15.0, hi=92.0)
