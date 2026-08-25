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
from app.utils.geo import horizon_tangent_distance_km
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

# Score at which we recommend going outside.
# This is the bar for "worth changing your plans for", not "better than
# average" — most evenings land in the 40s and 50s, so a 45 recommended
# going outside on a thoroughly ordinary sky.
GO_OUTSIDE_THRESHOLD = 70.0

# ---------------------------------------------------------------------------
# Light-corridor constants
# ---------------------------------------------------------------------------

# Representative altitudes for the three cloud layers Open-Meteo reports.
# WMO puts high cloud at 5–13 km, middle at 2–7 km, low below 2 km; these are
# mid-band values used to derive each layer's illumination tangent distance.
LAYER_HEIGHT_KM: dict[str, float] = {
    "low": 1.0,    # → light grazes the surface ~113 km upstream
    "mid": 4.0,    # → ~226 km
    "high": 9.0,   # → ~339 km
}

# Floor on the corridor multiplier. A fully blocked corridor still leaves some
# diffuse skylight, so it dims the colour score rather than zeroing it.
CORRIDOR_FLOOR = 0.25


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
    # Afterglow potential at this snapshot's sun elevation.
    # Non-zero only when sun < 0° and high clouds are present.
    # Stored here for transparency / explanation; already baked into cloud_quality.
    afterglow: float = 0.0
    # Upstream illumination multiplier already applied to cloud_quality.
    # None when no corridor data was available.
    light_corridor: Optional[float] = None

    def to_physics_breakdown(self) -> PhysicsBreakdown:
        return PhysicsBreakdown(
            cloud_quality_score=round(self.cloud_quality, 1),
            atmosphere_score=round(self.atmosphere, 1),
            moisture_score=round(self.moisture, 1),
            horizon_score=round(self.horizon, 1),
            weighted_physics_score=round(self.physics_score, 1),
            component_weights=self.weights,
            afterglow_score=round(self.afterglow, 1) if self.afterglow > 0 else None,
            light_corridor_factor=(
                round(self.light_corridor, 3) if self.light_corridor is not None else None
            ),
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
        self,
        weather: WeatherSnapshot,
        horizon_obstruction_deg: float,
        corridor_samples: Optional[list[tuple[float, float, float]]] = None,
    ) -> ScoringResult:
        """
        Compute the full scoring breakdown for *weather* at *horizon_obstruction_deg*.

        *corridor_samples* are ``(distance_km, cloud_low_pct, cloud_mid_pct)``
        readings taken upstream along the sunset azimuth. When supplied, they
        scale the colour score by how much light actually reaches the clouds
        overhead (see light_corridor_factor). When omitted the score is
        unadjusted, so callers without corridor data behave exactly as before.

        Returns a ScoringResult containing per-component scores, the weighted
        physics score, and a confidence estimate.
        """
        sun_elev = weather.sun_elevation_deg

        cq = self.cloud_quality_score(
            weather.cloud_low,
            weather.cloud_mid,
            weather.cloud_high,
            weather.cloud_total,
            sun_elevation_deg=sun_elev,
        )

        # Illumination gate: a beautiful canvas that no light reaches is grey.
        corridor: Optional[float] = None
        if corridor_samples:
            corridor = self.light_corridor_factor(
                corridor_samples,
                cloud_low=weather.cloud_low,
                cloud_mid=weather.cloud_mid,
                cloud_high=weather.cloud_high,
            )
            cq = clamp(cq * corridor)

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

        # Afterglow potential — computed for breakdown / explanation only.
        # The effect is already embedded in cq (cloud_quality_score with sun_elev);
        # we do NOT add ag to physics_score again.
        ag = self.afterglow_score(
            sun_elevation_deg=sun_elev,
            cloud_high=weather.cloud_high,
            cloud_low=weather.cloud_low,
            cloud_total=weather.cloud_total,
            atmosphere=atm,
        )

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
            afterglow=ag,
            light_corridor=corridor,
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

        # Consistency bonus: fraction of points ≥ 50, scaled to +3 pts max
        # (was +5 — reduced because inflated atmosphere made it fire too easily)
        good_count = sum(1 for v in vals if v >= 50.0)
        consistency_bonus = (good_count / len(vals)) * 3.0 if len(vals) > 1 else 0.0

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
        sun_elevation_deg: float = 0.0,
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

        Afterglow physics (sun_elevation_deg < 0)
        -----------------------------------------
        When the sun drops below the horizon it still illuminates high clouds
        from below via Rayleigh-scattered light along the limb.  This produces
        the deepest reds and crimsons, and peaks around −3°.  Three adjustments
        are made in this regime:

        1. High cloud score gets a conditional boost (up to +28 pts).
           The boost scales with:
           - an elevation bell curve peaking at −3° (sigma 2°)
           - the fraction of high cloud coverage (canvas size)
           - low-cloud interference (heavy low cloud blocks the illuminated layer)
           No boost when high_pct < 15 % (nothing to illuminate) or when
           the sky is overcast (total_pct ≥ 82 %).

        2. The low-cloud penalty is softened by up to 25 % in afterglow phase.
           When sunlight arrives from below the horizon, a thin low-cloud layer
           is less effective at blocking the illuminated high-cloud canvas.

        Horizon glow (near-clear sky, sun near horizon)
        ------------------------------------------------
        When the sky is mostly clear (< 20 % total cloud) and the sun is within
        a few degrees of the horizon, the atmospheric path length is long enough
        to scatter away most blue light, producing vivid orange/red tones without
        any clouds.  This is additive with cloud-based colour: a single cloud
        catching the light in an otherwise clear sky gets both contributions.
        Peaks at sun ≈ 0°, fades above +6° and below −4°, max +15 pts.

        Key calibration points
        ----------------------
        - High 45%, low < 20%                →  peak colour potential (~85–90)
        - High 45%, low < 20%, sun at −3°    →  afterglow peak (~100)
        - Full overcast (≥90%)               →  heavily penalised (<15)
        - Completely clear sky, sun at 0°    →  horizon glow lifts to ~35–40
        - High high + heavy low, sun at −3°  →  low interference blocks afterglow
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

        # --- Blocking clouds: the overcast criterion must be TYPE-AWARE ---
        # Low stratus fully blocks the sun/light path; mid clouds partially block;
        # high cirrus barely blocks at all — it diffuses rather than occludes.
        # A 97 % cirrus sky is visually open and is excellent for afterglow;
        # a 97 % stratus sky is the opposite.  Using total cloud cover to trigger
        # the overcast penalty treats these identically, which is wrong.
        blocking_clouds = low_pct + mid_pct * 0.6  # high clouds have weight 0

        # --- Overcast penalty: driven by BLOCKING (low+mid) clouds, not total ---
        # Kicks in at blocking > 70, saturates at blocking = 100.
        if blocking_clouds <= 70.0:
            overcast_penalty = 0.0
        else:
            overcast_penalty = clamp((blocking_clouds - 70.0) / 30.0 * 78.0)

        # --- Cirrus-sheet floor: when sky is nearly blocking-cloud-free ---
        # A full cirrus layer (high ≈ 100 %, low ≈ 0 %) still provides a
        # colour canvas and should not score as poorly as the bell-curve tail
        # at 97 % suggests (~18 pts).  Set a floor of high_pct / 100 × 45 pts
        # so that a dense cirrus deck earns at least 45 pts as a base canvas.
        if blocking_clouds < 30.0:
            cirrus_floor = clamp(high_pct / 100.0 * 45.0)
            high_s = max(high_s, cirrus_floor)

        # Combine upper-layer colour potential, suppressed by effective low penalty
        base = (high_s * 0.60 + mid_s * 0.40) * (1.0 - effective_low_penalty / 175.0)
        base = clamp(base - overcast_penalty)

        # --- Clear-sky penalty: mild scaling from 0.62 at 0% to 1.0 at 15% ---
        # A cloudless sky produces pastel colours at most — good but not epic.
        if total_pct < 15.0:
            base *= 0.62 + 0.38 * (total_pct / 15.0)

        # --- Afterglow enhancement (sun below horizon only) ---
        # Conditions: sun < 0°, meaningful high clouds present, sky not
        # blocked from below (blocking_clouds < 70).  We use blocking_clouds
        # here (not total_pct) so that a pure cirrus overcast — which is an
        # ideal afterglow canvas — still receives the boost.
        if (
            sun_elevation_deg < 0.0
            and high_pct >= 15.0
            and blocking_clouds < 70.0
        ):
            # Bell curve peaked at −3° (sigma 2°): models limb-illumination intensity
            elev_factor = math.exp(
                -0.5 * ((sun_elevation_deg + 3.0) / 2.0) ** 2
            )
            # Canvas factor: more high cloud = more surface area to be lit
            canvas_factor = clamp(high_pct / 55.0)
            # Low cloud interference: thick low cloud screens the illuminated layer
            low_factor = max(0.0, 1.0 - low_pct / 60.0)
            # Maximum +28 pts — ensures afterglow cannot manufacture an Epic
            # score from a mediocre base; it can lift a Good to Great/Epic.
            afterglow_boost = elev_factor * canvas_factor * low_factor * 28.0
            base = clamp(base + afterglow_boost)

        # --- Horizon glow: Rayleigh orange for near-clear skies at low sun angles ---
        # Only fires when total cloud < 20 % — if clouds are present they already
        # score their own colour; this covers the clear-sky pathway.
        # Bell curve peaks at ~0.5° (sun just kissing the horizon), sigma 2.5°.
        if total_pct < 20.0 and -4.0 <= sun_elevation_deg <= 6.0:
            elev_factor = math.exp(-0.5 * ((sun_elevation_deg - 0.5) / 2.5) ** 2)
            clearness = clamp(1.0 - total_pct / 20.0)
            horizon_glow = elev_factor * clearness * 15.0
            base = clamp(base + horizon_glow)

        return clamp(base)

    # ------------------------------------------------------------------
    # Light corridor — the upstream illumination path
    # ------------------------------------------------------------------

    @staticmethod
    def corridor_transmittance(
        samples: list[tuple[float, float, float]], layer_height_km: float
    ) -> float:
        """Fraction of sunset light reaching a cloud layer at *layer_height_km*.

        *samples* are ``(distance_km, cloud_low_pct, cloud_mid_pct)`` readings
        taken along the sunset azimuth, upstream of the observer.

        WHY THIS EXISTS
        ---------------
        Everything else in this engine looks only at the observer's own grid
        cell, which cannot see the single most important thing about a sunset:
        whether light can *reach* the clouds overhead. A solid deck 200 km
        toward the sunset kills the display no matter how good your local sky
        looks. Corfidi (NOAA SPC) describes the best sunsets as a mid/high deck
        covering everything "except a narrow clear strip near the horizon" —
        that strip is what this measures.

        GEOMETRY
        --------
        Light illuminating a cloud at height h grazes the surface roughly
        ``sqrt(2·R·h)`` away along the sun's azimuth (~113 km for 1 km cloud,
        ~226 km for 4 km, ~339 km for 9 km). Nearer than that the ray is
        already above the boundary layer; further out it hasn't descended into
        it yet. So samples are weighted by a broad Gaussian centred on that
        tangent distance — broad because forecast grids are coarse and the ray
        traverses a range of distances, not a point.

        Only low and mid cloud block: high cirrus diffuses light rather than
        occluding it, which is why it is excluded from the blocking sum.

        Returns 1.0 (fully transmitting) when there are no samples, so every
        caller degrades gracefully to the previous behaviour.
        """
        if not samples:
            return 1.0

        tangent = horizon_tangent_distance_km(layer_height_km)
        if tangent <= 0.0:
            return 1.0
        sigma = max(tangent * 0.45, 40.0)

        total_w = 0.0
        total_blocking = 0.0
        for distance_km, low_pct, mid_pct in samples:
            w = math.exp(-0.5 * ((distance_km - tangent) / sigma) ** 2)
            # Mid cloud only partially occludes; high cloud not at all.
            blocking = clamp(low_pct + mid_pct * 0.5) / 100.0
            total_w += w
            total_blocking += w * blocking

        if total_w == 0.0:
            return 1.0
        return clamp(1.0 - total_blocking / total_w, lo=0.0, hi=1.0)

    def light_corridor_factor(
        self,
        samples: list[tuple[float, float, float]],
        cloud_low: float,
        cloud_mid: float,
        cloud_high: float,
    ) -> float:
        """Multiplier in [CORRIDOR_FLOOR, 1.0] applied to the colour score.

        This is a *multiplier*, not another weighted component, because that is
        the physics: colour = canvas × illumination. An unlit canvas is grey no
        matter how well-shaped it is, so no amount of good local cloud should
        be able to compensate for a blocked corridor.

        Each layer is illuminated through its own tangent distance, so the
        layers present overhead determine which part of the corridor matters.
        A sky of pure cirrus cares about conditions ~340 km out; a low-cloud
        sky cares about ~110 km out.

        The floor keeps a blocked corridor from zeroing the score outright —
        diffuse skylight still tints an overcast evening slightly.
        """
        if not samples:
            return 1.0

        layers = (
            (cloud_low, LAYER_HEIGHT_KM["low"]),
            (cloud_mid, LAYER_HEIGHT_KM["mid"]),
            (cloud_high, LAYER_HEIGHT_KM["high"]),
        )
        total_cover = sum(cover for cover, _ in layers)

        if total_cover < 5.0:
            # Effectively clear overhead: the colour is horizon glow, which
            # arrives along the longest slant path, so it is the far field that
            # decides whether there is anything to see.
            transmittance = self.corridor_transmittance(samples, LAYER_HEIGHT_KM["high"])
        else:
            transmittance = sum(
                cover * self.corridor_transmittance(samples, h) for cover, h in layers
            ) / total_cover

        return CORRIDOR_FLOOR + (1.0 - CORRIDOR_FLOOR) * transmittance

    # ------------------------------------------------------------------
    # Afterglow potential (standalone, for breakdown and explanation)
    # ------------------------------------------------------------------

    def afterglow_score(
        self,
        sun_elevation_deg: float,
        cloud_high: float,
        cloud_low: float,
        cloud_total: float,
        atmosphere: float = 70.0,
    ) -> float:
        """
        Standalone afterglow potential score (0–100).

        This mirrors the afterglow logic inside cloud_quality_score() but is
        expressed as an independent 0–100 signal for use by the explanation
        engine and the API breakdown.  It is NOT added to the physics score
        separately — that would double-count what is already in cloud_quality.

        Afterglow is driven by limb illumination of high clouds:
        - Peaks at sun_elevation_deg ≈ −3° (civil twilight, clouds still lit)
        - Requires high_pct ≥ 15 % (canvas to be illuminated)
        - Blocked by overcast (total ≥ 85 %) or heavy low cloud (low ≥ 60 %)
        - Atmosphere quality amplifies colour saturation (clear = more vivid)

        Returns 0 when sun is at or above the horizon.
        """
        if sun_elevation_deg >= 0.0:
            return 0.0
        if cloud_high < 15.0:
            return 0.0  # No canvas — limb light has nothing to paint on
        # Block by low+mid overcast only — a pure cirrus overcast (high=97, low=0)
        # is an ideal afterglow canvas, not a blocking layer.
        blocking = cloud_low + (cloud_total - cloud_high) * 0.6
        if blocking >= 70.0:
            return 0.0

        # Elevation bell curve: peak at −3°, FWHM ≈ 4.7° (sigma 2°)
        elev_factor = math.exp(
            -0.5 * ((sun_elevation_deg + 3.0) / 2.0) ** 2
        )

        # High cloud canvas: more coverage → more surface for limb illumination
        canvas = clamp(cloud_high / 55.0)

        # Low cloud interference: > 60 % low cloud screens the canvas entirely
        low_interference = max(0.0, 1.0 - cloud_low / 60.0)

        # Atmosphere quality multiplier: clean air lets warm tones saturate fully
        # (0.6 floor so hazy air still gets some score, not zero)
        atm_factor = 0.6 + 0.4 * (atmosphere / 100.0)

        return clamp(elev_factor * canvas * low_interference * atm_factor * 100.0)

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
            # Estimated AOD: no artificial floor — tie it directly to visibility
            # so that the default 15 km archive baseline produces a neutral score,
            # not a phantom-high one.  Previously max(40, vis*0.80) could give ~77
            # even with mediocre visibility; now it scales proportionally.
            aer_score = vis_score * 0.75

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
        lead_time_hours: Optional[float] = None,
    ) -> float:
        """
        Estimate prediction confidence in [15, 92].

        Confidence is higher when:
        - The score is far from the ambiguous middle (40–60)
        - Multiple window points agree
        - Aerosol data is real (not proxy-estimated)
        - The sunset is imminent (forecast refreshes hourly and firms up)

        Confidence is lower when:
        - Signals conflict (good clouds + active rain)
        - The window is highly volatile (one great point, rest collapse)
        - Aerosol is estimated
        - Active rain conflicts with otherwise strong sky structure
        - The target sunset is many days out (forecast skill decays with lead time)

        *lead_time_hours* is the gap from now to the target sunset; pass None
        (the default) to skip the lead-time term entirely (e.g. for overrides).
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

        # Forecast lead-time: imminent sunsets (and observed past dates) get a
        # small boost; ~1 day out is neutral; each further day is penalised,
        # reflecting how forecast skill decays with lead time.
        if lead_time_hours is not None:
            base += self._lead_time_adjustment(lead_time_hours)

        return clamp(base, lo=15.0, hi=92.0)

    @staticmethod
    def _lead_time_adjustment(lead_time_hours: float) -> float:
        """Confidence adjustment (points) for how far the sunset is from now.

        +MAX_BONUS at/after the sunset moment (and for observed past dates),
        tapering to 0 at ~24h out, then -PER_DAY_PENALTY for every extra day.
        """
        MAX_BONUS = 5.0
        PER_DAY_PENALTY = 2.5
        if lead_time_hours <= 0.0:
            return MAX_BONUS
        if lead_time_hours < 24.0:
            return MAX_BONUS * (1.0 - lead_time_hours / 24.0)
        days_beyond = (lead_time_hours - 24.0) / 24.0
        return -PER_DAY_PENALTY * days_beyond
