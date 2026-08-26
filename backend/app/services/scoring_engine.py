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

SCORED COMPONENTS (weighted average, weights configurable)
----------------------------------------------------------
1. Colour potential (60 %) — the best available PATHWAY to beauty (below),
                             gated by upstream illumination
2. Atmosphere       (25 %) — how clean the air is: aerosol, plus visibility
                             where the source reports it
3. Moisture         (15 %) — the atmospheric water column, plus post-rain clearing

PATHWAYS — WHY THE COLOUR TERM IS A SET, NOT A FORMULA
------------------------------------------------------
A sunset can be beautiful in several unrelated ways, and they do not share
ingredients. Cloud on fire needs a mid/high deck and a clear strip to the west.
A clear-sky gradient needs the opposite — no cloud at all — plus clean dry air,
and it peaks after the sun has gone. A breaking storm needs rain that has just
stopped. A silhouette band needs a heavy deck that every other pathway reads as
a ruined evening.

Any single formula over these inputs has to pick a favourite. The original
engine picked lit cloud, and so a cloudless Tel Aviv evening that was genuinely
lovely scored 11/100 with the explanation "clear conditions produce less colour
drama". Adding terms for the other cases would not fix it: an average asks
every evening to be good in every way at once, which no real sunset is.

So each route is scored INDEPENDENTLY, by its own physics, with its own
preconditions, and the best one sets the score (see combine_pathways). Nothing
is optimised toward a preferred combination of parameters, and adding a new
kind of beautiful sunset means adding a function, not re-tuning the others.

The engine also reports WHICH pathway won, because the user needs that as much
as the number: it decides what to look for and when to be outside.

GATES (multiplicative, applied after the average)
-------------------------------------------------
- Light corridor   — is the upstream light path clear? (folded into cloud quality)
- Precipitation    — active rain ends a sunset; it does not "reduce" it
- Horizon          — permanent obstruction at the observer

WHY GATES INSTEAD OF WEIGHTS
----------------------------
Measured over a year at three cities, horizon had a standard deviation of
ZERO across days while holding 10 % of the weight, and precipitation was
absent on 82-93 % of evenings while its component held 20 %. As weighted
addends they did not discriminate between days — they simply added a large
near-constant to every score, which is a major reason "Epic" was firing on
10-15 % of evenings.

Physically they are gates, not ingredients: no light path means no colour,
however good the sky overhead is. Multiplying expresses that and stops them
inflating the baseline. See docs/scoring-v2-plan.md (D1, D2).

Each component returns a score in [0, 100].  The final beauty score is a
weighted average scaled by the gates, clamped to [0, 100].
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

# Horizon and precipitation are deliberately ABSENT — they are gates applied
# multiplicatively after the average, not weighted addends. See the module
# docstring and docs/scoring-v2-plan.md (D1).
DEFAULT_WEIGHTS: dict[str, float] = {
    "cloud_quality": 0.60,
    "atmosphere": 0.25,
    "moisture": 0.15,
}

# Floors on the two multiplicative gates. Neither can zero a score outright:
# an obstructed horizon still shows you the upper sky, and even in rain there
# is a little colour when the cloud breaks.
HORIZON_FLOOR = 0.35
PRECIP_FLOOR = 0.15

# ---------------------------------------------------------------------------
# Score → category thresholds
# ---------------------------------------------------------------------------

# ABSOLUTE thresholds on the raw physics score, which is what the app displays.
#
# Set from the pooled raw distribution across Tel Aviv, London and San
# Francisco (365 days each, scripts/evaluate.py) so that the bands mean the
# same THING everywhere — but deliberately NOT the same frequency everywhere.
# A location with genuinely better sunset conditions should earn more good
# evenings; that is the point of an absolute scale, and the per-location
# frequency is reported separately as a rank ("better than 31 % of evenings
# here").
#
# Re-derive these whenever the raw scale moves, alongside REFERENCE_QUANTILES.
SCORE_THRESHOLDS: list[tuple[float, str]] = [
    (85, "Epic"),
    (72, "Great"),
    (55, "Good"),
    (38, "Decent"),
    (0, "Poor"),
]

# ---------------------------------------------------------------------------
# Percentile → rank score (NOT the display path — kept for analysis)
# ---------------------------------------------------------------------------
#
# The displayed 0-100 used to be a RANK against the location's own history.
# That was reversed: percentile display is SELF-NORMALISING, so a fix that makes
# the engine more right about a whole kind of evening lifts every evening of
# that kind and moves none of their ranks. Measured — the near-field horizon fix
# took one evening's raw score 48.9 → 57.4 while its displayed score went
# 30.9 → 30.6. A number that cannot show the effect of an improvement is the
# wrong number to tune against. See docs/scoring-v2-plan.md, "Why percentile
# display was reversed".
#
# The percentile now appears beneath the score as SEASONAL context ("better than
# 51 % of August evenings here"), and the anchors below survive only because
# scripts/evaluate.py uses them to report what a purely rank-based scale would
# have shown. They still document the intended shape of a rarity scale:
#
#     Poor    bottom 30 %   "don't bother"
#     Decent  next   38 %   "ordinary evening"
#     Good    next   20 %   "nice if you're out"
#     Great   next    9 %   "worth stepping outside for"
#     Epic    top     3 %   "roughly ten nights a year"
CALIBRATION_ANCHORS: list[tuple[float, float]] = [
    # (cumulative percentile, displayed score at that percentile)
    (0.00,   0.0),
    (0.30,  30.0),   # Poor   → Decent
    (0.68,  50.0),   # Decent → Good
    (0.88,  65.0),   # Good   → Great
    (0.97,  80.0),   # Great  → Epic
    (1.00, 100.0),
]

# Score at which we recommend going outside — on the ABSOLUTE scale, so this
# fires more often in a place with better skies. That is intended: if Tel Aviv
# genuinely has a lovely sunset most summer evenings, an honest app says so
# rather than rationing the recommendation to a fixed share of nights.
GO_OUTSIDE_THRESHOLD = 75.0

# Ensemble cloud-cover spread (standard deviation, %, across forecast members
# at the sunset hour) -> confidence adjustment (points). Replaces the
# lead-time guess with an actual measurement of forecast uncertainty when one
# is available (see ScoringEngine.compute_confidence,
# WeatherService.get_ensemble_cloud_spread).
#
# Anchors calibrated against live icon_seamless data over ~7.5 days in Tel
# Aviv, London and San Francisco: spread p25-p75 ran roughly 11-31, min ~0,
# max ~45. 0 = every member agrees exactly; 45+ is members disagreeing by
# nearly half the 0-100 scale.
ENSEMBLE_SPREAD_CONFIDENCE_ANCHORS: list[tuple[float, float]] = [
    (0.0,   10.0),
    (10.0,   5.0),
    (20.0,   0.0),
    (30.0,  -6.0),
    (45.0, -12.0),
]

# ---------------------------------------------------------------------------
# Atmosphere response curves
# ---------------------------------------------------------------------------

# Aerosol optical depth (550 nm) → clarity score. Monotone DECREASING: clean
# air is the ingredient, haze is the spoiler. See ScoringEngine.aerosol_clarity
# for why this is not a bell curve.
#
#   ≤0.05  pristine — post-frontal, maritime, high desert
#    0.15  typical continental background
#    0.30  visible haze
#    0.50  poor air-quality day
#    0.80  heavy smoke / dust event
AEROSOL_ANCHORS: list[tuple[float, float]] = [
    (0.00, 100.0),
    (0.05, 100.0),
    (0.15,  88.0),
    (0.30,  68.0),
    (0.50,  42.0),
    (0.80,  15.0),
    (1.50,   5.0),
]

# Total column water vapour (kg/m² ≈ mm precipitable water) → dryness score.
# Reference points: 5-10 mm is a dry continental winter airmass, 15-25 mm is
# an ordinary temperate evening, 40+ mm is tropical or a summer Mediterranean
# heat load.
TCWV_ANCHORS: list[tuple[float, float]] = [
    ( 0.0, 100.0),
    ( 8.0, 100.0),
    (15.0,  85.0),
    (22.0,  68.0),
    (30.0,  48.0),
    (40.0,  25.0),
    (55.0,   8.0),
    (80.0,   0.0),
]

# ---------------------------------------------------------------------------
# Clear-sky twilight gradient
# ---------------------------------------------------------------------------
#
# A cloudless sky is not a failed sunset. When the air is clean and dry, the
# western sky after sunset becomes a saturated vertical gradient — deep orange
# at the horizon through peach and salmon into blue and then indigo. That is
# the Earth's own shadow rising in the east with the Belt of Venus above it,
# and looking west it is the anti-twilight arch's counterpart: a long slant
# path through clean lower atmosphere, reddened by Rayleigh scattering with
# nothing to occlude it.
#
# This is a SEPARATE PATHWAY to colour, not a degraded version of the cloud
# pathway, so cloud_quality_score takes the better of the two rather than
# adding a small bonus to a penalised base.
#
# The old model had a "horizon glow" term worth at most +15 points that peaked
# at solar elevation +0.5 degrees and switched off entirely below -4. That is
# almost exactly wrong: the gradient is weak while the sun is still up and
# strongest well after it has set. It scored a real Tel Aviv evening (photo
# evidence, 2026-08-23) at 11/100 and explained it as "clear conditions produce
# less colour drama".

# Solar elevation at which the gradient is most saturated, and the spread
# around it. Negative = below the horizon. The band roughly -1 to -8 degrees is
# civil twilight, where the shadow edge is high enough to see but the sky has
# not yet gone dark.
TWILIGHT_PEAK_ELEV_DEG = -4.0
TWILIGHT_SIGMA_DEG = 3.0

# Best achievable clear-sky gradient, on the same 0-100 scale as the cloud
# pathway. Deliberately below the cloud maximum: a perfect lit-cloud sky is a
# rarer and more dramatic event than a clean twilight gradient, and the bands
# are percentile-anchored, so this sets how often a clear evening outranks a
# cloudy one rather than an absolute claim about beauty.
#
# First set to 95, which was too generous and measurably so: Tel Aviv's raw p50
# jumped to 80 with p90 at 87, because a clear evening is the DEFAULT there and
# every one of them became a top-decile evening. Percentile calibration hid it —
# the displayed bands stayed perfectly shaped while the raw scores being ranked
# were crowded into ten points, so the ordering inside them was noise.
TWILIGHT_MAX = 78.0

# The gradient is made of air, so its quality is entirely the air's quality.
# Blend and sharpen, because on a clear evening these are the ONLY things that
# vary — a gentle response would score every cloudless night the same and tell
# the user nothing.
TWILIGHT_CLARITY_SHARE = 0.55
TWILIGHT_SHARPNESS = 1.9

# ...but "dry is better" was the wrong sign for THIS pathway.
#
# Measured against 23 labelled clear-sky evenings in Tel Aviv, every input the
# air term actually used ranked the wrong way round against the human score:
#
#     surface RH        +0.61     <- not used at all
#     TCWV              +0.29
#     dryness  (TCWV)   -0.17     <- 45 % of `air`, positive weight
#     clarity           -0.17     <- 55 % of `air`, positive weight
#
# Both independent moisture measures say the same thing — more water in the
# air, better arch — and the engine inverted it. On the 15 of those evenings
# where the ERA5 archive independently reports zero cloud at every level,
# surface RH still ranks +0.49, and the sign survives every leave-one-out. So
# this is the air itself, not cloud that the cloud-cover fields under-reported.
#
# Why surface RH here, when column_dryness() argues at length that RH measures
# the bottom two metres and TCWV is the honest instrument? Because that
# argument is about the wrong pathway. Light reaching a lit CLOUD arrives from
# above and crosses the whole column, so TCWV is right for lit_cloud. The
# twilight arch is light that has travelled a long, near-tangential path
# through the boundary layer itself — which is exactly the layer surface RH
# measures. The two components are not competing; they belong to different
# pathways, and the labels agree.
#
# Physically: a dry evening on this coast is an easterly, desert-sourced one,
# and it gives a hard pale gradient. A humid evening is onshore marine flow,
# and the vapour is what reddens and softens the arch.
#
# The anchors below are humidity -> twilight vapour quality (0-100). The RISING
# limb is measured; the FALLING limb above ~80 % is a physical prior (moisture
# becomes haze, and fog makes no arch at all) — no labelled evening here
# exceeds 78 % RH, so nothing above that is fitted and it must not be read as
# though it were.
TWILIGHT_HUMIDITY_ANCHORS: list[tuple[float, float]] = [
    (30.0, 55.0),
    (50.0, 72.0),
    (65.0, 88.0),
    (78.0, 100.0),
    (88.0, 82.0),
    (96.0, 45.0),
]

# How much of the air blend the boundary-layer vapour term takes when humidity
# is available. It displaces the column-dryness half rather than adding a third
# term: on this pathway dryness measured -0.17, so keeping it at full weight
# would leave a known wrong-signed input in the product.
TWILIGHT_VAPOUR_SHARE = 0.45

# The horizon strip you actually look at is NOT your own grid cell.
#
# Standing on a beach watching the sunset, the bright band is sky roughly
# 30-150 km away — cloud directly overhead does almost nothing to it, and a
# deck 60 km west hides it completely. Scoring "is the horizon open?" from the
# observer's cell reads the sky above and behind the viewer instead.
#
# Caught on a real evening (Tel Aviv, 2026-08-23, photo): the local cell
# reported 48 % low cloud while the sample 60 km west reported 3 %, and the
# photo shows a clean horizon over open sea. The corridor sampling already
# fetches those upstream readings, so this costs nothing.
#
# Weighted toward the nearest samples and blended with a minority share of the
# local cell, because the gradient does extend up the sky, where overhead cloud
# dims it. Distinct from the corridor, which measures the FAR field (~340 km)
# and asks a different question: not "can you see the band" but "is the band
# lit at all".
NEAR_FIELD_MAX_KM = 200.0
NEAR_FIELD_SIGMA_KM = 90.0
NEAR_FIELD_LOCAL_SHARE = 0.25

# Local blocking cloud below this leaves the horizon band fully visible; above
# TWILIGHT_BLOCKED_AT the gradient is hidden. Gentler than the cloud pathway's
# low-cloud penalty because what this needs is a clear STRIP at the horizon,
# and cloud in the grid cell is just as likely to be behind the observer. The
# upstream corridor measures the direction that actually matters.
TWILIGHT_OPEN_BELOW = 30.0
TWILIGHT_BLOCKED_AT = 90.0

# ---------------------------------------------------------------------------
# Pathway ceilings and combination
# ---------------------------------------------------------------------------
#
# Each ceiling says how good that KIND of evening can be at its absolute best.
# They differ because the kinds differ in how striking and how rare they are,
# and because the bands are percentile-anchored: a ceiling controls how often
# one kind of evening outranks another, not an absolute claim about beauty.
#
# These are the most judgement-laden numbers in the engine. They are stated
# once, here, rather than being buried as magic constants inside five
# functions, so that disagreeing with them is a one-line change.
#
#   lit_cloud          100  a sky genuinely on fire is the ceiling
#   twilight_gradient   78  see the TWILIGHT_* section
#   breaking_storm      96  rarer than lit cloud and at least as spectacular,
#                           but it can fizzle as the gap closes
#   crepuscular         62  a lovely thing to catch, not a reason to travel;
#                           also the least trustworthy detection (see below)
#   horizon_band        58  striking when it works, fails more often than not
CREPUSCULAR_MAX = 62.0
BREAKING_STORM_MAX = 96.0
HORIZON_BAND_MAX = 58.0

# A horizon band needs BOTH a heavy deck overhead and an open corridor to the
# west. Neither alone is this kind of evening.
HORIZON_BAND_MIN_DECK = 55.0
HORIZON_BAND_MIN_CORRIDOR = 0.55

# Human-readable name for each pathway, used by the explanation engine and the
# UI. Kept beside the ceilings so a new pathway cannot be added without also
# deciding what to CALL it — an unnamed pathway can win an evening and leave
# the user with a number and no idea what to go outside and look for.
PATHWAY_LABELS: dict[str, str] = {
    "lit_cloud": "Lit clouds",
    "twilight_gradient": "Clear-sky gradient",
    "crepuscular": "Sun rays",
    "breaking_storm": "Breaking storm",
    "horizon_band": "Band under the cloud",
}

# How much a second (third…) active pathway can add on top of the best one.
# Small on purpose: this expresses "and there is more than one thing going on
# tonight", not "add up the ways it could be nice".
MULTI_PATHWAY_LIFT = 12.0

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
    # Multiplicative gates applied after the weighted average.
    precipitation_gate: float = 1.0
    horizon_gate: float = 1.0
    # Clear-sky pathway score at this snapshot. Reported so the UI and the
    # explanation engine can tell the user WHICH kind of sunset this is —
    # "lit clouds" and "colour gradient" want different words and a different
    # best-viewing time. Already folded into cloud_quality via max().
    twilight_gradient: float = 0.0
    # Every pathway's independent score, and which one is carrying the evening.
    pathways: dict[str, float] = field(default_factory=dict)

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
            precipitation_gate=round(self.precipitation_gate, 3),
            horizon_gate=round(self.horizon_gate, 3),
            twilight_gradient_score=round(self.twilight_gradient, 1),
            pathway_scores={k: round(v, 1) for k, v in self.pathways.items()},
            dominant_pathway=ScoringEngine.dominant_pathway(self.pathways),
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

        # Atmosphere is computed FIRST because the clear-sky pathway inside
        # cloud_quality_score depends on it: with no cloud to catch the light,
        # the gradient is made of nothing but air, so its saturation is a
        # direct function of how clean that air is.
        atm = self.atmosphere_score(
            weather.visibility_m,
            weather.aerosol_optical_depth,
        )

        mst = self.moisture_score(
            weather.precipitation_mm,
            weather.relative_humidity,
            tcwv=weather.tcwv_kg_m2,
            precip_last_3h=weather.precipitation_last_3h_mm,
            pressure_trend=weather.pressure_trend_hpa_3h,
            cloud_trend=weather.cloud_total_trend_3h,
            vis_trend=weather.visibility_trend_3h_m,
        )

        # The corridor is computed BEFORE the pathways, not after, because one
        # of them needs it as an input rather than as a scaling factor: a
        # horizon band is only distinguishable from a genuinely blocked sky by
        # the corridor being open while the sky overhead is not.
        corridor: Optional[float] = None
        if corridor_samples:
            corridor = self.light_corridor_factor(
                corridor_samples,
                cloud_low=weather.cloud_low,
                cloud_mid=weather.cloud_mid,
                cloud_high=weather.cloud_high,
            )

        pathways = self.pathway_scores(
            low_pct=weather.cloud_low,
            mid_pct=weather.cloud_mid,
            high_pct=weather.cloud_high,
            total_pct=weather.cloud_total,
            sun_elevation_deg=sun_elev,
            clarity=atm,
            dryness=mst,
            corridor=corridor,
            corridor_samples=corridor_samples,
            precip_mm=weather.precipitation_mm,
            precip_last_3h=weather.precipitation_last_3h_mm,
            pressure_trend=weather.pressure_trend_hpa_3h,
            cloud_trend=weather.cloud_total_trend_3h,
            humidity_pct=weather.relative_humidity,
        )
        cq = self.combine_pathways(pathways)
        twilight = pathways["twilight_gradient"]

        # Illumination gate: a beautiful canvas that no light reaches is grey.
        # Applied to every pathway alike — all of them are made of sunlight
        # that has travelled hundreds of km through the lower atmosphere, so
        # cloud along that path extinguishes all of them.
        if corridor is not None:
            cq = clamp(cq * corridor)

        hor = self.horizon_score(horizon_obstruction_deg)

        component_scores = {
            "cloud_quality": cq,
            "atmosphere": atm,
            "moisture": mst,
        }
        base = clamp(weighted_average(component_scores, self._weights))

        # Gates: applied after the average because they bound what is possible
        # rather than contributing a share of it. See the module docstring.
        precip_gate = self.precipitation_gate(weather.precipitation_mm)
        hor_gate = self.horizon_gate(horizon_obstruction_deg)
        physics_score = clamp(base * precip_gate * hor_gate)

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
            component_scores={**component_scores, "horizon": hor},
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
            precipitation_gate=precip_gate,
            horizon_gate=hor_gate,
            twilight_gradient=twilight,
            pathways=pathways,
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

    @staticmethod
    def horizon_strip_blocking(
        samples: Optional[list[tuple[float, float, float]]],
        local_low: float,
        local_mid: float,
    ) -> float:
        """How much cloud stands between the viewer and the sunset horizon.

        See the NEAR_FIELD_* constants for why this is not simply the local
        cell. Falls back to the local cell when no upstream samples exist, so
        callers without corridor data behave exactly as before.
        """
        local_blocking = local_low + local_mid * 0.6
        near = [(d, lo, mid) for d, lo, mid in (samples or []) if d <= NEAR_FIELD_MAX_KM]
        if not near:
            return local_blocking

        total_w = 0.0
        total_b = 0.0
        for distance_km, low_pct, mid_pct in near:
            w = math.exp(-0.5 * (distance_km / NEAR_FIELD_SIGMA_KM) ** 2)
            total_w += w
            total_b += w * (low_pct + mid_pct * 0.6)
        if total_w == 0.0:
            return local_blocking
        strip = total_b / total_w

        return (
            NEAR_FIELD_LOCAL_SHARE * local_blocking
            + (1.0 - NEAR_FIELD_LOCAL_SHARE) * strip
        )

    @staticmethod
    def twilight_gradient_score(
        sun_elevation_deg: float,
        low_pct: float,
        mid_pct: float,
        clarity: float = 70.0,
        dryness: float = 70.0,
        strip_blocking: Optional[float] = None,
        humidity_pct: Optional[float] = None,
    ) -> float:
        """Score the clear-sky twilight gradient, 0-100.

        The cloudless pathway to colour: see the TWILIGHT_* constants for the
        physics and for why this is a pathway rather than a penalty case.

        Three things gate it:

        *Timing.* Peaks at TWILIGHT_PEAK_ELEV_DEG below the horizon. While the
        sun is still up the sky near it is too bright for the gradient to show;
        once it is far enough down the shadow has climbed past the zenith and
        the colour has drained.

        *An open horizon strip.* Blocking cloud hides the band — but the strip
        in question is sky 30-150 km toward the sunset, not the observer's own
        cell, which is mostly sky above and behind them. *strip_blocking* comes
        from horizon_strip_blocking(); when it is absent this falls back to the
        local cell and behaves as it used to.

        *Air quality, sharply.* There is no cloud here to catch the light, so
        everything visible is scattering by the air itself: haze turns the same
        geometry into a flat brown murk. *clarity* is the atmosphere component.
        The response is deliberately steep — in a climate where most evenings
        are cloudless this is most of what separates a memorable gradient from
        an ordinary one, and a gentle curve would rate every clear night alike.

        The other half of the air blend is the water in it. When
        *humidity_pct* is given it scores the boundary layer the light actually
        crosses (TWILIGHT_HUMIDITY_ANCHORS), where moisture HELPS up to the
        point it becomes haze; without it the column-dryness *dryness* score
        stands in, preserving the previous behaviour for callers that have no
        humidity. See the TWILIGHT_HUMIDITY_ANCHORS comment for the measurement
        that established the direction, and for which half of that curve is
        measured and which half is a prior.
        """
        elev_factor = math.exp(
            -0.5 * ((sun_elevation_deg - TWILIGHT_PEAK_ELEV_DEG) / TWILIGHT_SIGMA_DEG) ** 2
        )

        blocking = (
            strip_blocking if strip_blocking is not None
            else low_pct + mid_pct * 0.6
        )
        if blocking <= TWILIGHT_OPEN_BELOW:
            openness = 1.0
        else:
            span = TWILIGHT_BLOCKED_AT - TWILIGHT_OPEN_BELOW
            openness = clamp(1.0 - (blocking - TWILIGHT_OPEN_BELOW) / span, lo=0.0, hi=1.0)

        if humidity_pct is not None:
            vapour = _interpolate(TWILIGHT_HUMIDITY_ANCHORS, humidity_pct)
            wet_share, wet_term = TWILIGHT_VAPOUR_SHARE, vapour
        else:
            wet_share, wet_term = 1.0 - TWILIGHT_CLARITY_SHARE, clamp(dryness)

        air = (
            TWILIGHT_CLARITY_SHARE * clamp(clarity)
            + wet_share * wet_term
        ) / 100.0
        air_factor = air ** TWILIGHT_SHARPNESS

        return clamp(elev_factor * openness * air_factor * TWILIGHT_MAX)

    def lit_cloud_score(
        self,
        low_pct: float,
        mid_pct: float,
        high_pct: float,
        total_pct: float,
        sun_elevation_deg: float = 0.0,
    ) -> float:
        """
        PATHWAY: cloud as a lit canvas — the classic sunset.

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

        return clamp(base)

    # ------------------------------------------------------------------
    # PATHWAY: crepuscular rays
    # ------------------------------------------------------------------

    @staticmethod
    def crepuscular_score(
        low_pct: float,
        mid_pct: float,
        high_pct: float,
        sun_elevation_deg: float,
        clarity: float = 70.0,
    ) -> float:
        """Shafts of light through broken cloud, with the sun still up.

        A different thing to look at than colour: beams and shadows radiating
        from behind a cloud edge. Needs cloud with GAPS in it — a solid deck
        gives no beams and a clear sky gives nothing to break — and the sun
        low but above the horizon, so the beams are long and near-horizontal.

        HONEST LIMITATION: this is the weakest pathway here, because Open-Meteo
        reports cloud FRACTION and brokenness is a texture property. 45 % cover
        could be one solid mass over half the sky or a field of gaps, and those
        are opposite cases. The mid-range fraction is a proxy for the second,
        and it will be wrong sometimes. Fixing it properly needs cloud-texture
        data (satellite imagery, or a convection-resolving model), which is a
        data problem, not a scoring one — see docs/scoring-v2-plan.md.
        """
        # Sun low but up: beams need a shallow angle and a lit sky.
        if not (0.0 <= sun_elevation_deg <= 12.0):
            return 0.0
        elev_factor = math.exp(-0.5 * ((sun_elevation_deg - 4.0) / 4.0) ** 2)

        # Brokenness proxy: peak at ~45 % cover in the beam-forming layers.
        breaking = low_pct + mid_pct * 0.8
        broken_factor = bell_curve(breaking, peak=45.0, sigma=22.0)

        # Beams are made visible by scattering off haze — but heavy haze kills
        # the contrast. This is the one pathway that wants MIDDLING clarity.
        clarity_factor = bell_curve(clamp(clarity), peak=72.0, sigma=22.0)

        # High cirrus above diffuses the source and softens the shafts.
        high_factor = 1.0 - clamp(high_pct, hi=100.0) / 100.0 * 0.3

        return clamp(elev_factor * broken_factor * clarity_factor * high_factor * CREPUSCULAR_MAX)

    # ------------------------------------------------------------------
    # PATHWAY: breaking storm
    # ------------------------------------------------------------------

    @staticmethod
    def breaking_storm_score(
        precip_mm: float,
        cloud_low: float,
        cloud_mid: float,
        cloud_high: float,
        sun_elevation_deg: float,
        precip_last_3h: Optional[float] = None,
        pressure_trend: Optional[float] = None,
        cloud_trend: Optional[float] = None,
    ) -> float:
        """The sky clearing from the west just as the sun reaches the gap.

        The most spectacular and least common of these. Rain has stopped, the
        deck is breaking up, and the low sun fires into the ragged underside of
        a departing storm — washed-clean air, extreme contrast against a dark
        eastern sky. Almost every "best sunset I ever saw" is this one.

        Scored separately rather than as a bonus inside the cloud pathway
        because its ingredients are different in KIND: it is about change over
        time (rain that stopped, cloud that is thinning, pressure that is
        rising), not about a state of the sky. The old engine could only
        express it as a +15 clearing bonus buried in the moisture component,
        where it was worth almost nothing.

        Returns 0 while it is still raining — then it is just rain.
        """
        if precip_mm >= 0.3:
            return 0.0
        if precip_last_3h is None or precip_last_3h < 0.4:
            return 0.0

        # Recency and weight of the rain that just ended.
        recent = clamp(precip_last_3h / 2.5, hi=1.0)

        # There must still be cloud to catch the light, and it must not be a
        # solid low deck. Ragged mid/high over a breaking low layer is ideal.
        canvas = clamp((cloud_mid + cloud_high) / 90.0, hi=1.0)
        not_socked_in = max(0.0, 1.0 - max(0.0, cloud_low - 45.0) / 45.0)

        # Evidence the sky is actually opening rather than just pausing.
        opening = 0.0
        if cloud_trend is not None and cloud_trend < 0.0:
            opening += min(1.0, -cloud_trend / 25.0) * 0.6
        if pressure_trend is not None and pressure_trend > 0.0:
            opening += min(1.0, pressure_trend / 2.5) * 0.4
        opening = clamp(opening, hi=1.0)
        if opening <= 0.0:
            return 0.0

        # Works from just before sunset through the afterglow.
        elev_factor = math.exp(-0.5 * ((sun_elevation_deg + 1.5) / 4.5) ** 2)

        # GEOMETRIC MEAN over the three pieces of evidence, not their product.
        # Multiplying five sub-unit factors was silently fatal: measured over a
        # year at three cities this pathway never once won an evening and never
        # exceeded 27 against a ceiling of 96, because a typical genuine case
        # (0.4 recent x 0.7 canvas x 0.6 opening) multiplies down to 0.17.
        #
        # Multiplication is the right operator for a GATE, where any single
        # zero should kill the result. It is the wrong one for accumulating
        # evidence that all points the same way, which is what these three are:
        # each is a partial, noisy indication that a storm is clearing. The
        # zero-checks above already handle the gate cases.
        evidence = (recent * canvas * opening) ** (1.0 / 3.0)

        return clamp(evidence * not_socked_in * elev_factor * BREAKING_STORM_MAX)

    # ------------------------------------------------------------------
    # PATHWAY: horizon band
    # ------------------------------------------------------------------

    @staticmethod
    def horizon_band_score(
        low_pct: float,
        mid_pct: float,
        sun_elevation_deg: float,
        corridor: Optional[float] = None,
        clarity: float = 70.0,
    ) -> float:
        """A heavy deck overhead, lit from underneath through a clear strip.

        The silhouette sunset: dark cloud filling the sky, and a bright band of
        colour under it where the light gets in beneath the deck. Every other
        pathway reads this as a bad evening, because locally it IS heavy cloud
        — the old engine's low-cloud penalty and overcast penalty both fire.

        What separates it from a genuinely blocked sky is upstream: the
        corridor has to be OPEN even though the sky overhead is not. That is
        exactly what light_corridor_factor measures, so this is the one pathway
        that takes the corridor as an input rather than only being scaled by it
        afterwards. Without heavy local cloud AND a clear corridor, it is not
        this kind of evening and the score is zero.

        Deliberately capped low: it is a real and often striking sunset, but a
        narrow band of light under a grey lid is a lesser event than a sky on
        fire, and it fails more often than it succeeds.
        """
        # No corridor measurement means no evidence, and absent evidence this
        # pathway is indistinguishable from a plainly blocked sky. Elsewhere a
        # missing corridor defaults to 1.0 ("assume unblocked") because it only
        # scales a score; here it would be the entire justification, so it must
        # be measured. Without this guard a stratus deck scored ABOVE a cirrus
        # overcast in the no-corridor case, which inverts the physics.
        if corridor is None:
            return 0.0
        deck = low_pct + mid_pct * 0.6
        if deck < HORIZON_BAND_MIN_DECK:
            return 0.0
        if corridor < HORIZON_BAND_MIN_CORRIDOR:
            return 0.0

        # More deck is more dramatic, up to the point of being total.
        deck_factor = clamp((deck - HORIZON_BAND_MIN_DECK) / 35.0, hi=1.0)
        # An open corridor is the whole mechanism; scale hard on it.
        corridor_factor = clamp(
            (corridor - HORIZON_BAND_MIN_CORRIDOR) / (1.0 - HORIZON_BAND_MIN_CORRIDOR),
            hi=1.0,
        )
        # The band lives in the few degrees around sunset.
        elev_factor = math.exp(-0.5 * ((sun_elevation_deg + 0.5) / 3.0) ** 2)
        clarity_factor = 0.45 + 0.55 * clamp(clarity) / 100.0

        return clamp(
            deck_factor * corridor_factor * elev_factor * clarity_factor * HORIZON_BAND_MAX
        )

    # ------------------------------------------------------------------
    # Pathways: evaluation and combination
    # ------------------------------------------------------------------

    def pathway_scores(
        self,
        *,
        low_pct: float,
        mid_pct: float,
        high_pct: float,
        total_pct: float,
        sun_elevation_deg: float = 0.0,
        clarity: float = 70.0,
        dryness: float = 70.0,
        corridor: Optional[float] = None,
        corridor_samples: Optional[list[tuple[float, float, float]]] = None,
        precip_mm: float = 0.0,
        precip_last_3h: Optional[float] = None,
        pressure_trend: Optional[float] = None,
        cloud_trend: Optional[float] = None,
        humidity_pct: Optional[float] = None,
    ) -> dict[str, float]:
        """Score every route to a beautiful sunset independently.

        See the PATHWAYS section at the top of this module for why this is a
        set rather than a formula.
        """
        strip_blocking = (
            self.horizon_strip_blocking(corridor_samples, low_pct, mid_pct)
            if corridor_samples else None
        )
        return {
            "lit_cloud": self.lit_cloud_score(
                low_pct, mid_pct, high_pct, total_pct, sun_elevation_deg
            ),
            "twilight_gradient": self.twilight_gradient_score(
                sun_elevation_deg, low_pct, mid_pct, clarity, dryness,
                strip_blocking, humidity_pct,
            ),
            "crepuscular": self.crepuscular_score(
                low_pct, mid_pct, high_pct, sun_elevation_deg, clarity
            ),
            "breaking_storm": self.breaking_storm_score(
                precip_mm, low_pct, mid_pct, high_pct, sun_elevation_deg,
                precip_last_3h, pressure_trend, cloud_trend,
            ),
            "horizon_band": self.horizon_band_score(
                low_pct, mid_pct, sun_elevation_deg, corridor, clarity
            ),
        }

    @staticmethod
    def combine_pathways(scores: dict[str, float]) -> float:
        """Collapse the pathway scores into one number.

        NOT a weighted average. Averaging asks every evening to be good in
        every way at once, which no real sunset is: it would rank a mediocre
        sky that ticks several boxes above a cloudless one that is doing one
        thing superbly. That is the failure this whole structure exists to
        avoid.

        NOT a plain maximum either. When two mechanisms are genuinely running —
        a breaking storm that also leaves a lit deck — the evening is better
        than either alone, and a max cannot say so.

        So: the best pathway sets the score, and the others add a modest lift
        that shrinks as the best approaches 100. An evening is as good as its
        best route to beauty, plus a little for having more than one.
        """
        if not scores:
            return 0.0
        values = sorted(scores.values(), reverse=True)
        best = values[0]
        if best <= 0.0:
            return 0.0
        secondary = sum(v for v in values[1:]) / 100.0
        lift = MULTI_PATHWAY_LIFT * min(secondary, 1.5) * (1.0 - best / 100.0)
        return clamp(best + lift)

    @staticmethod
    def dominant_pathway(scores: dict[str, float]) -> Optional[str]:
        """Which route to beauty this evening is taking, if any.

        The UI needs this as much as the number does: "lit clouds" and "clear
        gradient" want different words, different best-viewing times, and send
        someone outside looking for different things.
        """
        if not scores:
            return None
        key = max(scores, key=lambda k: scores[k])
        return key if scores[key] > 0.0 else None

    def cloud_quality_score(
        self,
        low_pct: float,
        mid_pct: float,
        high_pct: float,
        total_pct: float,
        sun_elevation_deg: float = 0.0,
        clarity: float = 70.0,
        dryness: float = 70.0,
    ) -> float:
        """Combined colour-potential score across all pathways.

        Kept as the component-level entry point (it is what the 0.60 weight is
        applied to). Pathways needing data beyond cloud and sun — the corridor,
        the precipitation history — fall back to their neutral defaults here,
        so this is the sky-only view; score() passes the full picture.
        """
        return self.combine_pathways(
            self.pathway_scores(
                low_pct=low_pct,
                mid_pct=mid_pct,
                high_pct=high_pct,
                total_pct=total_pct,
                sun_elevation_deg=sun_elevation_deg,
                clarity=clarity,
                dryness=dryness,
            )
        )

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

    @staticmethod
    def aerosol_clarity(aerosol_od: float) -> float:
        """Score air cleanliness from aerosol optical depth, 0-100.

        MONOTONE DECREASING, deliberately. This replaces a bell curve that
        peaked at AOD 0.18 and scored pristine air (AOD 0.03) at only ~56 —
        i.e. it treated a smog layer as the ideal and clean desert air as
        mediocre.

        That was backwards. Corfidi (NOAA/SPC, "A Guide to Forecasting Colorful
        Sunrises and Sunsets") is explicit that tropospheric aerosols SUBDUE
        sunset colour: they scatter and absorb across the visible band, which
        washes the reds toward a flat milky orange. Clean air is the main
        ingredient — the vivid skies are the ones after a front has scrubbed
        the boundary layer, not the hazy ones.

        Please do not "fix" this back into a bell curve. The intuition that a
        little haze helps comes from confusing aerosol with high cirrus, which
        is a cloud effect and is already scored in cloud_quality.
        """
        return _interpolate(AEROSOL_ANCHORS, aerosol_od)

    def atmosphere_score(
        self,
        visibility_m: Optional[float],
        aerosol_od: Optional[float],
    ) -> float:
        """
        Score atmospheric clarity — how cleanly light reaches you.

        Aerosol leads (see aerosol_clarity); visibility, when the source
        reports it, is a second look at the same physical quantity and gets
        the smaller share.

        *visibility_m* of None means the source did not report it — true of
        every ERA5 archive hour. It is then simply left out, rather than
        defaulting to 15 km as before: a constant contributes nothing but a
        fixed offset, and it made the climatology's atmosphere term a function
        of humidity alone.

        Surface humidity is NOT penalised here any more. Moisture is scored
        once, as a water column, in moisture_score — counting it in both places
        double-charged humid evenings.
        """
        aer_score = (
            self.aerosol_clarity(aerosol_od) if aerosol_od is not None else None
        )
        vis_score = (
            clamp(visibility_m / 25_000.0 * 100.0) if visibility_m is not None else None
        )

        if aer_score is not None and vis_score is not None:
            return clamp(aer_score * 0.65 + vis_score * 0.35)
        if aer_score is not None:
            return clamp(aer_score)
        if vis_score is not None:
            return clamp(vis_score)
        # Neither reported: a neutral score, so an evening is neither rewarded
        # nor punished for a gap in the data.
        return 60.0

    # ------------------------------------------------------------------
    # Component 3: Moisture / Precipitation (weight 0.20)
    # ------------------------------------------------------------------

    @staticmethod
    def column_dryness(tcwv_kg_m2: float) -> float:
        """Score the whole-atmosphere water load, 0-100. Drier is better.

        TCWV (kg/m², equivalently mm of precipitable water) is the total water
        vapour in the column above you. It is what actually mutes sunset
        colour: vapour both scatters and feeds the sub-visible haze that turns
        red into milky orange, along the entire slant path the light takes.

        Surface relative humidity — what this component used to score — is a
        measurement of the bottom two metres and says very little about that
        path. It is also bounded and heavily diurnal, so on dry evenings it sat
        pinned at 100 and contributed no information: measured over a year at
        three cities, moisture held 15 % of the weight while accounting for
        2.8-5.0 % of the variance in the final score.

        Anchors are absolute (8 mm is dry anywhere, 45 mm is tropical), which
        is safe now that the displayed score is a percentile against local
        climatology — an always-humid location is ranked against its own
        history, not against a desert.
        """
        return _interpolate(TCWV_ANCHORS, tcwv_kg_m2)

    def moisture_score(
        self,
        precip_mm: float,
        humidity_pct: float,
        tcwv: Optional[float] = None,
        *,
        precip_last_3h: Optional[float] = None,
        pressure_trend: Optional[float] = None,
        cloud_trend: Optional[float] = None,
        vis_trend: Optional[float] = None,
    ) -> float:
        """
        Score moisture — the atmospheric water column, and post-rain clearing.

        The base is column_dryness(*tcwv*). Surface humidity survives only as a
        small correction (max −10 above 85 %), because a saturated boundary
        layer does put haze in the lowest, brightest part of the view even when
        the column above is dry. When *tcwv* is unavailable — manual overrides,
        or any source that stops returning it — the old surface-RH curve stands
        in, so a missing field degrades the component rather than breaking it.

        NOTE: active precipitation is NOT scored here. It is a multiplicative
        gate (see precipitation_gate) because rain ends a sunset rather than
        reducing it by some number of points. *precip_mm* is still needed to
        decide whether the clearing bonus applies, which it cannot while it is
        still raining.

        - Dry column → high base score
        - Recent rain + currently dry → clearing bonus (post-rain glow)
        - Rising pressure / improving visibility / clearing clouds → bonus
        - Saturated surface air → mild penalty

        Clearing bonus: up to +15 pts when rain stopped recently and
        atmospheric signals show improvement.
        """
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

        if tcwv is not None:
            base = self.column_dryness(tcwv)
            # Small correction only — the column already carries the moisture.
            hum_penalty = max(0.0, (humidity_pct - 85.0) / 15.0 * 10.0)
        else:
            # Fallback: the pre-column behaviour, penalty only above 85 %.
            base = 100.0
            hum_penalty = max(0.0, (humidity_pct - 85.0) / 15.0 * 25.0)

        return clamp(base - hum_penalty + clearing_bonus)

    # ------------------------------------------------------------------
    # Component 4: Horizon (weight 0.10)
    # ------------------------------------------------------------------

    def horizon_score(self, obstruction_deg: float) -> float:
        """
        Score the unobstructed horizon, 0-100, for DISPLAY only.

        This is no longer part of the weighted average — it is reported so the
        user can see how much their own outlook costs them, and applied through
        horizon_gate(). Measured across a year it had a standard deviation of
        zero, because it is a property of where you stand, not of the evening.
        As a weighted addend it therefore added a constant ~9 points to every
        score without ever distinguishing one night from another.

        0 degrees = open ocean / flat field = 100.
        5 degrees = gentle hills / low suburbs = ~88.
        15+ degrees = dense urban / deep valley = ~49.
        """
        return clamp(self.horizon_gate(obstruction_deg) * 100.0)

    @staticmethod
    def horizon_gate(obstruction_deg: float) -> float:
        """Fraction of the display an obstructed horizon leaves visible.

        A raised horizon hides the sun early and cuts off the brightest,
        lowest part of the sky, so it scales the whole result rather than
        contributing a share of it.

        Deliberately gentler near zero than the old scoring curve: 2 degrees is
        an essentially open horizon (trees across a field) and used to cost
        nearly 9 points. It now costs under 4 %.

            0°  → 1.00      5°  → 0.88      15° → 0.49
            2°  → 0.96      10° → 0.70      25°+ → 0.35 (floor)
        """
        if obstruction_deg <= 0.0:
            return 1.0
        loss = (min(obstruction_deg, 25.0) / 25.0) ** 1.3
        return max(HORIZON_FLOOR, 1.0 - loss)

    @staticmethod
    def precipitation_gate(precip_mm: float) -> float:
        """Fraction of the display that survives active precipitation.

        Rain does not subtract points from a sunset — it replaces it with grey.
        Modelled as exponential decay so that drizzle barely registers (a
        breaking shower at sunset can be spectacular) while steady rain
        collapses the score.

            0.0 mm → 1.00      1.0 mm → 0.55      3.0 mm → 0.17
            0.2 mm → 0.89      2.0 mm → 0.30      5.0 mm+ → 0.15 (floor)
        """
        if precip_mm <= 0.0:
            return 1.0
        return max(PRECIP_FLOOR, math.exp(-0.6 * precip_mm))

    # ------------------------------------------------------------------
    # Category mapping
    # ------------------------------------------------------------------

    @staticmethod
    def percentile_to_display_score(percentile: float) -> float:
        """Map a climatological percentile in [0, 1] onto a 0-100 rank score.

        NO LONGER THE DISPLAY PATH. The app shows the absolute physics score;
        this remains for analysis (scripts/evaluate.py uses it to show what a
        purely rank-based scale would look like) and because the anchors below
        still document the intended shape of a rarity scale.

        Piecewise-linear through CALIBRATION_ANCHORS, so each band ends up with
        a fixed share of evenings regardless of climate. Monotone by
        construction: a better evening never displays a lower number.
        """
        return _interpolate(CALIBRATION_ANCHORS, clamp(percentile, lo=0.0, hi=1.0))

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
        ensemble_cloud_spread: Optional[float] = None,
    ) -> float:
        """
        Estimate prediction confidence in [15, 92].

        Confidence is higher when:
        - The score is far from the ambiguous middle (40–60)
        - Multiple window points agree
        - Aerosol data is real (not proxy-estimated)
        - Forecast ensemble members agree on cloud cover (or the sunset is imminent,
          when no ensemble reading is available)

        Confidence is lower when:
        - Signals conflict (good clouds + active rain)
        - The window is highly volatile (one great point, rest collapse)
        - Aerosol is estimated
        - Active rain conflicts with otherwise strong sky structure
        - Forecast ensemble members disagree on cloud cover, or (without an
          ensemble reading) the target sunset is many days out

        *ensemble_cloud_spread* is the standard deviation of total cloud cover
        (%) across forecast ensemble members at the sunset hour — an actual
        measurement of forecast uncertainty (see weather_service.get_ensemble_
        cloud_spread). When present it REPLACES the lead-time term below,
        because both approximate the same thing and the ensemble spread is the
        real version of what lead-time was guessing at. Pass None when no
        ensemble reading exists (past dates, or beyond the model's ~7.5-day
        honest horizon) to fall back to lead-time.

        *lead_time_hours* is the gap from now to the target sunset; pass None
        (the default) to skip the lead-time term entirely (e.g. for overrides).
        Ignored when *ensemble_cloud_spread* is given.
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

        # Forecast uncertainty: prefer the real measurement (ensemble spread)
        # over the lead-time guess when one is available.
        if ensemble_cloud_spread is not None:
            base += _interpolate(ENSEMBLE_SPREAD_CONFIDENCE_ANCHORS, ensemble_cloud_spread)
        elif lead_time_hours is not None:
            # Imminent sunsets (and observed past dates) get a small boost;
            # ~1 day out is neutral; each further day is penalised, reflecting
            # how forecast skill decays with lead time. Only a fallback for
            # when no ensemble reading exists — the guess ensemble spread replaces.
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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _interpolate(anchors: list[tuple[float, float]], x: float) -> float:
    """Piecewise-linear lookup through (x, y) *anchors*, which must be sorted by x.

    Values outside the table clamp to the nearest endpoint. Used for every
    response curve in this module so the shape of a curve is a readable table
    rather than a fitted constant — the aerosol bell curve survived as long as
    it did partly because `bell_curve(aod, peak=0.18, sigma=0.15)` does not
    look wrong until you evaluate it.
    """
    if not anchors:
        return 0.0
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for i in range(len(anchors) - 1):
        x_lo, y_lo = anchors[i]
        x_hi, y_hi = anchors[i + 1]
        if x_lo <= x <= x_hi:
            span = x_hi - x_lo
            frac = 0.0 if span == 0 else (x - x_lo) / span
            return y_lo + frac * (y_hi - y_lo)
    return anchors[-1][1]
