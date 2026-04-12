"""
Explanation engine: converts component scores into plain-English reasons.

Each reason is a short, non-technical sentence that a general audience can
understand.  Reasons are ordered by their absolute contribution to the final
score (most impactful first), capped at 6 items.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from app.schemas.prediction import PhysicsBreakdown
from app.schemas.weather import WeatherSnapshot

if TYPE_CHECKING:
    from app.services.scoring_engine import WindowResult


class ExplanationEngine:
    """
    Generates 3–6 natural-language reasons from a scoring breakdown and weather.

    Rules are simple threshold-based mappings — easy to audit and extend.
    Accepts an optional WindowResult so reasons can reference the best
    viewing moment and post-rain clearing signals.
    """

    def generate(
        self,
        weather: WeatherSnapshot,
        breakdown: PhysicsBreakdown,
        category: str,
        window_result: Optional["WindowResult"] = None,
    ) -> list[str]:
        """
        Return a list of 3–6 ordered explanation strings.

        Each string is a complete sentence suitable for display in the UI.
        """
        candidates: list[tuple[float, str]] = []

        # -----------------------------------------------------------------
        # Window timing hint (when multiple points were scored)
        # -----------------------------------------------------------------
        if window_result is not None and len(window_result.window_scores) > 1:
            best = window_result.best_label
            if best == "+15m":
                candidates.append((75.0, "Best viewing is likely about 15 minutes after sunset — stay for the afterglow."))
            elif best == "+30m":
                candidates.append((72.0, "Conditions may improve after sunset — the afterglow could be the highlight."))
            elif best == "-15m":
                candidates.append((70.0, "The best colour may arrive just before the sun dips below the horizon."))
            # For "sunset" we skip the timing hint — it's the default expectation

            # Volatile window note
            if window_result.volatility_penalty > 4.0:
                candidates.append((30.0, "Conditions look inconsistent across the window — confidence is moderate."))

        # -----------------------------------------------------------------
        # Cloud Quality reasons
        # -----------------------------------------------------------------
        cq = breakdown.cloud_quality_score

        if weather.cloud_high >= 50 and weather.cloud_low < 20:
            candidates.append((cq, "High clouds look very promising for catching vivid colour."))
        elif weather.cloud_high >= 25 and weather.cloud_low < 25:
            candidates.append((cq, "Some high clouds should help scatter warm light at sunset."))
        elif weather.cloud_high < 10 and weather.cloud_mid < 10:
            candidates.append(
                (100 - cq, "Very few clouds in the sky — clear conditions produce less colour drama.")
            )

        if weather.cloud_low >= 50:
            candidates.append(
                (100 - cq, "Heavy low clouds near the horizon may block the best light entirely.")
            )
        elif weather.cloud_low >= 25:
            candidates.append(
                (100 - cq, "Low clouds near the horizon could partially block the sunset.")
            )

        if weather.cloud_total >= 85:
            candidates.append(
                (100 - cq, "An overcast sky tends to diffuse light and reduce vivid colour.")
            )
        elif weather.cloud_total >= 60 and weather.cloud_high < 20:
            candidates.append(
                (100 - cq, "A mostly cloudy sky with few high clouds is not ideal for colour.")
            )

        if weather.cloud_mid >= 20 and weather.cloud_low < 15:
            candidates.append(
                (cq, "Mid-level clouds add texture and can produce beautiful colour gradients.")
            )

        # -----------------------------------------------------------------
        # Atmosphere reasons
        # -----------------------------------------------------------------
        atm = breakdown.atmosphere_score
        vis_km = weather.visibility_m / 1000.0

        if atm >= 70:
            if vis_km >= 15:
                candidates.append((atm, "Clear air and good visibility will help colours pop."))
            else:
                candidates.append((atm, "Atmospheric conditions look favourable for strong colour."))
        elif atm >= 45:
            if weather.aerosol_optical_depth is not None and weather.aerosol_optical_depth > 0.1:
                candidates.append(
                    (atm, "A touch of haze may produce warm golden tones — moderate aerosol helps.")
                )
            else:
                candidates.append((atm, "Atmospheric clarity is decent — expect reasonable colour."))
        else:
            if vis_km < 8:
                candidates.append(
                    (100 - atm, f"Reduced visibility ({vis_km:.0f} km) may mute sunset colours.")
                )
            else:
                candidates.append(
                    (100 - atm, "Hazy or humid air may wash out the colours.")
                )

        if weather.aerosol_is_estimated:
            candidates.append(
                (5.0, "Aerosol data was estimated from visibility — confidence is slightly lower.")
            )

        # -----------------------------------------------------------------
        # Moisture / Precipitation reasons
        # -----------------------------------------------------------------
        mst = breakdown.moisture_score

        if weather.precipitation_mm >= 2.0:
            candidates.append(
                (100 - mst, f"Rain near sunset ({weather.precipitation_mm:.1f} mm) strongly reduces colour chances.")
            )
        elif weather.precipitation_mm > 0.2:
            candidates.append(
                (100 - mst, "Light precipitation near sunset may dampen colour intensity.")
            )

        # Post-rain clearing bonus explanation
        if (
            weather.precipitation_mm < 0.1
            and weather.precipitation_last_3h_mm is not None
            and weather.precipitation_last_3h_mm > 0.5
        ):
            candidates.append(
                (65.0, "Recent rain followed by clearing can produce especially vivid afterglow colour.")
            )
        elif (
            weather.precipitation_mm < 0.1
            and weather.pressure_trend_hpa_3h is not None
            and weather.pressure_trend_hpa_3h > 1.5
        ):
            candidates.append(
                (55.0, "Rising pressure signals improving conditions heading into sunset.")
            )
        elif (
            weather.precipitation_mm < 0.1
            and weather.cloud_total_trend_3h is not None
            and weather.cloud_total_trend_3h < -15.0
        ):
            candidates.append(
                (50.0, "Clouds have been clearing over the past few hours — good sign.")
            )

        if weather.relative_humidity >= 85 and weather.precipitation_mm < 0.1:
            candidates.append(
                (100 - mst, "Very high humidity may create a milky haze around the horizon.")
            )

        # -----------------------------------------------------------------
        # Horizon reasons
        # -----------------------------------------------------------------
        hor = breakdown.horizon_score

        if hor < 50:
            candidates.append(
                (100 - hor, "Significant horizon obstruction may block the low sun at key moments.")
            )
        elif hor < 75:
            candidates.append(
                (100 - hor, "Some horizon obstruction could clip the lower portion of the sunset.")
            )

        # -----------------------------------------------------------------
        # Overall context
        # -----------------------------------------------------------------
        if category == "Epic":
            candidates.insert(0, (100.0, "Conditions look exceptional — this could be a memorable sunset."))
        elif category == "Poor" and weather.precipitation_mm < 0.1 and weather.cloud_total > 80:
            candidates.append((0.0, "Overcast skies without interesting cloud structure limit colour potential."))

        # -----------------------------------------------------------------
        # Sort by importance (descending) and deduplicate / cap
        # -----------------------------------------------------------------
        candidates.sort(key=lambda x: x[0], reverse=True)

        seen_prefixes: set[str] = set()
        reasons: list[str] = []
        for _, sentence in candidates:
            prefix = sentence[:30]
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                reasons.append(sentence)
            if len(reasons) == 6:
                break

        if len(reasons) < 3:
            reasons += self._fallback_reasons(weather, breakdown, category)[: 3 - len(reasons)]

        return reasons

    # ------------------------------------------------------------------
    # Fallback reasons when too few candidates
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_reasons(
        weather: WeatherSnapshot,
        breakdown: PhysicsBreakdown,
        category: str,
    ) -> list[str]:
        fallbacks = []
        if weather.cloud_high < 5:
            fallbacks.append("Clear skies are crisp but produce less dramatic colour than a partly-cloudy sunset.")
        if weather.cloud_total < 30:
            fallbacks.append("Low cloud cover means less light-scattering material overhead.")
        if breakdown.atmosphere_score > 60:
            fallbacks.append("Atmospheric clarity is good — what colour there is should be vivid.")
        fallbacks.append(f"Overall conditions point to a {category.lower()} sunset today.")
        return fallbacks
