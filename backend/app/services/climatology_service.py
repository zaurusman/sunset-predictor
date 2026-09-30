"""Local climatology: what a raw score is worth *at this location*.

WHY
---
Raw physics scores are not comparable across places. Measured over a year, the
bottom decile is 42.9 in Tel Aviv against 29.6 in London — Tel Aviv simply does
not get the blocked-corridor evenings London does, and its corridor drops below
0.6 on 2 % of evenings against London's 37 %. Fixed cutoffs on that raw scale
therefore mean different things in different cities, which is how "Epic" ended up
firing on 10-15 % of evenings in one place and "Decent" on 75 % in another.

So this service ranks tonight against the location's own history: "how does
tonight compare with evenings here?" That rank was briefly the displayed score
itself; it is now shown BENEATH the absolute score as context, and the ranking
window is SEASONAL (SEASON_WINDOW_DAYS below), so late August is compared with
late August rather than with December fronts.

Why the display reverted to the raw score: percentile display is
self-normalising. Improving the physics for a whole kind of evening lifts every
evening of that kind, leaving each one's rank where it was — measured at
raw 48.9 → 57.4 against displayed 30.9 → 30.6. See docs/scoring-v2-plan.md.

COST AND COLD START
-------------------
Only the evenings the seasonal rank can actually use are fetched — the last
SEASON_WINDOW_DAYS, plus last year's same date forward through the window and
the cache lifetime (see _build_ranges) — about 120 days rather than a year.
Scored exactly as before, light corridor included, so the percentile means
what it always did; the build just no longer pays for the ~245 evenings the
seasonal rank never looks at.

  - results are cached for CLIMATOLOGY_TTL (30 days); a climate does not move
  - one curve serves a CLIMATE_CELL_DEG cell (~25 km), built at its centre
  - a cold location is warmed in the BACKGROUND, and meanwhile falls back to
    REFERENCE_QUANTILES, a global curve averaged across three climates
  - a FAILED build is not retried for CLIMATOLOGY_RETRY_COOLDOWN_SECONDS;
    otherwise every prediction for a cold cell re-launched it, hammering an
    Open-Meteo that was very likely failing because of rate limits

The fallback matters: without it a cold location would show raw physics scores,
which are on a visibly different scale, and the number would jump once the warm
finished. With it, the first view is approximately right and later views refine.
"""
from __future__ import annotations

import asyncio
import time
from datetime import date, timedelta
from typing import Optional

from app.core.logging import get_logger
from app.services.astronomy_service import AstronomyService
from app.services.scoring_engine import ScoringEngine
from app.services.weather_service import WeatherService
from app.utils.cache import TTLCache

logger = get_logger(__name__)

# A climate does not change month to month; re-deriving it more often is waste.
CLIMATOLOGY_TTL_SECONDS = 30 * 86_400

# Below this many scored days the distribution is too thin to rank against.
# A full build yields ~120 (see _build_ranges); this tolerates a partial one.
MIN_USABLE_DAYS = 60

# Side length of the grid cell one curve serves, in degrees (~25 km). The
# archive is ERA5, whose native grid is 0.25°, so a finer cell buys nothing but
# extra builds; a metro area's users now share one instead of each paying for
# their own. Not coarser: at 0.5° Tel Aviv's curve would be built ~20 km
# inland, and the coast is exactly where sunset climate changes fastest.
CLIMATE_CELL_DEG = 0.25

# After a failed build, wait this long before trying again for that cell.
CLIMATOLOGY_RETRY_COOLDOWN_SECONDS = 1800

# Half-width of the seasonal comparison window, in days.
#
# Tonight is ranked against evenings from a similar time of YEAR, not against
# the whole twelve months. Sunset quality is strongly seasonal — Mediterranean
# summer is hazy and cloudless while the drama arrives with winter fronts — so
# ranking against the full year produced month-long runs where every evening
# read the same, which tells a daily-glance user nothing. Ranked seasonally the
# app can say "good for August" instead.
#
# 45 days each side gives ~91 samples: wide enough that the rank is stable,
# narrow enough that late August is not being compared with November.
SEASON_WINDOW_DAYS = 45

# BUMP THIS whenever a change moves the raw score scale — a component curve, a
# weight, a gate, anything score() touches.
#
# Cached curves are persisted to disk and live for 30 days, so without a
# version in the key a scoring change leaves every warm location ranking new
# scores against a distribution built by the old engine. That failure is
# silent: the number stays plausible and simply means the wrong thing. It was
# caught live after Phase 3 shifted the median from ~47 to ~53, with a stale
# curve still being served. The clear-sky pathway moved it again, to ~64.
#
# Bumping the version orphans the old entries; they expire on their own TTL.
SCALE_VERSION = 6

# Global fallback: the 0/5/10/…/100th percentile of raw scores, averaged across
# Tel Aviv, London and San Francisco (365 days each, generated by
# scripts/evaluate.py). Used only until a location's own curve is warm.
#
# REGENERATE THIS whenever the raw scale moves — any change to a component
# curve, a weight, or a gate. It is a snapshot of the raw distribution, so a
# stale curve silently mis-ranks every cold location. Phase 3 shifted the
# median from ~47 to ~53, which would have read as "below average" everywhere
# on first view.
REFERENCE_QUANTILES: list[float] = [
      6.0,  29.3,  35.2,  38.7,  41.3,
     44.4,  48.4,  52.2,  56.0,  58.8,
     61.5,  63.6,  65.5,  67.1,  69.6,
     72.0,  74.3,  76.9,  80.1,  82.8,
     94.3,
]


class ClimatologyService:
    """Builds and caches the raw-score distribution for a location."""

    def __init__(
        self,
        weather_service: WeatherService,
        astro_service: AstronomyService,
        scoring_engine: ScoringEngine,
        cache: TTLCache,
    ) -> None:
        self._weather = weather_service
        self._astro = astro_service
        self._scoring = scoring_engine
        self._cache = cache
        # Guards against a burst of requests for a cold location each kicking
        # off its own backfill.
        self._in_flight: set[tuple[float, float]] = set()
        # cell → monotonic time before which a failed build is not retried.
        self._retry_after: dict[tuple[float, float], float] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def percentile_of(
        self,
        lat: float,
        lon: float,
        raw_score: float,
        on_date: Optional[date] = None,
    ) -> tuple[float, bool]:
        """Rank *raw_score* against this location's history at this time of year.

        Returns ``(percentile, is_local)`` where percentile is in [0, 1] and
        *is_local* says whether the location's own climatology was used or the
        global reference curve stood in.

        Never blocks and never raises: a cold location returns a reference-based
        rank immediately.
        """
        entries = self._cache.get(self._key(lat, lon))
        if entries:
            window = _seasonal_window(
                entries, (on_date or date.today()).timetuple().tm_yday
            )
            if len(window) >= MIN_SEASONAL_SAMPLES:
                return _rank_in_sorted(window, raw_score), True
            # Too thin a season (a partial year of data) — fall back to the
            # full curve rather than ranking against a handful of evenings.
            return _rank_in_sorted(sorted(v for _, v in entries), raw_score), True
        return _rank_in_quantiles(REFERENCE_QUANTILES, raw_score), False

    def is_warm(self, lat: float, lon: float) -> bool:
        return self._cache.get(self._key(lat, lon)) is not None

    def warm_in_background(self, lat: float, lon: float) -> None:
        """Kick off a climatology build if one isn't cached or already running."""
        key = self._coords(lat, lon)
        if self.is_warm(lat, lon) or key in self._in_flight:
            return
        if time.monotonic() < self._retry_after.get(key, 0.0):
            return
        self._in_flight.add(key)
        try:
            asyncio.get_running_loop().create_task(self._warm(lat, lon, key))
        except RuntimeError:
            # No running loop (e.g. called from sync test code) — drop the
            # request rather than crash; the fallback curve still works.
            self._in_flight.discard(key)

    async def build(self, lat: float, lon: float) -> Optional[list[tuple[int, float]]]:
        """Build and cache ``(day_of_year, score)`` pairs. Returns None on failure.

        Fetched at the centre of the climate cell, so the curve does not depend
        on which user in the cell happened to trigger the build.
        """
        lat, lon = self._coords(lat, lon)
        windows = []
        for start, end in _build_ranges(date.today()):
            windows += await self._weather.get_historical_range_windows(lat, lon, start, end)
        if not windows:
            return None

        corridor_map = await self._weather.get_corridor_samples_map(
            lat, lon, [d for d, _ in windows]
        )

        scores: list[tuple[int, float]] = []
        for d, snaps in windows:
            samples = corridor_map.get(d, [])
            scored = [
                (s.timestamp_label or "sunset",
                 self._scoring.score(s, 2.0, corridor_samples=samples).physics_score)
                for s in snaps
            ]
            if scored:
                # Day-of-year is kept alongside the score so the rank can be
                # taken against a seasonal window rather than the whole year.
                scores.append((d.timetuple().tm_yday,
                               self._scoring.score_window(scored).final_score))

        if len(scores) < MIN_USABLE_DAYS:
            logger.warning(
                "Climatology for (%.2f, %.2f) has only %d days — not caching.",
                lat, lon, len(scores),
            )
            return None

        self._cache.set(self._key(lat, lon), scores, ttl_override=CLIMATOLOGY_TTL_SECONDS)
        ordered = sorted(v for _, v in scores)
        logger.info(
            "Climatology warm for (%.2f, %.2f): %d days, p50=%.1f, p90=%.1f",
            lat, lon, len(ordered),
            ordered[len(ordered) // 2], ordered[int(0.9 * (len(ordered) - 1))],
        )
        return scores

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _warm(self, lat: float, lon: float, key: tuple[float, float]) -> None:
        built = None
        try:
            built = await self.build(lat, lon)
        except Exception as exc:
            # Background task: a failure must not surface anywhere. The
            # reference curve keeps serving until the next attempt.
            logger.warning("Climatology warm failed for (%.2f, %.2f): %s", lat, lon, exc)
        finally:
            self._in_flight.discard(key)
            if built is None:
                self._retry_after[key] = time.monotonic() + CLIMATOLOGY_RETRY_COOLDOWN_SECONDS
            else:
                self._retry_after.pop(key, None)

    @staticmethod
    def _coords(lat: float, lon: float) -> tuple[float, float]:
        """Centre of the CLIMATE_CELL_DEG cell containing (lat, lon)."""
        return (
            round(round(lat / CLIMATE_CELL_DEG) * CLIMATE_CELL_DEG, 4),
            round(round(lon / CLIMATE_CELL_DEG) * CLIMATE_CELL_DEG, 4),
        )

    def _key(self, lat: float, lon: float) -> str:
        return TTLCache.make_key(
            "climatology", f"v{SCALE_VERSION}", *self._coords(lat, lon)
        )


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


def _build_ranges(today: date) -> list[tuple[date, date]]:
    """The two date ranges a build fetches: exactly what the seasonal rank
    needs for as long as the curve is cached.

    The rank takes evenings within SEASON_WINDOW_DAYS of today's day-of-year,
    one year each — the days just gone from THIS year, the days ahead from
    LAST year. As the cached curve ages by k days the window slides forward:
    the days it newly needs lie further into last year's range, so that range
    runs SEASON_WINDOW_DAYS + the cache lifetime past today's date. Days the
    window slides off are simply filtered out by _seasonal_window.
    """
    ttl_days = CLIMATOLOGY_TTL_SECONDS // 86_400
    yesterday = today - timedelta(days=1)
    recent = (today - timedelta(days=SEASON_WINDOW_DAYS), yesterday)
    a_year_ago = today - timedelta(days=365)
    last_year = (a_year_ago, a_year_ago + timedelta(days=SEASON_WINDOW_DAYS + ttl_days))
    return [last_year, recent]


# A seasonal window thinner than this is not worth ranking against.
MIN_SEASONAL_SAMPLES = 40


def _seasonal_window(entries: list[tuple[int, float]], day_of_year: int) -> list[float]:
    """Sorted scores from evenings within SEASON_WINDOW_DAYS of *day_of_year*.

    Wraps around the turn of the year, so a 5 January evening is compared with
    late December as well as mid February.
    """
    picked: list[float] = []
    for doy, score in entries:
        delta = abs(doy - day_of_year)
        if min(delta, 366 - delta) <= SEASON_WINDOW_DAYS:
            picked.append(score)
    picked.sort()
    return picked


def _rank_in_sorted(sorted_scores: list[float], value: float) -> float:
    """Fraction of *sorted_scores* below *value*, in [0, 1].

    Uses a midpoint rule for ties so that a value equal to a long run of
    identical scores lands in the middle of that run rather than at one end —
    which matters because the raw scale still has flat spots.
    """
    n = len(sorted_scores)
    if n == 0:
        return 0.5

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_scores[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    below = lo

    hi2 = below
    while hi2 < n and sorted_scores[hi2] == value:
        hi2 += 1
    equal = hi2 - below

    return (below + equal / 2.0) / n


def _rank_in_quantiles(quantiles: list[float], value: float) -> float:
    """Rank *value* against an evenly-spaced quantile curve, interpolating."""
    n = len(quantiles)
    if n < 2:
        return 0.5
    step = 1.0 / (n - 1)

    if value <= quantiles[0]:
        return 0.0
    if value >= quantiles[-1]:
        return 1.0

    for i in range(n - 1):
        lo, hi = quantiles[i], quantiles[i + 1]
        if lo <= value <= hi:
            span = hi - lo
            frac = 0.5 if span == 0 else (value - lo) / span
            return (i + frac) * step
    return 1.0
