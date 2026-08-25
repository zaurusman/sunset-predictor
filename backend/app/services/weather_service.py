"""Weather data service: fetches and normalises Open-Meteo API data."""
from __future__ import annotations

import asyncio
import math
import random
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.weather import WeatherOverride, WeatherSnapshot
from app.services.astronomy_service import AstronomyService
from app.utils.geo import destination_point
from app.utils.cache import TTLCache

logger = get_logger(__name__)
UTC = timezone.utc

# HTTP status codes worth retrying: 429 (rate-limited) and any 5xx (transient
# server error). Other 4xx (e.g. 400 bad params) won't fix themselves on retry.
_RETRYABLE_STATUS = {429}


class WeatherUnavailableError(Exception):
    """Raised when the weather provider is unreachable or rate-limiting us
    after exhausting retries.

    Mapped to HTTP 503 at the API layer so callers see a clear "try again
    shortly" signal instead of a generic 500.
    """


# ---------------------------------------------------------------------------
# Open-Meteo variable lists
# ---------------------------------------------------------------------------

FORECAST_HOURLY_VARS = ",".join([
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "visibility",
    "relative_humidity_2m",
    "dew_point_2m",
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
    "weather_code",
])

AIR_QUALITY_HOURLY_VARS = "aerosol_optical_depth,dust"

# Light-corridor sampling: only the blocking layers are needed upstream, so the
# request stays small even across six coordinates.
CORRIDOR_HOURLY_VARS = "cloud_cover_low,cloud_cover_mid"

# Distances (km) along the sunset azimuth at which the corridor is sampled.
# Chosen to bracket the illumination tangent distances of the three cloud
# layers — ~113 km (low), ~226 km (mid), ~339 km (high) — with points either
# side of each so the Gaussian weighting in the scoring engine has support.
CORRIDOR_DISTANCES_KM: list[float] = [60.0, 120.0, 180.0, 240.0, 320.0, 400.0]

ARCHIVE_HOURLY_VARS = ",".join([
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "visibility",
    "relative_humidity_2m",
    "dew_point_2m",
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
])


class WeatherService:
    """
    Fetches weather data from Open-Meteo and returns normalised WeatherSnapshot objects.

    Caches responses to avoid redundant API calls (TTL configurable).
    All returned datetimes are UTC-aware.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        astro_service: AstronomyService,
        cache: TTLCache,
        settings: Settings,
    ) -> None:
        self._http = http_client
        self._astro = astro_service
        self._cache = cache
        self._settings = settings

    def _ckey_coords(self, lat: float, lon: float) -> tuple[float, float]:
        """Round (lat, lon) for the cache key (not for the actual fetch).

        Open-Meteo grid-snaps coordinates, so rounding lets nearby lookups —
        different users, jittery geolocation — share one cached fetch instead
        of each triggering another API call. Precision is configurable via
        CACHE_COORD_DECIMALS (1 ≈ 11 km, 2 ≈ 1 km).
        """
        decimals = self._settings.CACHE_COORD_DECIMALS
        return round(lat, decimals), round(lon, decimals)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_snapshot_at_sunset(
        self,
        lat: float,
        lon: float,
        target_date: date,
        override: Optional[WeatherOverride] = None,
    ) -> WeatherSnapshot:
        """
        Fetch the weather snapshot for the sunset hour on *target_date*.

        If *override* is provided, any non-None fields in it replace
        the corresponding fetched values.

        Optimisation: if the override covers ALL critical measurement fields,
        the weather API call is skipped entirely (useful for testing and for
        users who want fully-controlled predictions).
        """
        # Fast path: if the override fully specifies all weather measurements,
        # skip the external API call and build the snapshot directly.
        if override is not None and _override_is_complete(override):
            sunset_time = self._astro.get_sunset_time(lat, lon, target_date)
            sun_elev = self._astro.get_solar_elevation(lat, lon, sunset_time)
            return WeatherSnapshot(
                cloud_low=override.cloud_low,       # type: ignore[arg-type]
                cloud_mid=override.cloud_mid,       # type: ignore[arg-type]
                cloud_high=override.cloud_high,     # type: ignore[arg-type]
                cloud_total=override.cloud_total,   # type: ignore[arg-type]
                visibility_m=override.visibility_m, # type: ignore[arg-type]
                relative_humidity=override.relative_humidity,  # type: ignore[arg-type]
                dewpoint_c=override.dewpoint_c if override.dewpoint_c is not None else 10.0,
                temperature_c=override.temperature_c if override.temperature_c is not None else 15.0,
                precipitation_mm=override.precipitation_mm,  # type: ignore[arg-type]
                wind_speed_kmh=override.wind_speed_kmh if override.wind_speed_kmh is not None else 0.0,
                pressure_hpa=override.pressure_hpa if override.pressure_hpa is not None else 1013.0,
                aerosol_optical_depth=override.aerosol_optical_depth,
                sun_elevation_deg=sun_elev,
                data_source="override",
                aerosol_is_estimated=override.aerosol_optical_depth is None,
            )

        cache_key = TTLCache.make_key("snapshot", *self._ckey_coords(lat, lon), str(target_date))
        if override is None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for snapshot lat=%.4f lon=%.4f date=%s", lat, lon, target_date)
                return cached

        sunset_time = self._astro.get_sunset_time(lat, lon, target_date)
        today = datetime.now(UTC).date()
        days_ago = (today - target_date).days

        if target_date < today:
            if days_ago <= 7:
                # Use forecast + past_days for very recent dates — the archive
                # has a ~5-day lag so it may not have data yet.
                snapshot = await self._fetch_recent_past_snapshot(lat, lon, target_date, sunset_time, days_ago)
            else:
                snapshot = await self._fetch_archive_snapshot(lat, lon, target_date, sunset_time)
        else:
            snapshot = await self._fetch_forecast_snapshot(lat, lon, target_date, sunset_time)

        if override is not None:
            snapshot = self._apply_override(snapshot, override)

        if override is None:
            self._cache.set(cache_key, snapshot)

        return snapshot

    async def get_forecast_range(
        self,
        lat: float,
        lon: float,
        days: int,
        horizon_obstruction_deg: float = 2.0,
    ) -> list[tuple[date, WeatherSnapshot]]:
        """
        Return (date, WeatherSnapshot) pairs for the next *days* days.

        Uses a single Open-Meteo API call for all days, then slices per day.
        """
        cache_key = TTLCache.make_key("forecast_range", *self._ckey_coords(lat, lon), days)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        today = datetime.now(UTC).date()
        end_date = today + timedelta(days=days - 1)

        weather_data = await self._fetch_forecast_raw(lat, lon, days=days)
        aq_data = await self._fetch_air_quality_raw(lat, lon, days=days)

        results: list[tuple[date, WeatherSnapshot]] = []
        for offset in range(days):
            d = today + timedelta(days=offset)
            try:
                sunset_time = self._astro.get_sunset_time(lat, lon, d)
                snapshot = self._extract_snapshot_for_hour(
                    weather_data, aq_data, lat, lon, sunset_time
                )
                results.append((d, snapshot))
            except Exception as exc:
                logger.warning("Failed to build snapshot for %s: %s", d, exc)

        self._cache.set(cache_key, results)
        return results

    # ------------------------------------------------------------------
    # Archive (historical)
    # ------------------------------------------------------------------

    async def get_window_snapshots(
        self,
        lat: float,
        lon: float,
        target_date: date,
        sunset_time: datetime,
    ) -> list[WeatherSnapshot]:
        """
        Return four WeatherSnapshot objects covering the sunset viewing window:
          "-15m"  → sunset − 15 min
          "sunset" → exact sunset time
          "+15m"  → sunset + 15 min
          "+30m"  → sunset + 30 min

        All four share a single API fetch.  Because Open-Meteo provides hourly
        data, adjacent window points may resolve to the same hourly bucket —
        that is acceptable; the variation across the window still reflects real
        hourly changes when conditions are evolving.

        Trend fields (precipitation_last_3h_mm, pressure_trend_hpa_3h,
        cloud_total_trend_3h, visibility_trend_3h_m) are extracted from the
        3 hours prior to sunset and injected into every snapshot so the moisture
        scorer can detect post-rain clearing.

        Results are cached for the configured TTL to avoid redundant API calls
        and to keep the score stable within a single server session.
        """
        cache_key = TTLCache.make_key("window_snaps", *self._ckey_coords(lat, lon), str(target_date))
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for window_snaps lat=%.4f lon=%.4f date=%s", lat, lon, target_date)
            return cached

        today = datetime.now(UTC).date()
        days_ago = (today - target_date).days

        # Single raw fetch for all window points
        if target_date < today:
            if days_ago <= 7:
                weather_data = await self._fetch_forecast_raw(lat, lon, days=1, past_days=days_ago + 1)
                aq_data = await self._fetch_air_quality_raw(lat, lon, days=1, past_days=days_ago + 1)
            else:
                weather_data = await self._fetch_archive_raw(lat, lon, target_date)
                aq_data = None
        else:
            days_ahead = (target_date - today).days + 1
            weather_data = await self._fetch_forecast_raw(lat, lon, days=max(days_ahead + 1, 2))
            aq_data = await self._fetch_air_quality_raw(lat, lon, days=max(days_ahead + 1, 2))

        snapshots = self._extract_window_snapshots_from_raw(
            weather_data, aq_data, lat, lon, sunset_time
        )
        self._cache.set(cache_key, snapshots)
        return snapshots

    async def get_forecast_range_windows(
        self,
        lat: float,
        lon: float,
        days: int,
    ) -> list[tuple[date, list[WeatherSnapshot]]]:
        """
        Return (date, window_snapshots) pairs for the next *days* days.

        Uses a single Open-Meteo batch call for all days so the forecast
        endpoint makes the same number of API requests as before, while each
        day now gets window-level (4-point) scoring instead of a single snapshot.
        Results are cached for the configured TTL.
        """
        cache_key = TTLCache.make_key("forecast_range_windows", *self._ckey_coords(lat, lon), days)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        today = datetime.now(UTC).date()
        weather_data = await self._fetch_forecast_raw(lat, lon, days=days)
        aq_data = await self._fetch_air_quality_raw(lat, lon, days=days)

        # Pre-parse timestamps once so the per-day extraction loop doesn't
        # re-parse the same list on every call to _extract_snapshot_for_hour.
        _prepopulate_parsed_times(weather_data)
        if aq_data is not None:
            _prepopulate_parsed_times(aq_data)

        results: list[tuple[date, list[WeatherSnapshot]]] = []
        for offset in range(days):
            d = today + timedelta(days=offset)
            try:
                sunset_time = self._astro.get_sunset_time(lat, lon, d)
                window_snaps = self._extract_window_snapshots_from_raw(
                    weather_data, aq_data, lat, lon, sunset_time
                )
                results.append((d, window_snaps))
            except Exception as exc:
                logger.warning("Failed to build window snapshots for %s: %s", d, exc)

        self._cache.set(cache_key, results)
        return results

    # ------------------------------------------------------------------
    # Light corridor — upstream sampling along the sunset azimuth
    # ------------------------------------------------------------------

    async def get_corridor_samples(
        self,
        lat: float,
        lon: float,
        target_date: date,
        sunset_time: datetime,
    ) -> list[tuple[float, float, float]]:
        """Cloud cover upstream of the observer, toward the setting sun.

        Returns ``(distance_km, cloud_low_pct, cloud_mid_pct)`` for each sample
        point along the sunset azimuth. Empty list on any failure — the scoring
        engine treats that as "no corridor information" and leaves the score
        unadjusted, so a corridor outage degrades to the previous behaviour
        rather than breaking predictions.

        COST
        ----
        This is ONE HTTP request regardless of sample count: Open-Meteo accepts
        comma-separated coordinate lists (up to 1000 points). Results are cached
        on the same rounded-coordinate grid as every other fetch, so nearby
        users share it. Given the rate-limit sensitivity of this app, that
        one-request property is the reason the design is viable at all.

        The remote atmosphere is sampled at the OBSERVER's sunset instant, not
        at the remote location's own sunset — light arrives effectively
        instantly, so what matters is the state of the corridor at the moment
        the observer is looking.
        """
        cache_key = TTLCache.make_key(
            "corridor", *self._ckey_coords(lat, lon), str(target_date)
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            azimuth = self._astro.get_sunset_azimuth(lat, lon, target_date)
            points = [
                (d, *destination_point(lat, lon, azimuth, d))
                for d in CORRIDOR_DISTANCES_KM
            ]

            today = datetime.now(UTC).date()
            days_ago = (today - target_date).days
            lats = ",".join(f"{p[1]:.4f}" for p in points)
            lons = ",".join(f"{p[2]:.4f}" for p in points)

            if target_date < today and days_ago > 7:
                raw = await self._fetch_archive_raw_multi(lats, lons, target_date)
            else:
                past_days = days_ago + 1 if target_date < today else 0
                days_ahead = max((target_date - today).days + 2, 2)
                raw = await self._fetch_forecast_raw_multi(
                    lats, lons, days=days_ahead, past_days=past_days
                )

            samples: list[tuple[float, float, float]] = []
            # A multi-coordinate response is a LIST of per-location objects, in
            # request order; a single-coordinate response is a bare object.
            entries = raw if isinstance(raw, list) else [raw]
            for (distance_km, _plat, _plon), entry in zip(points, entries):
                cloud = self._extract_corridor_cloud(entry, sunset_time)
                if cloud is not None:
                    samples.append((distance_km, cloud[0], cloud[1]))

            if not samples:
                logger.debug("Corridor fetch returned no usable samples for %s", target_date)
                return []

            self._cache.set(cache_key, samples)
            return samples

        except Exception as exc:
            # Never let the corridor break a prediction — it is an enhancement
            # to the score, not a prerequisite for producing one.
            logger.warning(
                "Light-corridor sampling failed for lat=%.3f lon=%.3f date=%s: %s "
                "— scoring without it.",
                lat, lon, target_date, exc,
            )
            return []

    async def get_corridor_samples_map(
        self, lat: float, lon: float, dates: list[date]
    ) -> dict[date, list[tuple[float, float, float]]]:
        """Corridor samples for many dates, batched by calendar month.

        The sunset azimuth swings through ~60° over a year outside the tropics,
        so the corridor for January points somewhere quite different from July's
        and a single set of coordinates cannot serve a whole range. Grouping by
        month keeps the azimuth error under a couple of degrees — far below the
        angular width the Gaussian distance weighting already tolerates — while
        collapsing the request count from one-per-day to one-per-month.

        A 7-day forecast is therefore one request; a 12-month heatmap is twelve,
        each cached for a day. Any month that fails is simply absent from the
        result, and those days score without corridor adjustment.
        """
        if not dates:
            return {}

        by_month: dict[tuple[int, int], list[date]] = {}
        for d in dates:
            by_month.setdefault((d.year, d.month), []).append(d)

        out: dict[date, list[tuple[float, float, float]]] = {}
        for (year, month), group in sorted(by_month.items()):
            group.sort()
            cache_key = TTLCache.make_key(
                "corridor_month", *self._ckey_coords(lat, lon), year, month,
                str(group[0]), str(group[-1]),
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                out.update(cached)
                continue

            try:
                month_map = await self._fetch_corridor_month(lat, lon, group)
            except Exception as exc:
                logger.warning(
                    "Corridor batch failed for %04d-%02d at lat=%.3f lon=%.3f: %s "
                    "— those days score without it.",
                    year, month, lat, lon, exc,
                )
                continue

            # Archive months are immutable; give them a long TTL.
            is_past = group[-1] < datetime.now(UTC).date() - timedelta(days=8)
            self._cache.set(cache_key, month_map, ttl_override=86400 if is_past else None)
            out.update(month_map)

        return out

    async def _fetch_corridor_month(
        self, lat: float, lon: float, group: list[date]
    ) -> dict[date, list[tuple[float, float, float]]]:
        """One corridor request covering every date in *group* (same month)."""
        # Azimuth taken at the middle of the group so the error is symmetric
        # across it, rather than accumulating toward one end.
        mid = group[len(group) // 2]
        azimuth = self._astro.get_sunset_azimuth(lat, lon, mid)
        points = [
            (d_km, *destination_point(lat, lon, azimuth, d_km))
            for d_km in CORRIDOR_DISTANCES_KM
        ]
        lats = ",".join(f"{p[1]:.4f}" for p in points)
        lons = ",".join(f"{p[2]:.4f}" for p in points)

        today = datetime.now(UTC).date()
        start, end = group[0], group[-1]

        if end < today - timedelta(days=7):
            raw = await self._fetch_archive_range_raw_multi(lats, lons, start, end)
        else:
            past_days = max((today - start).days + 1, 0)
            days_ahead = max((end - today).days + 2, 2)
            raw = await self._fetch_forecast_raw_multi(
                lats, lons, days=days_ahead, past_days=min(past_days, 92)
            )

        entries = raw if isinstance(raw, list) else [raw]
        for entry in entries:
            _prepopulate_parsed_times(entry)

        result: dict[date, list[tuple[float, float, float]]] = {}
        for d in group:
            sunset_time = self._astro.get_sunset_time(lat, lon, d)
            samples: list[tuple[float, float, float]] = []
            for (distance_km, _plat, _plon), entry in zip(points, entries):
                cloud = self._extract_corridor_cloud(entry, sunset_time)
                if cloud is not None:
                    samples.append((distance_km, cloud[0], cloud[1]))
            if samples:
                result[d] = samples
        return result

    async def _fetch_archive_range_raw_multi(
        self, lats: str, lons: str, start_date: date, end_date: date
    ) -> Any:
        url = f"{self._settings.OPEN_METEO_ARCHIVE_URL}/archive"
        params = {
            "latitude": lats,
            "longitude": lons,
            "hourly": CORRIDOR_HOURLY_VARS,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "timezone": "UTC",
        }
        return await self._get_json(url, params)

    @staticmethod
    def _extract_corridor_cloud(
        entry: dict[str, Any], sunset_time: datetime
    ) -> Optional[tuple[float, float]]:
        """Pull (cloud_low, cloud_mid) at the sunset hour from one corridor point."""
        hourly = entry.get("hourly", {})
        time_strs: list[str] = hourly.get("time", [])
        if not time_strs:
            return None
        times = hourly.get("_times_parsed") or [
            datetime.fromisoformat(t).replace(tzinfo=UTC) for t in time_strs
        ]
        idx = min(range(len(times)), key=lambda i: abs((times[i] - sunset_time).total_seconds()))

        def get(key: str) -> float:
            vals = hourly.get(key, [])
            if idx < len(vals) and vals[idx] is not None:
                return float(vals[idx])
            return 0.0

        return get("cloud_cover_low"), get("cloud_cover_mid")

    async def _fetch_forecast_raw_multi(
        self, lats: str, lons: str, days: int, past_days: int = 0
    ) -> Any:
        url = f"{self._settings.OPEN_METEO_BASE_URL}/forecast"
        params: dict = {
            "latitude": lats,
            "longitude": lons,
            "hourly": CORRIDOR_HOURLY_VARS,
            "forecast_days": days,
            "timezone": "UTC",
        }
        if past_days > 0:
            params["past_days"] = past_days
        return await self._get_json(url, params)

    async def _fetch_archive_raw_multi(
        self, lats: str, lons: str, target_date: date
    ) -> Any:
        url = f"{self._settings.OPEN_METEO_ARCHIVE_URL}/archive"
        params = {
            "latitude": lats,
            "longitude": lons,
            "hourly": CORRIDOR_HOURLY_VARS,
            "start_date": str(target_date),
            "end_date": str(target_date),
            "timezone": "UTC",
        }
        return await self._get_json(url, params)

    async def get_historical_snapshot(
        self, lat: float, lon: float, target_date: date
    ) -> WeatherSnapshot:
        """Fetch a historical weather snapshot from the Open-Meteo archive."""
        sunset_time = self._astro.get_sunset_time(lat, lon, target_date)
        return await self._fetch_archive_snapshot(lat, lon, target_date, sunset_time)

    async def get_historical_range_windows(
        self,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
    ) -> list[tuple[date, list[WeatherSnapshot]]]:
        """
        Fetch 4-point window snapshots per day for [start_date, end_date].

        Mirrors predict()'s data-source split exactly so heatmap scores match:
          - days_ago > 7  → archive API (one bulk request for the whole range)
          - days_ago <= 7 → forecast API with past_days (same as get_window_snapshots)
        """
        cache_key = TTLCache.make_key("hist_range_windows", *self._ckey_coords(lat, lon), str(start_date), str(end_date))
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(
                "Cache hit for hist_range_windows lat=%.4f lon=%.4f %s..%s",
                lat, lon, start_date, end_date,
            )
            return cached

        today = datetime.now(UTC).date()
        # dates with days_ago > 7 are safely in the archive; ≤7 use forecast+past_days
        archive_boundary = today - timedelta(days=8)

        # One bulk archive fetch for the old portion
        archive_data: Optional[dict] = None
        if start_date <= archive_boundary:
            archive_end = min(end_date, archive_boundary)
            archive_data = await self._fetch_archive_range_raw(lat, lon, start_date, archive_end)

        # One forecast fetch covers all of the recent 7 days
        recent_weather: Optional[dict] = None
        recent_aq: Optional[dict] = None
        if end_date > archive_boundary:
            recent_weather = await self._fetch_forecast_raw(lat, lon, days=1, past_days=7)
            recent_aq = await self._fetch_air_quality_raw(lat, lon, days=1, past_days=7)

        # Pre-parse timestamps once so the per-day loop doesn't re-parse the
        # same 8760-entry list on every _extract_snapshot_for_hour / _extract_trends call.
        for _d in [archive_data, recent_weather, recent_aq]:
            if _d is not None:
                _prepopulate_parsed_times(_d)

        results: list[tuple[date, list[WeatherSnapshot]]] = []
        current = start_date
        while current <= end_date:
            try:
                days_ago = (today - current).days
                if days_ago <= 7:
                    weather_data, aq_data = recent_weather, recent_aq
                else:
                    weather_data, aq_data = archive_data, None

                if weather_data is None:
                    logger.warning("No weather data source available for %s, skipping", current)
                    current += timedelta(days=1)
                    continue

                sunset_time = self._astro.get_sunset_time(lat, lon, current)
                window_snaps = self._extract_window_snapshots_from_raw(
                    weather_data, aq_data, lat, lon, sunset_time
                )
                results.append((current, window_snaps))
            except Exception as exc:
                logger.warning("Failed to build window snapshots for %s: %s", current, exc)
            current += timedelta(days=1)

        # Archive data never changes; recent forecast data can be refreshed — use default TTL
        ttl = 86400 if end_date <= archive_boundary else None
        self._cache.set(cache_key, results, ttl_override=ttl)
        return results

    # ------------------------------------------------------------------
    # Internal: fetch helpers
    # ------------------------------------------------------------------

    async def _fetch_forecast_snapshot(
        self, lat: float, lon: float, target_date: date, sunset_time: datetime
    ) -> WeatherSnapshot:
        days_ahead = (target_date - datetime.now(UTC).date()).days + 1
        weather_data = await self._fetch_forecast_raw(lat, lon, days=max(days_ahead + 1, 2))
        aq_data = await self._fetch_air_quality_raw(lat, lon, days=max(days_ahead + 1, 2))
        return self._extract_snapshot_for_hour(weather_data, aq_data, lat, lon, sunset_time)

    async def _fetch_recent_past_snapshot(
        self, lat: float, lon: float, target_date: date, sunset_time: datetime, days_ago: int
    ) -> WeatherSnapshot:
        """Use the forecast endpoint with past_days for dates within the last 7 days.

        The archive API has a ~5-day lag; the forecast endpoint can serve past
        data immediately via the past_days parameter (max 92).
        """
        weather_data = await self._fetch_forecast_raw(lat, lon, days=1, past_days=days_ago + 1)
        aq_data = await self._fetch_air_quality_raw(lat, lon, days=1, past_days=days_ago + 1)
        return self._extract_snapshot_for_hour(weather_data, aq_data, lat, lon, sunset_time)

    async def _fetch_archive_snapshot(
        self, lat: float, lon: float, target_date: date, sunset_time: datetime
    ) -> WeatherSnapshot:
        weather_data = await self._fetch_archive_raw(lat, lon, target_date)
        return self._extract_snapshot_for_hour(weather_data, None, lat, lon, sunset_time)

    async def _fetch_forecast_raw(
        self, lat: float, lon: float, days: int = 7, past_days: int = 0
    ) -> dict[str, Any]:
        url = f"{self._settings.OPEN_METEO_BASE_URL}/forecast"
        params: dict = {
            "latitude": lat,
            "longitude": lon,
            "hourly": FORECAST_HOURLY_VARS,
            "forecast_days": days,
            "timezone": "UTC",
        }
        if past_days > 0:
            params["past_days"] = past_days
        return await self._get_json(url, params)

    async def _fetch_archive_raw(
        self, lat: float, lon: float, target_date: date
    ) -> dict[str, Any]:
        url = f"{self._settings.OPEN_METEO_ARCHIVE_URL}/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ARCHIVE_HOURLY_VARS,
            "start_date": str(target_date),
            "end_date": str(target_date),
            "timezone": "UTC",
        }
        return await self._get_json(url, params)

    async def _fetch_archive_range_raw(
        self,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        """Fetch a date range of archive data in one HTTP request."""
        url = f"{self._settings.OPEN_METEO_ARCHIVE_URL}/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ARCHIVE_HOURLY_VARS,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "timezone": "UTC",
        }
        return await self._get_json(url, params)

    async def _fetch_air_quality_raw(
        self, lat: float, lon: float, days: int = 7, past_days: int = 0
    ) -> Optional[dict[str, Any]]:
        """Fetch aerosol optical depth from the AQ API. Returns None on failure."""
        url = f"{self._settings.OPEN_METEO_AIR_QUALITY_URL}/air-quality"
        params: dict = {
            "latitude": lat,
            "longitude": lon,
            "hourly": AIR_QUALITY_HOURLY_VARS,
            "forecast_days": days,
            "timezone": "UTC",
        }
        if past_days > 0:
            params["past_days"] = past_days
        try:
            return await self._get_json(url, params)
        except Exception as exc:
            logger.debug("Air quality API unavailable: %s — using proxy", exc)
            return None

    async def _get_json(self, url: str, params: dict) -> Any:
        """Execute a GET request and return parsed JSON.

        Retries on transient failures — Open-Meteo rate-limits (HTTP 429),
        transient 5xx responses, and connection/timeout errors — using
        exponential backoff with jitter. When the response carries a
        ``Retry-After`` header it overrides the computed delay so we wait
        exactly as long as the provider asks.

        Non-retryable client errors (e.g. 400) fail fast. If retries are
        exhausted, raises :class:`WeatherUnavailableError` (→ HTTP 503).
        """
        max_retries = self._settings.HTTP_MAX_RETRIES
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._http.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in _RETRYABLE_STATUS and status < 500:
                    raise  # genuine client error — retrying won't help
                last_exc = exc
                if attempt >= max_retries:
                    break
                delay = self._retry_delay(exc.response, attempt)
                logger.warning(
                    "Open-Meteo %s for %s (attempt %d/%d) — retrying in %.1fs",
                    status, url, attempt + 1, max_retries + 1, delay,
                )
            except httpx.TransportError as exc:
                # Connection reset, timeout, DNS failure — transient.
                last_exc = exc
                if attempt >= max_retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Open-Meteo transport error for %s (attempt %d/%d): %s — retrying in %.1fs",
                    url, attempt + 1, max_retries + 1, exc, delay,
                )

            await asyncio.sleep(delay)

        raise WeatherUnavailableError(
            f"Weather provider unavailable after {max_retries + 1} attempt(s): {last_exc}"
        ) from last_exc

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Delay before the next retry.

        Honours an integer ``Retry-After`` header (seconds) when present,
        otherwise falls back to exponential backoff. Always capped at
        HTTP_MAX_RETRY_DELAY.
        """
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), self._settings.HTTP_MAX_RETRY_DELAY)
            except ValueError:
                pass  # HTTP-date form — fall back to computed backoff
        return self._backoff_delay(attempt)

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter, capped at HTTP_MAX_RETRY_DELAY."""
        base = self._settings.HTTP_BACKOFF_BASE
        delay = base * (2 ** attempt) + random.uniform(0.0, base)
        return min(delay, self._settings.HTTP_MAX_RETRY_DELAY)

    # ------------------------------------------------------------------
    # Internal: data extraction
    # ------------------------------------------------------------------

    def _extract_snapshot_for_hour(
        self,
        weather_data: dict[str, Any],
        aq_data: Optional[dict[str, Any]],
        lat: float,
        lon: float,
        sunset_time: datetime,
    ) -> WeatherSnapshot:
        """
        Find the hourly row closest to *sunset_time* and build a WeatherSnapshot.
        """
        hourly = weather_data.get("hourly", {})
        time_strs: list[str] = hourly.get("time", [])

        if not time_strs:
            raise ValueError("No hourly time data in Open-Meteo response")

        # Use pre-parsed list when available (populated by _prepopulate_parsed_times
        # in batch callers); fall back to parsing on first use for single-snapshot paths.
        times: list[datetime] = hourly.get("_times_parsed") or [
            datetime.fromisoformat(t).replace(tzinfo=UTC) for t in time_strs
        ]

        # Find index of the hour nearest to sunset
        idx = min(range(len(times)), key=lambda i: abs((times[i] - sunset_time).total_seconds()))

        def get(key: str, default: float = 0.0) -> float:
            values = hourly.get(key, [])
            if idx < len(values) and values[idx] is not None:
                return float(values[idx])
            return default

        cloud_low = get("cloud_cover_low", 0.0)
        cloud_mid = get("cloud_cover_mid", 0.0)
        cloud_high = get("cloud_cover_high", 0.0)
        cloud_total = get("cloud_cover", max(cloud_low, cloud_mid, cloud_high))
        # Archive API often omits visibility; 15 km is a neutral "hazy-but-clear"
        # baseline.  24 km (the previous default) was a pristine clear-sky value
        # that systematically over-rewarded every archive day.
        visibility_m = get("visibility", 15000.0)
        humidity = get("relative_humidity_2m", 50.0)
        dewpoint = get("dew_point_2m", 10.0)
        temperature = get("temperature_2m", 15.0)
        precipitation = get("precipitation", 0.0)
        wind_speed = get("wind_speed_10m", 0.0)
        pressure = get("surface_pressure", 1013.0)

        # Aerosol optical depth from AQ API
        aerosol_od: Optional[float] = None
        aerosol_is_estimated = False

        if aq_data is not None:
            aq_hourly = aq_data.get("hourly", {})
            aq_times_raw: list[str] = aq_hourly.get("time", [])
            if aq_times_raw:
                aq_times = [
                    datetime.fromisoformat(t).replace(tzinfo=UTC) for t in aq_times_raw
                ]
                aq_idx = min(
                    range(len(aq_times)),
                    key=lambda i: abs((aq_times[i] - sunset_time).total_seconds()),
                )
                aod_vals = aq_hourly.get("aerosol_optical_depth", [])
                if aq_idx < len(aod_vals) and aod_vals[aq_idx] is not None:
                    aerosol_od = float(aod_vals[aq_idx])

        if aerosol_od is None:
            # Proxy estimation from visibility and humidity.
            # NOTE: This is a rough approximation. A clear atmosphere (high visibility,
            # low humidity) suggests low AOD; hazy conditions suggest higher AOD.
            # Values are calibrated against typical real-world AOD ranges (0.05–0.6).
            vis_km = visibility_m / 1000.0
            aerosol_od = max(0.05, min(0.8, (1.0 - vis_km / 40.0) * 0.4 + humidity / 100.0 * 0.15))
            aerosol_is_estimated = True

        # Solar elevation at the ACTUAL requested time (sunset_time), not at the
        # snapped hourly bucket.  This matters for window snapshots: the bucket
        # for "+15m after sunset" may be the same hour as "sunset", giving
        # both the same (wrong) sun elevation.  Using the real target time
        # ensures each window point gets the correct elevation for the
        # afterglow bell-curve calculation.
        sun_elev = self._astro.get_solar_elevation(lat, lon, sunset_time)

        return WeatherSnapshot(
            cloud_low=cloud_low,
            cloud_mid=cloud_mid,
            cloud_high=cloud_high,
            cloud_total=cloud_total,
            visibility_m=visibility_m,
            relative_humidity=humidity,
            dewpoint_c=dewpoint,
            temperature_c=temperature,
            precipitation_mm=precipitation,
            wind_speed_kmh=wind_speed,
            pressure_hpa=pressure,
            aerosol_optical_depth=aerosol_od,
            sun_elevation_deg=sun_elev,
            data_source="archive" if "archive" in str(weather_data.get("generationtime_ms", "")) else "forecast",
            aerosol_is_estimated=aerosol_is_estimated,
        )

    def _extract_window_snapshots_from_raw(
        self,
        weather_data: dict,
        aq_data: Optional[dict],
        lat: float,
        lon: float,
        sunset_time: datetime,
    ) -> list[WeatherSnapshot]:
        """
        Build four window snapshots from already-fetched raw API data.

        Shared by get_window_snapshots() (single-day predict) and
        get_forecast_range_windows() (multi-day forecast) so both paths
        use identical extraction logic.
        """
        trends = self._extract_trends(weather_data, sunset_time)

        window_offsets: list[tuple[str, timedelta]] = [
            ("-15m",   timedelta(minutes=-15)),
            ("sunset", timedelta(minutes=0)),
            ("+15m",   timedelta(minutes=15)),
            ("+30m",   timedelta(minutes=30)),
        ]

        snapshots: list[WeatherSnapshot] = []
        for label, offset in window_offsets:
            target_time = sunset_time + offset
            snap = self._extract_snapshot_for_hour(weather_data, aq_data, lat, lon, target_time)
            snap_data = snap.model_dump()
            snap_data.update(trends)
            snap_data["timestamp_label"] = label
            snapshots.append(WeatherSnapshot(**snap_data))

        return snapshots

    def _extract_trends(
        self, weather_data: dict, sunset_time: datetime
    ) -> dict:
        """
        Compute 3-hour trend fields from hourly data prior to sunset.

        Looks at the 3 hours immediately before the sunset hour and computes:
        - precipitation_last_3h_mm  : total precip in those 3 hours
        - pressure_trend_hpa_3h     : pressure[sunset] − pressure[sunset−3h]
        - cloud_total_trend_3h      : total cloud[sunset] − cloud[sunset−3h]
        - visibility_trend_3h_m     : visibility[sunset] − visibility[sunset−3h]

        Returns an empty dict when no hourly data is available (archive fallback).
        All fields are optional in WeatherSnapshot so missing is safe.
        """
        hourly = weather_data.get("hourly", {})
        time_strs: list[str] = hourly.get("time", [])
        if not time_strs:
            return {}

        times = hourly.get("_times_parsed") or [
            datetime.fromisoformat(t).replace(tzinfo=UTC) for t in time_strs
        ]
        sunset_idx = min(range(len(times)), key=lambda i: abs((times[i] - sunset_time).total_seconds()))
        past_idx = max(0, sunset_idx - 3)

        def get(key: str, idx: int, default: float = 0.0) -> float:
            vals = hourly.get(key, [])
            if idx < len(vals) and vals[idx] is not None:
                return float(vals[idx])
            return default

        precip_sum = sum(get("precipitation", i) for i in range(past_idx, sunset_idx))
        pressure_trend = get("surface_pressure", sunset_idx) - get("surface_pressure", past_idx)
        cloud_trend = get("cloud_cover", sunset_idx) - get("cloud_cover", past_idx)
        vis_trend = get("visibility", sunset_idx) - get("visibility", past_idx)

        return {
            "precipitation_last_3h_mm": round(precip_sum, 2),
            "pressure_trend_hpa_3h": round(pressure_trend, 1),
            "cloud_total_trend_3h": round(cloud_trend, 1),
            "visibility_trend_3h_m": round(vis_trend, 0),
        }

    # ------------------------------------------------------------------
    # Override application
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_override(
        snapshot: WeatherSnapshot, override: WeatherOverride
    ) -> WeatherSnapshot:
        """Return a new snapshot with override fields applied."""
        data = snapshot.model_dump()
        for field, val in override.model_dump(exclude_none=True).items():
            data[field] = val
        # If override changed aerosol, it's no longer estimated
        if override.aerosol_optical_depth is not None:
            data["aerosol_is_estimated"] = False
        data["data_source"] = "override"
        return WeatherSnapshot(**data)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# The minimum set of fields that must be provided for a "complete" override
# (i.e. no API call needed).
_REQUIRED_OVERRIDE_FIELDS = {
    "cloud_low", "cloud_mid", "cloud_high", "cloud_total",
    "visibility_m", "relative_humidity", "precipitation_mm",
}


def _prepopulate_parsed_times(data: dict) -> None:
    """
    Parse the hourly time strings in *data* once and store the result under
    the ``_times_parsed`` key so repeated calls to ``_extract_snapshot_for_hour``
    and ``_extract_trends`` on the same raw dict skip re-parsing.

    Mutates *data* in-place; safe because the dict is local to a single request.
    """
    hourly = data.get("hourly", {})
    if hourly.get("time") and "_times_parsed" not in hourly:
        hourly["_times_parsed"] = [
            datetime.fromisoformat(t).replace(tzinfo=UTC) for t in hourly["time"]
        ]


def _override_is_complete(override: WeatherOverride) -> bool:
    """
    Return True if the override supplies all the fields needed to build a
    WeatherSnapshot without fetching from the weather API.
    """
    provided = {k for k, v in override.model_dump().items() if v is not None}
    return _REQUIRED_OVERRIDE_FIELDS.issubset(provided)
