"""Weather data service: fetches and normalises Open-Meteo API data."""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.weather import WeatherOverride, WeatherSnapshot
from app.services.astronomy_service import AstronomyService
from app.utils.cache import TTLCache

logger = get_logger(__name__)
UTC = timezone.utc

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

        cache_key = TTLCache.make_key("snapshot", lat, lon, str(target_date))
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
        cache_key = TTLCache.make_key("forecast_range", lat, lon, days)
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
        cache_key = TTLCache.make_key("window_snaps", lat, lon, str(target_date))
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
        cache_key = TTLCache.make_key("forecast_range_windows", lat, lon, days)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        today = datetime.now(UTC).date()
        weather_data = await self._fetch_forecast_raw(lat, lon, days=days)
        aq_data = await self._fetch_air_quality_raw(lat, lon, days=days)

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

    async def get_historical_snapshot(
        self, lat: float, lon: float, target_date: date
    ) -> WeatherSnapshot:
        """Fetch a historical weather snapshot from the Open-Meteo archive."""
        sunset_time = self._astro.get_sunset_time(lat, lon, target_date)
        return await self._fetch_archive_snapshot(lat, lon, target_date, sunset_time)

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

    async def _get_json(self, url: str, params: dict) -> dict[str, Any]:
        """Execute a GET request and return parsed JSON."""
        response = await self._http.get(url, params=params)
        response.raise_for_status()
        return response.json()

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

        # Parse all timestamps (they come as naive UTC strings from API)
        times = [
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
        visibility_m = get("visibility", 10000.0)
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

        # Solar elevation at the sunset hour (informational)
        sun_elev = self._astro.get_solar_elevation(lat, lon, times[idx])

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

        times = [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in time_strs]
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


def _override_is_complete(override: WeatherOverride) -> bool:
    """
    Return True if the override supplies all the fields needed to build a
    WeatherSnapshot without fetching from the weather API.
    """
    provided = {k for k, v in override.model_dump().items() if v is not None}
    return _REQUIRED_OVERRIDE_FIELDS.issubset(provided)
