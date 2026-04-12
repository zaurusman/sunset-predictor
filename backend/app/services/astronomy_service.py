"""Astronomy helpers: sunset time, solar geometry via the astral library."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from astral import Observer
from astral.sun import sun, elevation as solar_elevation

from app.core.logging import get_logger

logger = get_logger(__name__)

UTC = timezone.utc


class AstronomyService:
    """
    Provides sunset time and solar geometry for a given location and date.

    All returned datetimes are UTC-aware.
    """

    # Best viewing window offsets around sunset
    _WINDOW_START_OFFSET = timedelta(minutes=-10)
    _WINDOW_END_OFFSET = timedelta(minutes=25)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sun_times(self, lat: float, lon: float, target_date: date) -> dict[str, datetime]:
        """
        Return a dictionary of solar event times (UTC) for the given date.

        Keys: dawn, sunrise, noon, sunset, dusk
        """
        observer = Observer(latitude=lat, longitude=lon)
        try:
            times = sun(observer, date=target_date, tzinfo=UTC)
            return dict(times)
        except Exception as exc:
            # astral raises ValueError for locations with no sunset (polar day/night)
            logger.warning(
                "sun() failed for lat=%.4f lon=%.4f date=%s: %s — falling back to estimated times",
                lat, lon, target_date, exc,
            )
            return self._estimate_sun_times(lat, lon, target_date)

    def get_sunset_time(self, lat: float, lon: float, target_date: date) -> datetime:
        """Return the UTC datetime of sunset for the given date."""
        times = self.get_sun_times(lat, lon, target_date)
        return times["sunset"]

    def get_solar_elevation(self, lat: float, lon: float, dt: datetime) -> float:
        """Return solar elevation angle in degrees for the given UTC datetime."""
        observer = Observer(latitude=lat, longitude=lon)
        try:
            return solar_elevation(observer, dateandtime=dt)
        except Exception:
            return 0.0

    def get_best_viewing_window(
        self, sunset_time: datetime
    ) -> tuple[datetime, datetime]:
        """
        Return the recommended viewing window (start, end) around sunset.

        Default: 10 minutes before sunset to 25 minutes after sunset.
        This window is informational — it does NOT factor into the beauty score.
        """
        start = sunset_time + self._WINDOW_START_OFFSET
        end = sunset_time + self._WINDOW_END_OFFSET
        return start, end

    def get_sunset_utc_hour(self, lat: float, lon: float, target_date: date) -> int:
        """
        Return the UTC hour (0–23) of the sunset.

        Used to select the correct hourly weather row from Open-Meteo.
        """
        sunset = self.get_sunset_time(lat, lon, target_date)
        return sunset.hour

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _estimate_sun_times(
        self, lat: float, lon: float, target_date: date
    ) -> dict[str, datetime]:
        """
        Very rough fallback for polar regions where astral fails.

        Sets sunset to 18:00 UTC (arbitrary but functional for weather lookup).
        """
        base = datetime(target_date.year, target_date.month, target_date.day, tzinfo=UTC)
        return {
            "dawn": base.replace(hour=5),
            "sunrise": base.replace(hour=6),
            "noon": base.replace(hour=12),
            "sunset": base.replace(hour=18),
            "dusk": base.replace(hour=19),
        }
