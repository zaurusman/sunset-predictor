"""Internal weather data schemas."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class WeatherSnapshot(BaseModel):
    """
    Normalised weather observation or forecast snapshot at (approximately) sunset time.

    All fields reflect conditions around the local sunset hour.
    """

    # Cloud cover components (0–100 %)
    cloud_low: float = Field(ge=0, le=100, description="Low cloud cover %")
    cloud_mid: float = Field(ge=0, le=100, description="Mid cloud cover %")
    cloud_high: float = Field(ge=0, le=100, description="High cloud cover %")
    cloud_total: float = Field(ge=0, le=100, description="Total cloud cover %")

    # Atmosphere
    #
    # visibility_m is None when the data source does not report it. The ERA5
    # archive never does — measured over 72 consecutive archive hours it
    # returned null every time — so historical days genuinely have no
    # visibility, and the previous 15 km default silently fed a constant into
    # the atmosphere score for every archive day. None means "unknown"; the
    # scorer leaves it out rather than inventing a value.
    visibility_m: Optional[float] = Field(
        default=None, ge=0, description="Horizontal visibility in metres, or None if unreported"
    )
    relative_humidity: float = Field(ge=0, le=100, description="Relative humidity %")
    dewpoint_c: float = Field(description="Dew point temperature °C")
    temperature_c: float = Field(description="Air temperature °C")
    precipitation_mm: float = Field(ge=0, description="Precipitation in mm")
    wind_speed_kmh: float = Field(ge=0, description="Wind speed km/h")
    pressure_hpa: float = Field(description="Surface pressure hPa")

    # Total column integrated water vapour (kg/m², equivalently mm of
    # precipitable water). This is the whole-atmosphere moisture load, which is
    # what actually mutes sunset colour — surface relative humidity says only
    # what the bottom few metres are doing. Available on BOTH the forecast and
    # archive endpoints, which is why it, and not pressure-level humidity, is
    # what the moisture component scores: the archive returns null for every
    # relative_humidity_*hPa level, so scoring those would make live forecasts
    # incomparable with the climatology they are ranked against.
    tcwv_kg_m2: Optional[float] = Field(
        default=None, ge=0, description="Total column integrated water vapour (kg/m²)"
    )

    # Aerosol optical depth — None means unavailable (fallback proxy was used)
    aerosol_optical_depth: Optional[float] = Field(
        default=None, description="Aerosol optical depth at 550 nm (0–5 scale)"
    )

    # Solar geometry at the sunset hour (informational; NOT used in scoring)
    sun_elevation_deg: float = Field(
        description="Solar elevation angle at sunset (degrees above horizon)"
    )

    # Provenance
    data_source: str = Field(
        description="'forecast' | 'archive' | 'override'",
        default="forecast",
    )
    aerosol_is_estimated: bool = Field(
        default=False,
        description="True when aerosol_optical_depth was estimated from visibility/humidity proxy",
    )

    # ---------------------------------------------------------------------------
    # Optional trend fields — populated by the weather service when hourly history
    # is available (forecast and recent-past paths only; None for archive).
    # Used by the moisture scorer to detect post-rain clearing.
    # ---------------------------------------------------------------------------
    precipitation_last_3h_mm: Optional[float] = Field(
        default=None, ge=0,
        description="Total precipitation in the 3 hours prior to sunset (mm)",
    )
    pressure_trend_hpa_3h: Optional[float] = Field(
        default=None,
        description="Surface pressure change over the 3 hours prior to sunset (hPa; positive = rising)",
    )
    cloud_total_trend_3h: Optional[float] = Field(
        default=None,
        description="Total cloud cover change over the 3 hours prior to sunset (%; negative = clearing)",
    )
    visibility_trend_3h_m: Optional[float] = Field(
        default=None,
        description="Visibility change over the 3 hours prior to sunset (m; positive = improving)",
    )

    # Window label: set when this snapshot is one of several window points
    timestamp_label: Optional[str] = Field(
        default=None,
        description="Window position label: '-15m' | 'sunset' | '+15m' | '+30m'",
    )


class WeatherOverride(BaseModel):
    """
    Optional manual override for all weather fields.

    Any field left as None will be filled from the actual forecast.
    Useful for testing the scoring engine with controlled inputs.
    """

    cloud_low: Optional[float] = Field(default=None, ge=0, le=100)
    cloud_mid: Optional[float] = Field(default=None, ge=0, le=100)
    cloud_high: Optional[float] = Field(default=None, ge=0, le=100)
    cloud_total: Optional[float] = Field(default=None, ge=0, le=100)
    visibility_m: Optional[float] = Field(default=None, ge=0)
    relative_humidity: Optional[float] = Field(default=None, ge=0, le=100)
    dewpoint_c: Optional[float] = None
    temperature_c: Optional[float] = None
    precipitation_mm: Optional[float] = Field(default=None, ge=0)
    wind_speed_kmh: Optional[float] = Field(default=None, ge=0)
    pressure_hpa: Optional[float] = None
    aerosol_optical_depth: Optional[float] = Field(default=None, ge=0)
