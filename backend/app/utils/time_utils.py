"""Date / time helpers. All internal datetimes are UTC-aware."""
from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


UTC = timezone.utc


def utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=UTC)


def to_utc(dt: datetime) -> datetime:
    """Convert a datetime to UTC. Naive datetimes are assumed to be UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def date_to_utc_datetime(d: date) -> datetime:
    """Convert a date to midnight UTC datetime."""
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def format_iso(dt: datetime) -> str:
    """Return ISO-8601 string in UTC, e.g. '2024-06-21T19:45:00Z'."""
    return to_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def nearest_hour(dt: datetime) -> datetime:
    """Round a datetime to the nearest hour."""
    half = timedelta(minutes=30)
    return (dt + half).replace(minute=0, second=0, microsecond=0)


def get_timezone_for_coordinates(lat: float, lon: float) -> ZoneInfo:
    """
    Return a rough timezone for the given coordinates.

    This is a lightweight approximation using longitude offset.
    For production use, integrate `timezonefinder` package instead.
    """
    offset_hours = round(lon / 15)
    offset_hours = max(-12, min(14, offset_hours))
    # Try well-known tz names first
    utc_label = (
        "UTC"
        if offset_hours == 0
        else f"Etc/GMT{'-' if offset_hours > 0 else '+'}{abs(offset_hours)}"
    )
    try:
        return ZoneInfo(utc_label)
    except (ZoneInfoNotFoundError, Exception):
        return ZoneInfo("UTC")


def local_sunset_date(lat: float, lon: float) -> date:
    """
    Return today's date in the approximate local timezone for the coordinates.

    Used to default the prediction target when no date is provided.
    """
    tz = get_timezone_for_coordinates(lat, lon)
    return datetime.now(tz=tz).date()
