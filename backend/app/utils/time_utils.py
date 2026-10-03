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


# How far a phone's clock may sit from the longitude timezone above: summer
# time, and zones set well east of the sun (Spain, western China).
_CLOCK_SLACK = timedelta(hours=3)


def tonight_dates(lat: float, lon: float) -> set[date]:
    """Every date a client may mean by "tonight" at these coordinates.

    The app sends the phone's own date, which can turn over a few hours before
    or after the longitude timezone does (Israel's summer time is an hour
    ahead of it), and the 7-day page's first day (see first_forecast_date).
    Never the UTC date as such: out west that is tomorrow's evening for part
    of every day.
    """
    tz = get_timezone_for_coordinates(lat, lon)
    now = datetime.now(tz=tz)
    return {
        (now - _CLOCK_SLACK).date(),
        (now + _CLOCK_SLACK).date(),
        first_forecast_date(lat, lon),
    }


def first_forecast_date(lat: float, lon: float) -> date:
    """The first evening /forecast lists: the location's date, or the UTC
    date when that is earlier.

    East of Greenwich, between local and UTC midnight, that is the evening
    just gone, as /forecast has always started there (the app drops past
    days). Out west it is tonight; the UTC date would be tomorrow.
    """
    return min(local_sunset_date(lat, lon), datetime.now(tz=UTC).date())
