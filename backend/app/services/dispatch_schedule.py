"""Works out WHEN a dispatch is actually needed, so nothing else has to guess.

Polling blindly is expensive in a way that is easy to miss: Render's free tier
spins a service down after 15 minutes idle, so every wake-up costs ~15 instance
-minutes whether or not there was anything to do. A ping every 20 minutes
therefore holds the service up ~18 hours a day — around 540 of the 750 free
instance-hours a month, spent before a single real visitor arrives.

But sunset times are known in advance. For each subscriber the window is
[sunset - lead, sunset], so the set of clock hours in which ANY window is open
is computable a day ahead. The scheduler can then skip every other hour and
never touch the service at all.

The output is deliberately coarse — whole UTC hours, not exact minutes. A
window is at least MIN_LEAD_MINUTES wide, so an hourly check cannot fall
through one, and hour-granularity keeps the schedule stable enough to commit.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Iterable

from app.core.logging import get_logger
from app.services.astronomy_service import AstronomyService
from app.utils.time_utils import utcnow

logger = get_logger(__name__)

# Floor on how early an alert may be requested. This is what makes an HOURLY
# check safe: a window narrower than 60 minutes could open and close between
# two checks, and the alert would silently never fire.
MIN_LEAD_MINUTES = 60

# Days ahead to cover. Two, so a schedule computed just before midnight still
# describes tomorrow evening.
HORIZON_DAYS = 2


def _windows(
    records: Iterable[dict[str, Any]],
    astro: AstronomyService,
    start: datetime,
    horizon_days: int = HORIZON_DAYS,
) -> list[tuple[datetime, datetime]]:
    """Every [window_open, sunset] interval across the horizon."""
    out: list[tuple[datetime, datetime]] = []

    for record in records:
        try:
            lat = float(record["latitude"])
            lon = float(record["longitude"])
        except (KeyError, TypeError, ValueError):
            # A malformed record must not cost every other subscriber their
            # schedule; the dispatcher logs it properly when it runs.
            continue

        lead = max(MIN_LEAD_MINUTES, int(record.get("lead_minutes", 120)))

        for offset in range(horizon_days + 1):
            day: date = (start + timedelta(days=offset)).date()
            try:
                sunset = astro.get_sunset_time(lat, lon, day)
            except Exception:
                continue
            open_at = sunset - timedelta(minutes=lead)
            if sunset < start:
                continue  # already over
            out.append((open_at, sunset))

    return out


def compute_schedule(
    records: Iterable[dict[str, Any]],
    astro: AstronomyService,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Describe when dispatch needs to run.

    Returns the UTC hours to cover, the next moment a window opens, and the
    subscriber count. With no subscribers every field is empty — and the
    scheduler then never wakes the service at all, which is the correct cost
    for a feature nobody is using.
    """
    now = now or utcnow()
    records = list(records)
    windows = _windows(records, astro, now)

    hours: set[int] = set()
    for open_at, close_at in windows:
        # Walk the window hour by hour and mark every hour it touches. Marking
        # only the opening hour would miss a window that starts at 16:55 and
        # runs to 18:55 — the 17:00 and 18:00 checks are the ones that fire.
        cursor = open_at.replace(minute=0, second=0, microsecond=0)
        while cursor <= close_at:
            hours.add(cursor.hour)
            cursor += timedelta(hours=1)

    upcoming = sorted(open_at for open_at, _ in windows if open_at >= now)

    return {
        "subscriber_count": len(records),
        "cron_hours": sorted(hours),
        "next_window_opens": upcoming[0].isoformat() if upcoming else None,
        "computed_at": now.isoformat(),
        # Whole hours, so a consumer can gate on the current hour with no
        # date handling and no timezone reasoning of its own.
        "cron_expression": (
            f"0 {','.join(str(h) for h in sorted(hours))} * * *" if hours else None
        ),
    }
