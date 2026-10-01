"""Open-Meteo call budget: per-client and process-wide.

The free API allows 10,000 weighted calls a day, and Render's outbound IP is
shared with other tenants. Nothing stopped one script from spending all of it:
about 60 /heatmap calls at fresh coordinates (~164 weighted calls each) would
use the day's quota and break predictions for everyone (scripts/capacity).

Accounting happens per Open-Meteo call (WeatherService._get_json charges it),
attributed to the client whose request caused it via a context variable, so a
background climatology build counts against the user who triggered it.

Decisions happen only at the START of a request, never mid-way: refusing one
call halfway through a prediction would let the corridor or aerosol fetch fail
and the request quietly score without it. Cached requests cost nothing, but
are refused too once a client is over its limit.

Tonight first (see TONIGHT / OTHER below): everything except tonight's
prediction is held to a share of Open-Meteo's per-minute, per-hour and per-day
limits, so it can never spend what tonight needs. Over the minute share it
WAITS (pacing, so a burst of heatmaps loads slower instead of failing); over
the hour/day share new work is refused up front (optional_work_allowed).
"""
from __future__ import annotations

import asyncio
import contextvars
import math
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Iterator, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# Who the current request is for (None outside a request: alert runs, scripts).
current_client: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_client", default=None
)

_MINUTE = 60.0
_HOUR = 3600.0
_DAY = 86_400.0

# What an Open-Meteo call is for. TONIGHT: tonight's prediction, and the
# shared fetches tonight reads (the 7-day weather/aerosol bundle, tonight's
# corridor, the ensemble), whoever triggers them. OTHER: everything else —
# the 7-day page's later days, other dates, history, climatology. OTHER work
# is paced and capped below Open-Meteo's limits, so it can never use up the
# share kept for tonight; TONIGHT is never held back by this module.
TONIGHT = "tonight"
OTHER = "other"
call_priority: contextvars.ContextVar[str] = contextvars.ContextVar("call_priority", default=OTHER)


@contextmanager
def priority(level: str) -> Iterator[None]:
    """Run the enclosed calls (and tasks created inside) at *level*."""
    token = call_priority.set(level)
    try:
        yield
    finally:
        call_priority.reset(token)


# Epoch seconds by which the current request's non-tonight calls must have
# been admitted (set per request by CallBudgetMiddleware). One deadline for
# the whole request, not per call: a heatmap fetches in rounds, and waiting
# a full minute in each would outlast Cloudflare's ~100 s request timeout.
request_deadline: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "request_deadline", default=None
)


@contextmanager
def background_work() -> Iterator[None]:
    """For tasks started from a request that outlive it (climatology builds):
    non-tonight priority, and no request deadline."""
    token = call_priority.set(OTHER)
    deadline = request_deadline.set(None)
    try:
        yield
    finally:
        request_deadline.reset(deadline)
        call_priority.reset(token)


class BudgetExhausted(Exception):
    """OTHER work refused: its share of Open-Meteo's limits is used up."""


def weighted_cost(params: dict) -> float:
    """Open-Meteo's own weighting: each location counts, and a request is
    multiple calls beyond 10 variables or 2 weeks of data."""
    n_locations = len(str(params.get("latitude", "0")).split(","))
    n_vars = sum(
        len(str(params[k]).split(",")) for k in ("hourly", "daily", "current") if k in params
    )
    if "start_date" in params and "end_date" in params:
        days = (date.fromisoformat(str(params["end_date"]))
                - date.fromisoformat(str(params["start_date"]))).days + 1
    else:
        days = int(params.get("forecast_days", 7)) + int(params.get("past_days", 0))
    return n_locations * max(1.0, n_vars / 10) * max(1.0, days / 14)


class CallBudget:
    """Rolling-window counters. A limit of 0 disables that check."""

    def __init__(
        self,
        client_hourly_limit: float,
        daily_soft_cap: float,
        clock=time.time,
        other_minute_cap: float = 0,
        other_hour_cap: float = 0,
        other_max_wait: float = 85.0,
        sleep=asyncio.sleep,
    ) -> None:
        self._client_limit = client_hourly_limit
        self._soft_cap = daily_soft_cap
        self._clock = clock
        self._sleep = sleep
        # OTHER work's share of Open-Meteo's per-minute and per-hour limits
        # (0 = no cap). The remainder is kept for TONIGHT.
        self._other_minute_cap = other_minute_cap
        self._other_hour_cap = other_hour_cap
        self._other_max_wait = other_max_wait
        self._minute: deque[tuple[float, float]] = deque()
        self._minute_total = 0.0
        self._hour: deque[tuple[float, float]] = deque()
        self._hour_total = 0.0
        # Set when Open-Meteo itself says a limit is reached — it also counts
        # other tenants on the shared outbound IP, which we can't see.
        self._other_wait_until = 0.0     # minutely: OTHER waits
        self._other_refuse_until = 0.0   # hourly/daily: OTHER refused
        self._lock = threading.Lock()
        self._clients: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
        self._client_totals: dict[str, float] = defaultdict(float)
        self._day: deque[tuple[float, float]] = deque()
        self._day_total = 0.0
        self._soft_cap_logged = False
        self._charges = 0

    def charge(self, weight: float, client: Optional[str] = None) -> None:
        """Record one Open-Meteo call (never refuses — see module docstring)."""
        client = client if client is not None else current_client.get()
        now = self._clock()
        with self._lock:
            self._charge_locked(now, weight, client)

    def _charge_locked(self, now: float, weight: float, client: Optional[str]) -> None:
        """Record one call; the caller holds self._lock."""
        self._record(now, weight)
        if client is not None:
            self._clients[client].append((now, weight))
            self._client_totals[client] += weight
        self._expire_day(now)
        self._charges += 1
        if self._charges % 500 == 0:  # drop clients idle for an hour
            for c in list(self._clients):
                self._expire_client(c, now)
        if self._soft_cap and self._day_total >= self._soft_cap and not self._soft_cap_logged:
            self._soft_cap_logged = True
            logger.warning(
                "Open-Meteo daily soft cap reached (%.0f weighted calls in 24 h): pausing "
                "the 7-day forecast, other dates, heatmaps and climatology builds; tonight continues.", self._day_total,
            )

    def client_retry_after(self, client: str) -> Optional[int]:
        """Seconds until *client* is back under its hourly limit, or None if it is."""
        if not self._client_limit:
            return None
        now = self._clock()
        with self._lock:
            self._expire_client(client, now)
            total = self._client_totals.get(client, 0.0)
            if total < self._client_limit:
                return None
            # Wait for enough of the oldest calls to age out.
            excess = total - self._client_limit
            freed = 0.0
            for t, w in self._clients[client]:
                freed += w
                if freed > excess:
                    return max(1, math.ceil(t + _HOUR - now))
            return int(_HOUR)

    def optional_work_allowed(self, headroom: float = 0.0) -> bool:
        """Whether new OTHER work (a 7-day forecast, another date, a heatmap,
        a climatology build) may start: False once the 24 h total passes the
        soft cap, the hour's total passes OTHER's hourly share, or Open-Meteo
        has said its hourly/daily limit is reached. *headroom* is what the
        work may cost, so it is refused up front rather than half-way."""
        now = self._clock()
        with self._lock:
            self._expire(now)
            if now < self._other_refuse_until:
                return False
            if self._other_hour_cap and self._hour_total + headroom >= self._other_hour_cap:
                return False
            if not self._soft_cap:
                return True
            if self._day_total + headroom < self._soft_cap:
                self._soft_cap_logged = False
                return True
            return False

    async def acquire_other(self, weight: float) -> None:
        """Admit and record one OTHER call, waiting while the minute's total
        would pass OTHER's per-minute share. Raises BudgetExhausted when the
        hourly/daily share is used up or the wait would be too long."""
        deadline = self._clock() + self._other_max_wait
        if request_deadline.get() is not None:
            deadline = min(deadline, request_deadline.get())
        while True:
            now = self._clock()
            with self._lock:
                self._expire(now)
                if now < self._other_refuse_until:
                    raise BudgetExhausted("Open-Meteo hourly/daily limit reached")
                if self._other_hour_cap and self._hour_total + weight > self._other_hour_cap:
                    raise BudgetExhausted("hourly share for non-tonight work used up")
                if self._soft_cap and self._day_total + weight > self._soft_cap:
                    raise BudgetExhausted("daily share for non-tonight work used up")
                wait = self._other_wait_until - now
                if wait <= 0 and self._other_minute_cap and self._minute_total > 0 \
                        and self._minute_total + weight > self._other_minute_cap:
                    # Until enough of the oldest calls leave the minute window.
                    excess = self._minute_total + weight - self._other_minute_cap
                    freed = 0.0
                    for t, w in self._minute:
                        freed += w
                        if freed >= excess:
                            wait = t + _MINUTE - now
                            break
                if wait <= 0:
                    self._charge_locked(now, weight, current_client.get())
                    return
            if now + wait > deadline:
                raise BudgetExhausted("per-minute share for non-tonight work is busy")
            await self._sleep(wait + 0.05)

    def note_rate_limited(self, reason: str) -> None:
        """Open-Meteo answered 429: hold OTHER work back accordingly."""
        now = self._clock()
        with self._lock:
            if reason.startswith("Minutely"):
                self._other_wait_until = max(self._other_wait_until, now + _MINUTE)
            elif reason.startswith("Hourly"):
                self._other_refuse_until = max(self._other_refuse_until, now + _HOUR)
            elif reason.startswith("Daily"):
                midnight = datetime.fromtimestamp(now, tz=timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                self._other_refuse_until = max(self._other_refuse_until, midnight.timestamp())
            else:  # unspecified 429 — treat as the minute window
                self._other_wait_until = max(self._other_wait_until, now + _MINUTE)

    def other_wait_remaining(self) -> float:
        """Seconds OTHER work should still wait after a minutely 429."""
        return max(0.0, self._other_wait_until - self._clock())

    def _record(self, now: float, weight: float) -> None:
        self._day.append((now, weight))
        self._day_total += weight
        self._hour.append((now, weight))
        self._hour_total += weight
        self._minute.append((now, weight))
        self._minute_total += weight
        self._expire(now)

    def _expire(self, now: float) -> None:
        self._expire_day(now)
        while self._hour and self._hour[0][0] <= now - _HOUR:
            self._hour_total -= self._hour.popleft()[1]
        while self._minute and self._minute[0][0] <= now - _MINUTE:
            self._minute_total -= self._minute.popleft()[1]

    def day_total(self) -> float:
        with self._lock:
            self._expire_day(self._clock())
            return self._day_total

    def _expire_day(self, now: float) -> None:
        while self._day and self._day[0][0] <= now - _DAY:
            self._day_total -= self._day.popleft()[1]

    def _expire_client(self, client: str, now: float) -> None:
        q = self._clients.get(client)
        if q is None:
            return
        while q and q[0][0] <= now - _HOUR:
            self._client_totals[client] -= q.popleft()[1]
        if not q:
            del self._clients[client]
            self._client_totals.pop(client, None)


def client_key(headers: dict[str, str], peer: Optional[str]) -> str:
    """The real client IP. Render sits behind Cloudflare, which sets
    CF-Connecting-IP itself (a client can't forge it); X-Forwarded-For's first
    entry is client-supplied and only a fallback."""
    cf = headers.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    xff = headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return peer or "unknown"


class PrioritySlots:
    """A concurrency limit whose free slots go to waiting TONIGHT calls first.

    A cold heatmap queues ~40 requests behind OPEN_METEO_MAX_CONCURRENCY; a
    plain FIFO semaphore made tonight's prediction wait behind all of them.
    """

    def __init__(self, slots: int) -> None:
        self._free = slots
        self._waiting: dict[str, deque[asyncio.Future]] = {TONIGHT: deque(), OTHER: deque()}

    async def acquire(self, level: str) -> None:
        if self._free > 0 and not any(self._waiting.values()):
            self._free -= 1
            return
        fut = asyncio.get_running_loop().create_future()
        queue = self._waiting[TONIGHT if level == TONIGHT else OTHER]
        queue.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()  # granted just as we were cancelled: pass it on
            else:
                try:
                    queue.remove(fut)
                except ValueError:
                    pass
            raise

    def release(self) -> None:
        for level in (TONIGHT, OTHER):
            queue = self._waiting[level]
            while queue:
                fut = queue.popleft()
                if not fut.done():
                    fut.set_result(None)  # the slot moves straight to it
                    return
        self._free += 1
