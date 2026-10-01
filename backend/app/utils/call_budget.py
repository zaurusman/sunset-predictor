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
"""
from __future__ import annotations

import contextvars
import math
import threading
import time
from collections import defaultdict, deque
from datetime import date
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# Who the current request is for (None outside a request: alert runs, scripts).
current_client: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_client", default=None
)

_HOUR = 3600.0
_DAY = 86_400.0


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

    def __init__(self, client_hourly_limit: float, daily_soft_cap: float, clock=time.time) -> None:
        self._client_limit = client_hourly_limit
        self._soft_cap = daily_soft_cap
        self._clock = clock
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
            self._day.append((now, weight))
            self._day_total += weight
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
                    "new heatmaps and climatology builds; predictions continue.", self._day_total,
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

    def optional_work_allowed(self) -> bool:
        """False once the process-wide 24 h total passes the soft cap."""
        if not self._soft_cap:
            return True
        with self._lock:
            self._expire_day(self._clock())
            if self._day_total < self._soft_cap:
                self._soft_cap_logged = False
                return True
            return False

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
