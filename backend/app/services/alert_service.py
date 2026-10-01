"""Hourly Epic-sunset alert run.

COST MODEL
----------
Belled places are grouped into the same 0.1° cells as the weather cache. Each
cell is predicted at most once per local day, when its sunset is ~4 h away —
so Open-Meteo usage scales with distinct places, not with subscribers, and a
cell a user already looked at today is served from cache.

PACING
------
One run checks at most ``max_cells`` cells, the closest to sunset first, and
reports how many are still due. The cron calls again a minute later until none
remain, so a busy hour costs ~1 forecast refresh per cell spread over several
minutes instead of one burst against Open-Meteo's per-minute limit. The lead
window reaches below 4 h so cells carried over still get checked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Awaitable, Callable, Optional, Protocol
from urllib.parse import quote
from zoneinfo import ZoneInfo

from app.core.logging import get_logger
from app.schemas.prediction import PredictRequest, PredictResponse
from app.utils.call_budget import TONIGHT, priority
from app.schemas.push import AlertRunSummary
from app.services.subscription_store import StoredSubscription, SubscriptionStore, cell_key
from app.services.weather_service import WeatherUnavailableError
from app.utils.time_utils import utcnow

logger = get_logger(__name__)

Predictor = Callable[[float, float, date], Awaitable[PredictResponse]]

_WINDOW_OFFSETS_MIN = {"-15m": -15, "sunset": 0, "+15m": 15, "+30m": 30}


class Sender(Protocol):
    async def send(self, sub: StoredSubscription, payload: dict) -> str: ...


def prediction_predictor(prediction_service) -> Predictor:
    async def predict(lat: float, lon: float, day: date) -> PredictResponse:
        # Alerts are about tonight's sunset: tonight's priority.
        with priority(TONIGHT):
            return await prediction_service.predict(
                PredictRequest(latitude=lat, longitude=lon, target_date=day),
                warm_climatology=False,
            )
    return predict


def build_payload(place: dict, prediction, tz: str, cell: str, day: date) -> dict:
    try:
        zone = ZoneInfo(tz)
    except Exception:
        zone = ZoneInfo("UTC")
    offset = _WINDOW_OFFSETS_MIN.get(prediction.best_window_point, 0)
    best = (prediction.sunset_time + timedelta(minutes=offset)).astimezone(zone)
    score = int(math.floor(prediction.beauty_score_0_100 + 0.5))  # matches the UI's Math.round
    name = place["name"]
    return {
        "title": "🔥 Epic sunset tonight",
        "body": f"{name} — {score}/100. Best around {best:%H:%M}.",
        "url": f"/?lat={place['latitude']}&lon={place['longitude']}&name={quote(name)}",
        "tag": f"epic-{cell}-{day.isoformat()}",
    }


@dataclass
class _Cell:
    lat: float
    lon: float
    members: list[tuple[StoredSubscription, dict]] = field(default_factory=list)


class AlertService:
    def __init__(
        self,
        store: SubscriptionStore,
        predictor: Predictor,
        sunset_for: Callable[[float, float, date], datetime],
        local_date_for: Callable[[float, float], date],
        sender: Sender,
        lead_min_hours: float = 2.5,
        lead_max_hours: float = 4.5,
        decimals: int = 1,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._store = store
        self._predict = predictor
        self._sunset_for = sunset_for
        self._local_date_for = local_date_for
        self._sender = sender
        self._lead_min = lead_min_hours
        self._lead_max = lead_max_hours
        self._decimals = decimals
        self._clock = clock

    def _group(self, subs: list[StoredSubscription]) -> dict[str, _Cell]:
        cells: dict[str, _Cell] = {}
        for sub in subs:
            seen: set[str] = set()
            for place in sub.places:
                key = cell_key(place["latitude"], place["longitude"], self._decimals)
                if key in seen:
                    continue  # one push per subscriber per cell
                seen.add(key)
                cell = cells.setdefault(key, _Cell(place["latitude"], place["longitude"]))
                cell.members.append((sub, place))
        return cells

    async def run(self, force: bool = False, max_cells: Optional[int] = None) -> AlertRunSummary:
        summary = AlertRunSummary()
        cells = self._group(await self._store.all())
        summary.cells = len(cells)
        now = self._clock()
        gone: set[str] = set()

        due: list[tuple[float, str, _Cell, date]] = []
        for key, cell in cells.items():
            day = self._local_date_for(cell.lat, cell.lon)
            lead = (self._sunset_for(cell.lat, cell.lon, day) - now).total_seconds() / 3600
            if not force:
                if not (self._lead_min <= lead < self._lead_max):
                    continue
                if await self._store.cell_checked(key, day):
                    continue
            due.append((lead, key, cell, day))
        due.sort(key=lambda d: d[0])   # closest to sunset first
        if max_cells is not None and not force:
            summary.remaining = max(0, len(due) - max_cells)
            due = due[:max_cells]

        for _, key, cell, day in due:
            summary.cells_due += 1

            try:
                prediction = await self._predict(cell.lat, cell.lon, day)
            except WeatherUnavailableError as exc:
                logger.warning("Alert check for cell %s deferred — weather unavailable: %s", key, exc)
                continue
            except Exception:
                logger.exception("Alert check for cell %s failed", key)
                continue
            summary.cells_checked += 1

            if not force:
                await self._store.record_cell_check(key, day, prediction.beauty_score_0_100)
                if prediction.category != "Epic":
                    continue

            for sub, place in cell.members:
                if sub.endpoint in gone:
                    continue
                if not force and sub.last_notified.get(key) == day.isoformat():
                    continue
                result = await self._sender.send(sub, build_payload(place, prediction, sub.tz, key, day))
                if result == "ok":
                    summary.notifications_sent += 1
                    await self._store.mark_notified(sub.endpoint, key, day)
                elif result == "gone":
                    gone.add(sub.endpoint)
                    await self._store.delete(sub.endpoint)
                    summary.pruned += 1

        logger.info("Alert run: %s", summary.model_dump())
        return summary
