"""When did each weather model last publish a run?

Forecast data only changes when a model publishes a new run. ICON-EU (the
regional part of icon_seamless over Israel) publishes every 3 h, the ensembles
every 6-12 h, and CAMS air quality every 12-24 h. Fixed cache TTLs (2 h for the
forecast, 1 h for the ensemble) re-downloaded unchanged data on every expiry,
while sometimes serving a run that had already been superseded.

Open-Meteo publishes per-model metadata at /data/<model>/static/meta.json,
including ``last_run_availability_time`` and the update interval. A cached
forecast is current as long as no model it came from has published since the
cache entry was stored (WeatherService._since).

Polling is lazy and cheap. A model's metadata is re-read only when it is
needed, at most once an hour, plus every few minutes once its next run is
due. A model whose area doesn't cover the location is skipped. When metadata
can't be read, callers fall back to the old fixed TTLs, so behaviour is never
worse than before.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

# Model families behind the requests WeatherService makes, as Open-Meteo
# metadata names, with the API host serving each (settings attribute).
FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    # icon_seamless = ICON-D2 (central Europe) + ICON-EU + ICON global
    "forecast": ("OPEN_METEO_BASE_URL", ("dwd_icon_d2", "dwd_icon_eu", "dwd_icon")),
    # icon_seamless ensemble
    "ensemble": ("OPEN_METEO_ENSEMBLE_URL", ("dwd_icon_d2_eps", "dwd_icon_eu_eps", "dwd_icon_eps")),
    # air-quality `auto` domain: CAMS Europe where covered, else CAMS global
    "aq": ("OPEN_METEO_AIR_QUALITY_URL", ("cams_europe", "cams_global")),
}

_IDLE_POLL = 3600.0     # re-check at least hourly in case a schedule shifts
_DUE_POLL = 180.0       # once a run is due, check every 3 min until it lands
_DUE_EARLY = 600.0      # ...starting 10 min before it is expected
_FAIL_RETRY = 180.0     # after a failed read, try again in 3 min
_TRUST_FOR = 2 * 3600.0  # stop trusting metadata not refreshed for this long

_BBOX = re.compile(r"BBOX\[([^\]]+)\]")


@dataclass
class _ModelState:
    available: float            # epoch seconds the latest run became available
    interval: float             # seconds between runs
    bbox: Optional[tuple[float, float, float, float]]  # lat_min, lon_min, lat_max, lon_max
    read_at: float              # last successful read

    def covers(self, lat: float, lon: float) -> bool:
        if self.bbox is None:
            return True
        lat_min, lon_min, lat_max, lon_max = self.bbox
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _parse(meta: dict, now: float) -> _ModelState:
    bbox = None
    m = _BBOX.search(meta.get("crs_wkt", ""))
    if m:
        a, b, c, d = (float(x) for x in m.group(1).split(","))
        # cams_europe lists latitudes north-first; normalise.
        bbox = (min(a, c), min(b, d), max(a, c), max(b, d))
    return _ModelState(
        available=float(meta["last_run_availability_time"]),
        interval=float(meta.get("update_interval_seconds") or 3600),
        bbox=bbox,
        read_at=now,
    )


class ModelRunClock:
    def __init__(self, http_client: httpx.AsyncClient, settings, clock=time.time) -> None:
        self._http = http_client
        self._settings = settings
        self._clock = clock
        self._states: dict[str, _ModelState] = {}
        self._last_attempt: dict[str, float] = {}
        self._inflight: dict[str, asyncio.Future] = {}

    async def latest_run(self, family: str, lat: float, lon: float) -> Optional[float]:
        """When the newest run of any model in *family* covering (lat, lon)
        became available, or None if that isn't reliably known."""
        host_attr, models = FAMILIES[family]
        relevant = []
        for model in models:
            state = self._states.get(model)
            if state is not None and not state.covers(lat, lon):
                continue  # known not to cover this location: no need to poll it
            relevant.append(model)
        states = await asyncio.gather(*(self._current(host_attr, m) for m in relevant))
        now = self._clock()
        latest: Optional[float] = None
        for state in states:
            if state is None or now - state.read_at > _TRUST_FOR:
                return None
            if state.covers(lat, lon):
                latest = state.available if latest is None else max(latest, state.available)
        return latest

    async def _current(self, host_attr: str, model: str) -> Optional[_ModelState]:
        if self._due(model):
            task = self._inflight.get(model)
            if task is None:
                task = asyncio.ensure_future(self._read(host_attr, model))
                self._inflight[model] = task
                task.add_done_callback(lambda _: self._inflight.pop(model, None))
            await asyncio.shield(task)
        return self._states.get(model)

    def _due(self, model: str) -> bool:
        now = self._clock()
        last = self._last_attempt.get(model)
        if last is None:
            return True
        state = self._states.get(model)
        if state is None:
            return now - last >= _FAIL_RETRY
        if now >= state.available + state.interval - _DUE_EARLY:
            return now - last >= _DUE_POLL
        return now - last >= _IDLE_POLL

    async def _read(self, host_attr: str, model: str) -> None:
        now = self._clock()
        self._last_attempt[model] = now
        base = getattr(self._settings, host_attr).rsplit("/v1", 1)[0]
        url = f"{base}/data/{model}/static/meta.json"
        try:
            resp = await self._http.get(url, timeout=5.0)
            resp.raise_for_status()
            state = _parse(resp.json(), now)
        except Exception as exc:
            logger.warning("Model run metadata unavailable for %s: %s", model, exc)
            return
        previous = self._states.get(model)
        if previous is None or state.available > previous.available:
            logger.info("Model run: %s available since %s", model,
                        time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(state.available)))
        self._states[model] = state
