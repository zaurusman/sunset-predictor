"""Decides who gets an alert tonight, and sends it.

Called by an external scheduler (see .github/workflows/sunset-notifications.yml)
rather than an in-process timer, because Render's free tier sleeps an idle
service — an internal cron would simply not fire. An external ping both wakes
the service and triggers the run.

The rule for each subscriber, evaluated once per evening:

    notify_at = sunset - lead_minutes
    if now is between notify_at and sunset, and we have not checked today:
        score tonight; alert if it reaches their threshold

Scoring happens at most ONCE per subscriber per day — `last_checked_date` is
stamped whether or not an alert goes out. Without that, a cron every 15 minutes
would re-score every subscriber every run for the whole lead window and burn
through Open-Meteo's rate limit for nothing. The cost is that the alert
reflects the forecast at the moment of the first check inside the window, not
a continuously updated one.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from app.core.logging import get_logger
from app.schemas.notification import DispatchResult
from app.schemas.prediction import PredictRequest
from app.services.astronomy_service import AstronomyService
from app.services.prediction_service import PredictionService
from app.services.push_service import PushGoneError, PushService
from app.services.subscription_store import SubscriptionStore
from app.utils.time_utils import get_timezone_for_coordinates, utcnow

logger = get_logger(__name__)

# Consecutive send failures tolerated before a subscription is dropped. A push
# service can be briefly unreachable; it should not cost someone their alerts.
MAX_FAILURES = 5


def _headline(score: float, category: str) -> str:
    """Notification title. Loud only when the sky has earned it."""
    if score >= 80:
        return "Tonight could be extraordinary"
    if score >= 70:
        return "Worth heading out tonight"
    return f"A {category.lower()} sunset tonight"


class NotificationDispatcher:
    def __init__(
        self,
        *,
        store: SubscriptionStore,
        push_service: PushService,
        prediction_service: PredictionService,
        astro_service: AstronomyService,
    ) -> None:
        self._store = store
        self._push = push_service
        self._predictions = prediction_service
        self._astro = astro_service

    async def run(self, now: datetime | None = None) -> DispatchResult:
        """Evaluate every subscription once and send whatever is due.

        `now` is injectable so tests can place the clock inside or outside a
        subscriber's window without waiting for a real sunset.
        """
        now = now or utcnow()
        result = DispatchResult(
            checked=0, in_window=0, scored=0, sent=0,
            below_threshold=0, pruned=0, failed=0,
        )

        for record in self._store.all():
            result.checked += 1
            try:
                await self._process(record, now, result)
            except Exception as exc:
                # One bad subscription must never abort the whole run — the
                # people after it in the list are waiting on the same cron tick.
                logger.error(
                    "Dispatch failed for subscription %s: %s",
                    record.get("endpoint", "")[:60],
                    exc,
                    exc_info=True,
                )
                result.failed += 1

        logger.info(
            "Dispatch run: checked=%d in_window=%d scored=%d sent=%d "
            "below_threshold=%d pruned=%d failed=%d",
            result.checked, result.in_window, result.scored, result.sent,
            result.below_threshold, result.pruned, result.failed,
        )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _process(
        self, record: dict[str, Any], now: Any, result: DispatchResult
    ) -> None:
        lat = float(record["latitude"])
        lon = float(record["longitude"])

        # "Today" is the subscriber's local today, not the server's — a UTC
        # server would otherwise roll the date mid-evening for anyone east of it.
        tz = get_timezone_for_coordinates(lat, lon)
        local_today: date = now.astimezone(tz).date()

        if record.get("last_checked_date") == local_today.isoformat():
            return

        sunset = self._astro.get_sunset_time(lat, lon, local_today)
        notify_at = sunset - timedelta(minutes=int(record.get("lead_minutes", 120)))

        # Too early: a later cron tick will catch it.
        if now < notify_at:
            return
        # Too late: sunset has passed. Stamp the day so we do not keep
        # re-checking a subscriber the cron missed (a sleeping service, say)
        # and then wake them after dark.
        if now > sunset:
            await self._store.update_fields(
                record["endpoint"], last_checked_date=local_today.isoformat()
            )
            return

        result.in_window += 1

        prediction = await self._predictions.predict(
            PredictRequest(latitude=lat, longitude=lon, target_date=local_today)
        )
        result.scored += 1

        # Stamp before sending: a send that fails should not re-score tomorrow's
        # quota-worth of predictions on the next tick 15 minutes from now.
        await self._store.update_fields(
            record["endpoint"], last_checked_date=local_today.isoformat()
        )

        threshold = float(record.get("threshold", 70.0))
        score = prediction.beauty_score_0_100
        if score < threshold:
            result.below_threshold += 1
            return

        await self._send(record, prediction, result)

    async def _send(
        self, record: dict[str, Any], prediction: Any, result: DispatchResult
    ) -> None:
        score = prediction.beauty_score_0_100
        place = record.get("location_name") or "your spot"
        local_sunset = prediction.sunset_time.astimezone(
            get_timezone_for_coordinates(record["latitude"], record["longitude"])
        )

        payload = {
            "title": _headline(score, prediction.category),
            "body": (
                f"{round(score)}/100 in {place} — sunset at "
                f"{local_sunset.strftime('%H:%M')}."
            ),
            "score": round(score),
            "category": prediction.category,
            "url": self._deep_link(record),
        }

        subscription = {
            "endpoint": record["endpoint"],
            "keys": record["keys"],
        }

        try:
            await self._push.send(subscription=subscription, payload=payload)
        except PushGoneError:
            await self._store.delete(record["endpoint"])
            result.pruned += 1
            logger.info("Pruned dead subscription %s", record["endpoint"][:60])
            return
        except Exception as exc:
            failures = int(record.get("failure_count", 0)) + 1
            if failures >= MAX_FAILURES:
                await self._store.delete(record["endpoint"])
                result.pruned += 1
                logger.warning(
                    "Dropped subscription after %d consecutive failures: %s",
                    failures, exc,
                )
            else:
                await self._store.update_fields(
                    record["endpoint"], failure_count=failures
                )
                logger.warning("Push send failed (attempt %d): %s", failures, exc)
            result.failed += 1
            return

        await self._store.update_fields(record["endpoint"], failure_count=0)
        result.sent += 1

    def _deep_link(self, record: dict[str, Any]) -> str:
        """Where tapping the notification lands — that place, not the default."""
        from urllib.parse import urlencode

        params = urlencode(
            {
                "lat": record["latitude"],
                "lon": record["longitude"],
                "name": record.get("location_name", ""),
            }
        )
        return f"/?{params}"
