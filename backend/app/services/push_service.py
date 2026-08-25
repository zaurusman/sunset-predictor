"""Web Push delivery via VAPID.

Sends an encrypted payload to a browser's push service (FCM, Mozilla, WNS —
whichever the browser chose). The push service holds the message and wakes the
service worker, so this works with the tab closed and the phone locked.

The service is disabled (raises PushNotConfiguredError) when VAPID keys are
missing, so the endpoints return a clear 503 rather than failing per-send.
"""
from __future__ import annotations

import asyncio
import json
from functools import partial
from typing import Any

from pywebpush import WebPushException, webpush

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Ask the push service to hold an undelivered alert only until sunset is over.
# A "tonight looks Epic" push arriving tomorrow morning is worse than no push.
DEFAULT_TTL_SECONDS = 3 * 60 * 60


class PushNotConfiguredError(RuntimeError):
    """Raised when VAPID keys are not configured."""


class PushGoneError(RuntimeError):
    """The endpoint is permanently dead (404/410) — drop the subscription."""


class PushService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def is_configured(self) -> bool:
        s = self._settings
        return bool(s.VAPID_PUBLIC_KEY and s.VAPID_PRIVATE_KEY and s.VAPID_SUBJECT)

    def _require_configured(self) -> None:
        if not self.is_configured:
            raise PushNotConfiguredError(
                "Push notifications are not configured on this server. "
                "Set VAPID_PUBLIC_KEY, VAPID_PRIVATE_KEY and VAPID_SUBJECT."
            )

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def send(
        self,
        *,
        subscription: dict[str, Any],
        payload: dict[str, Any],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        """Deliver one notification.

        Raises PushGoneError when the subscription is dead and should be
        pruned, and WebPushException for anything else worth retrying later.
        Runs the blocking SDK call in a thread so the event loop is not stalled.
        """
        self._require_configured()

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, partial(self._send, subscription, payload, ttl_seconds)
            )
        except WebPushException as exc:
            status = getattr(exc.response, "status_code", None)
            # 404: endpoint never existed. 410 Gone: the browser unsubscribed,
            # was uninstalled, or cleared its data. Neither will ever succeed.
            if status in (404, 410):
                raise PushGoneError(f"Subscription is gone (HTTP {status})") from exc
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send(
        self,
        subscription: dict[str, Any],
        payload: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        """Blocking pywebpush call — invoke via run_in_executor."""
        s = self._settings
        webpush(
            subscription_info=subscription,
            data=json.dumps(payload),
            ttl=ttl_seconds,
            vapid_private_key=s.VAPID_PRIVATE_KEY,
            vapid_claims={"sub": s.VAPID_SUBJECT},
        )
