"""Thin async wrapper around pywebpush."""
from __future__ import annotations

import asyncio
import json
from typing import Literal

from app.core.logging import get_logger
from app.services.subscription_store import StoredSubscription

logger = get_logger(__name__)

SendResult = Literal["ok", "gone", "error"]

# A sunset alert is worthless after sunset; don't let push services hold it longer.
_TTL_SECONDS = 4 * 3600


class WebPushSender:
    def __init__(self, private_key: str, subject: str) -> None:
        self._private_key = private_key
        self._subject = subject

    async def send(self, sub: StoredSubscription, payload: dict) -> SendResult:
        from pywebpush import WebPushException, webpush

        try:
            await asyncio.to_thread(
                webpush,
                subscription_info={"endpoint": sub.endpoint, "keys": {"p256dh": sub.p256dh, "auth": sub.auth}},
                data=json.dumps(payload),
                vapid_private_key=self._private_key,
                # Fresh dict per call: pywebpush mutates the claims (adds aud/exp).
                vapid_claims={"sub": self._subject},
                ttl=_TTL_SECONDS,
            )
            return "ok"
        except WebPushException as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (404, 410):
                return "gone"
            logger.warning("Web push failed (status=%s) for %s…: %s", status, sub.endpoint[:60], exc)
            return "error"
        except Exception as exc:
            logger.warning("Web push error for %s…: %s", sub.endpoint[:60], exc)
            return "error"
