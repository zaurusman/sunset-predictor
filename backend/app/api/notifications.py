"""Web Push subscription management and the cron-triggered dispatch run."""
from __future__ import annotations

import secrets

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi import status as http_status

from app.core.logging import get_logger
from app.schemas.notification import (
    DispatchResult,
    NotificationConfig,
    SubscribeRequest,
    SubscriptionResponse,
    UnsubscribeRequest,
)
from app.utils.time_utils import utcnow

logger = get_logger(__name__)
router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get(
    "/config",
    response_model=NotificationConfig,
    summary="Push configuration for this server",
)
async def get_config(request: Request) -> NotificationConfig:
    """Tells the frontend whether to offer alerts, and with which VAPID key.

    Serving the key from here rather than baking it into the bundle means the
    frontend does not need a rebuild when keys are rotated, and a server
    without keys configured simply hides the feature.
    """
    settings = request.app.state.settings
    push = request.app.state.push_service
    return NotificationConfig(
        enabled=push.is_configured,
        vapid_public_key=settings.VAPID_PUBLIC_KEY if push.is_configured else "",
        default_threshold=settings.NOTIFY_DEFAULT_THRESHOLD,
        default_lead_minutes=settings.NOTIFY_DEFAULT_LEAD_MINUTES,
    )


@router.post(
    "/subscribe",
    response_model=SubscriptionResponse,
    summary="Register this browser for sunset alerts",
)
async def subscribe(body: SubscribeRequest, request: Request) -> SubscriptionResponse:
    push = request.app.state.push_service
    if not push.is_configured:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Push notifications are not configured on this server.",
        )

    store = request.app.state.subscription_store
    record = await store.upsert(
        {
            "endpoint": body.endpoint,
            "keys": body.keys.model_dump(),
            "latitude": body.latitude,
            "longitude": body.longitude,
            "location_name": body.location_name,
            "threshold": body.threshold,
            "lead_minutes": body.lead_minutes,
            "created_at": utcnow().isoformat(),
            "last_checked_date": None,
            "failure_count": 0,
        }
    )
    logger.info(
        "Subscribed %s for %s (threshold=%.0f, lead=%dm). Total subscribers: %d",
        body.endpoint[:60],
        body.location_name or f"{body.latitude:.3f},{body.longitude:.3f}",
        body.threshold,
        body.lead_minutes,
        store.count(),
    )
    return SubscriptionResponse(**{
        k: v for k, v in record.items() if k in SubscriptionResponse.model_fields
    })


@router.post(
    "/unsubscribe",
    status_code=http_status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Stop sending alerts to this browser",
)
async def unsubscribe(body: UnsubscribeRequest, request: Request) -> Response:
    store = request.app.state.subscription_store
    removed = await store.delete(body.endpoint)
    # Deliberately 204 either way: unsubscribing something already gone is the
    # desired end state, and a 404 would only make the client handle a
    # non-problem.
    logger.info(
        "Unsubscribe %s (existed=%s). Total subscribers: %d",
        body.endpoint[:60], removed, store.count(),
    )
    return Response(status_code=http_status.HTTP_204_NO_CONTENT)


@router.get(
    "/status",
    response_model=SubscriptionResponse,
    summary="Read back this browser's alert settings",
)
async def get_status(endpoint: str, request: Request) -> SubscriptionResponse:
    """Lets a returning device confirm the server still knows about it.

    A browser can hold a live PushSubscription that the server has forgotten —
    exactly what happens when Render's ephemeral disk is wiped — and this is
    how the UI notices instead of promising alerts that will never arrive.
    """
    store = request.app.state.subscription_store
    record = store.get(endpoint)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="No subscription registered for that endpoint.",
        )
    return SubscriptionResponse(**{
        k: v for k, v in record.items() if k in SubscriptionResponse.model_fields
    })


@router.post(
    "/dispatch",
    response_model=DispatchResult,
    summary="Run the evening dispatch (called by an external scheduler)",
)
async def dispatch(
    request: Request,
    x_dispatch_secret: str = Header(default=""),
) -> DispatchResult:
    """Evaluate every subscription and send whatever is due.

    Safe to call as often as the scheduler likes: each subscriber is scored at
    most once per local day, so extra calls cost a store scan and nothing else.
    """
    settings = request.app.state.settings

    if not settings.NOTIFY_DISPATCH_SECRET:
        # Without a secret this endpoint would let anyone drain the Open-Meteo
        # quota and spam every subscriber, so it stays shut rather than open.
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dispatch is not configured. Set NOTIFY_DISPATCH_SECRET.",
        )

    # compare_digest, not ==, so a wrong guess takes the same time as a right
    # one and cannot be narrowed down character by character.
    if not secrets.compare_digest(x_dispatch_secret, settings.NOTIFY_DISPATCH_SECRET):
        logger.warning("Rejected dispatch call with a bad secret")
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Invalid dispatch secret.",
        )

    dispatcher = request.app.state.notification_dispatcher
    return await dispatcher.run()
