"""Web Push subscription endpoints and the secret-protected hourly alert run."""
from __future__ import annotations

import hmac
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request, Response

from app.schemas.push import (
    AlertRunSummary,
    SubscribeRequest,
    UnsubscribeRequest,
    VapidKeyResponse,
)

router = APIRouter(tags=["push"])

_DISABLED = "Sunset alerts are not configured on this server."


def _store(request: Request):
    store = getattr(request.app.state, "subscription_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail=_DISABLED)
    return store


@router.get("/push/vapid-key", response_model=VapidKeyResponse)
async def vapid_key(request: Request) -> VapidKeyResponse:
    _store(request)
    key = request.app.state.settings.VAPID_PUBLIC_KEY
    if not key:
        raise HTTPException(status_code=503, detail=_DISABLED)
    return VapidKeyResponse(public_key=key)


@router.post("/push/subscribe", status_code=204)
async def subscribe(body: SubscribeRequest, request: Request) -> Response:
    await _store(request).upsert(
        body.subscription.endpoint,
        body.subscription.keys.p256dh,
        body.subscription.keys.auth,
        [p.model_dump() for p in body.places],
        body.tz,
    )
    return Response(status_code=204)


@router.delete("/push/subscribe", status_code=204)
async def unsubscribe(body: UnsubscribeRequest, request: Request) -> Response:
    await _store(request).delete(body.endpoint)
    return Response(status_code=204)


@router.post("/internal/alerts/run", response_model=AlertRunSummary, include_in_schema=False)
async def run_alerts(
    request: Request,
    force: bool = False,
    x_alerts_secret: Optional[str] = Header(default=None),
) -> AlertRunSummary:
    secret = request.app.state.settings.ALERTS_SECRET
    if not secret or not hmac.compare_digest(x_alerts_secret or "", secret):
        raise HTTPException(status_code=401, detail="Unauthorized")
    service = getattr(request.app.state, "alert_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail=_DISABLED)
    per_call = request.app.state.settings.ALERT_CELLS_PER_CALL
    return await service.run(force=force, max_cells=per_call or None)
