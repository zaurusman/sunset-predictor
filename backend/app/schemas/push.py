"""Request / response schemas for Web Push subscription and alert runs."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PushKeys(BaseModel):
    p256dh: str = Field(..., min_length=1, max_length=256)
    auth: str = Field(..., min_length=1, max_length=128)


class PushSubscriptionIn(BaseModel):
    endpoint: str = Field(..., pattern=r"^https://", max_length=2048)
    keys: PushKeys


class AlertPlace(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    name: str = Field(..., min_length=1, max_length=120)


class SubscribeRequest(BaseModel):
    subscription: PushSubscriptionIn
    places: list[AlertPlace] = Field(..., max_length=5)
    tz: str = Field("UTC", max_length=64, description="IANA time zone of the device")


class UnsubscribeRequest(BaseModel):
    endpoint: str = Field(..., max_length=2048)


class VapidKeyResponse(BaseModel):
    public_key: str


class AlertRunSummary(BaseModel):
    cells: int = 0
    cells_due: int = 0
    cells_checked: int = 0
    notifications_sent: int = 0
    pruned: int = 0
