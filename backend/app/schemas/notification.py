"""Schemas for Web Push subscriptions and the dispatch run."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class PushKeys(BaseModel):
    """The two keys a browser hands back from PushManager.subscribe()."""

    p256dh: str = Field(..., min_length=1, description="Client public key (base64url)")
    auth: str = Field(..., min_length=1, description="Client auth secret (base64url)")


class SubscribeRequest(BaseModel):
    """Register (or update) one browser for sunset alerts at one place.

    A browser has exactly one push endpoint, so re-subscribing with a different
    location or threshold updates the existing record rather than adding a
    second one — a phone gets one alert per evening, for wherever it last
    asked about.
    """

    endpoint: str = Field(..., min_length=1, description="Push service endpoint URL")
    keys: PushKeys

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    location_name: str = Field(default="", max_length=200)

    threshold: float = Field(
        default=70.0,
        ge=0,
        le=100,
        description=(
            "Only notify when the predicted score reaches this. Defaults to the "
            "same bar the UI uses for 'worth heading out'."
        ),
    )
    lead_minutes: int = Field(
        default=120,
        # The floor is what makes an hourly scheduler safe: a window narrower
        # than 60 minutes could open and close between two checks and the
        # alert would silently never fire. See dispatch_schedule.py.
        ge=60,
        le=480,
        description="How long before sunset to send the alert.",
    )


class UnsubscribeRequest(BaseModel):
    endpoint: str = Field(..., min_length=1)


class SubscriptionResponse(BaseModel):
    """What the client gets back after subscribing — its own settings, echoed."""

    latitude: float
    longitude: float
    location_name: str
    threshold: float
    lead_minutes: int
    created_at: datetime
    last_checked_date: Optional[date] = None


class NotificationConfig(BaseModel):
    """Told to the frontend at load so it knows whether to offer the toggle."""

    enabled: bool = Field(description="True when VAPID keys are configured server-side")
    vapid_public_key: str = Field(
        default="",
        description="Application server key for PushManager.subscribe(); empty when disabled",
    )
    default_threshold: float
    default_lead_minutes: int


class DispatchSchedule(BaseModel):
    """When dispatch actually needs to run, so a scheduler can skip the rest.

    Consumed by the planner workflow, which commits it to the repo; the
    dispatch workflow then reads it locally and only calls the backend during
    an hour that appears here.
    """

    subscriber_count: int
    cron_hours: list[int] = Field(
        description="UTC hours (0–23) in which at least one alert window is open"
    )
    next_window_opens: Optional[datetime] = Field(
        default=None, description="Next moment any subscriber's window opens"
    )
    computed_at: datetime
    cron_expression: Optional[str] = Field(
        default=None, description="The hours as a crontab line; null when nobody is subscribed"
    )


class DispatchResult(BaseModel):
    """Per-run accounting, returned to whatever cron called the endpoint."""

    checked: int = Field(description="Subscriptions inspected")
    in_window: int = Field(description="Inside their pre-sunset send window")
    scored: int = Field(description="Predictions actually run")
    sent: int = Field(description="Alerts delivered to a push service")
    below_threshold: int = Field(description="Scored, but not worth an alert")
    pruned: int = Field(description="Dropped because the push service said they are gone")
    failed: int = Field(description="Send attempts that errored")
