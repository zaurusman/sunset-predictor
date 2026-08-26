"""Schemas for POST /rate — human sunset ratings used as ML training labels.

WHY THIS EXISTS
---------------
The previous ML attempt failed because its labels were Reddit upvotes on
photos, which suffer two fatal problems:

  1. Selection bias — every labelled sky was one somebody chose to photograph
     and post. The dataset contained no bad evenings, so a model could only
     learn "which nice sunset got more upvotes", not "is tonight worth going
     outside for".
  2. Location mismatch — post timestamps were joined to weather at five fixed
     cities regardless of where the photo was actually taken.

A first-party rating fixes both: it is tied to a known lat/lon, and the user
rates *every* evening they check, including the dull ones. That negative signal
is the part that cannot be scraped.

Each rating is stored alongside the RAW weather snapshots that produced it —
not just the score — so that future scoring changes can be re-evaluated offline
against the same labels without refetching history.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

# Same four moments the engine samples (app/services/weather_service.py). A
# rating without one of these is assumed to describe the whole evening, which
# is how every rating before this field existed was recorded.
ObservedMoment = Literal["-15m", "sunset", "+15m", "+30m"]


class RatingRequest(BaseModel):
    """A human rating of one evening's sunset at one location."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "How the sunset actually looked: 1 = nothing, 2 = dull, "
            "3 = pleasant, 4 = very good, 5 = exceptional."
        ),
    )

    target_date: Optional[date] = Field(
        default=None,
        description="Evening being rated. Defaults to the local sunset date.",
    )
    observed_moment: Optional[ObservedMoment] = Field(
        default=None,
        description=(
            "Which sampled moment this rating actually describes, if known — "
            "e.g. a photo timestamped 20 minutes after sunset should be rated "
            "against '+15m' or '+30m', not the whole evening. A sunset-lit-cloud "
            "evening and an afterglow-gradient evening are different physical "
            "events that happen to share a date; without this a rating collapses "
            "them into one label and the engine cannot tell which moment it got "
            "right or wrong. None means the rating describes the evening as a "
            "whole (e.g. the in-app one-tap rating, which isn't tied to a moment)."
        ),
    )
    location_name: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=500)


class RatingResponse(BaseModel):
    """Acknowledgement returned after a rating is stored."""

    success: bool
    message: str
    rated_date: date
    predicted_score: Optional[float] = Field(
        default=None,
        description="What the model scored this evening, for immediate feedback.",
    )
    total_ratings: int = Field(
        description="How many ratings this server has stored, across all locations."
    )


class RatingStats(BaseModel):
    """Aggregate view of collected ratings — progress toward a usable dataset."""

    total_ratings: int
    distinct_locations: int
    distinct_dates: int
    rating_histogram: dict[int, int] = Field(
        description="Count per star value 1–5. A healthy training set has mass at BOTH ends."
    )
    mean_rating: Optional[float] = None
    spearman_vs_model: Optional[float] = Field(
        default=None,
        description=(
            "Rank correlation between stored ratings and the score the model gave "
            "at capture time. This is the single number that says whether the "
            "engine is working. None until there are enough ratings to be meaningful."
        ),
    )
    spearman_sample_size: int = 0
    note: str = ""
