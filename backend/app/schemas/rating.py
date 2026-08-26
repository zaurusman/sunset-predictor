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

from pydantic import BaseModel, Field, model_validator

# Same four moments the engine samples (app/services/weather_service.py). A
# rating without one of these is assumed to describe the whole evening, which
# is how every rating before this field existed was recorded.
ObservedMoment = Literal["-15m", "sunset", "+15m", "+30m"]


class RatingRequest(BaseModel):
    """A human rating of one evening's sunset at one location.

    Supply *rating_0_100* whenever the rating is a considered judgement (rating
    a photo, scoring an evening from memory). Use *rating* only for the
    one-tap UI, which genuinely cannot express more than five levels.
    """

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

    # THE CANONICAL SCALE. Same 0-100 the app displays, so a label and a
    # prediction are directly comparable without a conversion in between.
    #
    # This exists because the 1-5 field below was destroying most of the
    # signal in considered ratings. Fifteen photo labels scored on 0-100 held
    # ELEVEN distinct values; squashed through round(score/25 + 1) they held
    # THREE, with eleven of the fifteen collapsed onto a single value. Spearman
    # is a rank correlation, so a tie carries no information — most of the
    # dataset was rating-vs-rating ties and the correlation was measuring
    # almost nothing.
    rating_0_100: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description=(
            "How good the sky actually was, 1-100, on the same scale the app "
            "displays. Preferred over `rating` — use this for any considered "
            "judgement."
        ),
    )

    # The one-tap scale. Kept because five labelled buttons is the right
    # instrument for a phone tap on the way past a window, and because every
    # rating collected before rating_0_100 existed is on this scale.
    rating: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description=(
            "Coarse fallback scale for the one-tap UI: 1 = nothing, 2 = dull, "
            "3 = pleasant, 4 = very good, 5 = exceptional. Prefer rating_0_100."
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

    @model_validator(mode="after")
    def _require_a_rating(self) -> "RatingRequest":
        if self.rating_0_100 is None and self.rating is None:
            raise ValueError("Supply rating_0_100 (preferred) or rating.")
        return self

    def score_0_100(self) -> float:
        """The rating on the canonical 0-100 scale.

        Falls back to the band CENTRE of the coarse scale, which is the honest
        reading of a tap: "very good" means somewhere in the fourth fifth, not
        exactly 75. Callers that care about precision should check whether
        rating_0_100 was supplied rather than trusting this to be exact.
        """
        if self.rating_0_100 is not None:
            return float(self.rating_0_100)
        return COARSE_TO_0_100[self.rating]


# Band centres, not edges: a tap on "very good" says the evening fell in the
# fourth fifth of the scale, so its best single estimate is the middle of that
# fifth (70-90 -> 80). The old conversion, round(score/25 + 1) inverted, put it
# at the bottom edge instead and made every tap read 5 points pessimistic.
COARSE_TO_0_100: dict[int, float] = {
    1: 10.0,   # nothing
    2: 30.0,   # dull
    3: 50.0,   # pleasant
    4: 70.0,   # very good
    5: 90.0,   # exceptional
}


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
