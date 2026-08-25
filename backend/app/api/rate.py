"""POST /rate and GET /ratings/stats — collect human sunset ratings.

These ratings are the training labels the previous ML attempt never had. See
app/schemas/rating.py for why the Reddit-derived labels could not work, and
docs/scoring-v2-plan.md for how these feed back into scoring.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi import status as http_status

from app.core.logging import get_logger
from app.schemas.rating import RatingRequest, RatingResponse, RatingStats
from app.services.weather_service import WeatherUnavailableError
from app.utils.math_utils import spearman
from app.utils.time_utils import utcnow

logger = get_logger(__name__)
router = APIRouter(tags=["rating"])

# Below this many paired ratings a rank correlation is noise, so we withhold it
# rather than show a number that will swing wildly with the next few rows.
_MIN_PAIRS_FOR_SPEARMAN = 15


@router.post(
    "/rate",
    response_model=RatingResponse,
    summary="Rate tonight's sunset",
    description=(
        "Record how the sunset actually looked, 1–5. The server stores the rating "
        "together with the raw weather snapshots and the score the model gave, so "
        "the engine can be measured and tuned against real outcomes."
    ),
)
async def rate_sunset(request: Request, body: RatingRequest) -> RatingResponse:
    svc = request.app.state.prediction_service
    store = request.app.state.rating_store
    settings = request.app.state.settings

    target_date: date = body.target_date or svc.local_sunset_date_for(
        body.latitude, body.longitude
    )

    # Refuse future dates: you cannot have seen a sunset that hasn't happened.
    # Without this the store fills with aspirational ratings that poison training.
    today = svc.local_sunset_date_for(body.latitude, body.longitude)
    if target_date > today:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot rate {target_date} — that sunset hasn't happened yet.",
        )

    # ------------------------------------------------------------------
    # Capture the model's view of this evening alongside the human label.
    # A rating without its inputs is useless for training, so a weather
    # failure here is fatal to the request rather than silently degrading.
    # ------------------------------------------------------------------
    try:
        prediction, snapshots = await svc.capture_rating_context(
            lat=body.latitude,
            lon=body.longitude,
            target_date=target_date,
            horizon_deg=settings.DEFAULT_HORIZON_OBSTRUCTION_DEG,
        )
    except WeatherUnavailableError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weather data is unavailable right now — please rate again shortly.",
        ) from exc
    except Exception as exc:
        logger.error("Failed to capture rating context: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=http_status.HTTP_502_BAD_GATEWAY,
            detail="Could not retrieve the weather behind this evening's score.",
        ) from exc

    record = {
        "schema_version": 1,
        "recorded_at": utcnow().isoformat(),
        "target_date": str(target_date),
        "latitude": body.latitude,
        "longitude": body.longitude,
        "location_name": _sanitize(body.location_name),
        "rating": body.rating,
        "notes": _sanitize(body.notes),
        # What the model said at capture time — the thing we are measuring.
        "predicted_score": prediction.beauty_score_0_100,
        "predicted_category": prediction.category,
        "predicted_confidence": prediction.confidence_0_100,
        "algorithm_version": prediction.algorithm_version,
        "ml_model_used": prediction.ml_model_used,
        "physics_breakdown": prediction.physics_component_breakdown.model_dump(),
        "window_scores": prediction.window_scores,
        "best_window_point": prediction.best_window_point,
        # Raw inputs — lets any future scoring change be replayed offline.
        "window_snapshots": [s.model_dump(mode="json") for s in snapshots],
    }

    total = await store.append(record)
    logger.info(
        "Stored rating %d/5 for %s at (%.3f, %.3f); model said %.1f. Total ratings: %d",
        body.rating, target_date, body.latitude, body.longitude,
        prediction.beauty_score_0_100, total,
    )

    return RatingResponse(
        success=True,
        message=_thanks_message(body.rating, prediction.beauty_score_0_100),
        rated_date=target_date,
        predicted_score=prediction.beauty_score_0_100,
        total_ratings=total,
    )


@router.get(
    "/ratings/stats",
    response_model=RatingStats,
    summary="Progress toward a usable training set",
    description=(
        "Aggregates the stored ratings. The key number is spearman_vs_model: "
        "rank correlation between what people saw and what the engine predicted."
    ),
)
async def rating_stats(request: Request) -> RatingStats:
    store = request.app.state.rating_store
    # Deduplicated: a changed mind is one observation, not two.
    records = store.latest_per_evening()

    if not records:
        return RatingStats(
            total_ratings=0,
            distinct_locations=0,
            distinct_dates=0,
            rating_histogram={},
            note="No ratings yet. Rate an evening to start the dataset.",
        )

    histogram: dict[int, int] = {}
    locations: set[tuple[float, float]] = set()
    dates: set[str] = set()
    human: list[float] = []
    model: list[float] = []

    for rec in records:
        r = rec.get("rating")
        if isinstance(r, int):
            histogram[r] = histogram.get(r, 0) + 1
        locations.add((round(rec.get("latitude", 0.0), 2), round(rec.get("longitude", 0.0), 2)))
        dates.add(str(rec.get("target_date")))
        pred = rec.get("predicted_score")
        if isinstance(r, int) and isinstance(pred, (int, float)):
            human.append(float(r))
            model.append(float(pred))

    rho: Optional[float] = None
    if len(human) >= _MIN_PAIRS_FOR_SPEARMAN:
        rho = spearman(human, model)

    return RatingStats(
        total_ratings=len(records),
        distinct_locations=len(locations),
        distinct_dates=len(dates),
        rating_histogram=dict(sorted(histogram.items())),
        mean_rating=round(sum(human) / len(human), 2) if human else None,
        spearman_vs_model=round(rho, 4) if rho is not None else None,
        spearman_sample_size=len(human),
        note=_stats_note(len(records), histogram, rho),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(text: str) -> str:
    """Strip control characters from user-supplied text."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()


def _thanks_message(rating: int, predicted: float) -> str:
    """Acknowledge the rating, and say plainly when the model disagreed."""
    # Map the 1–5 label onto the 0–100 scale to compare like with like.
    implied = (rating - 1) / 4.0 * 100.0
    gap = predicted - implied
    if gap > 30:
        return "Noted — the model oversold that one. That's exactly the case it needs to learn."
    if gap < -30:
        return "Noted — the model undersold that one. Useful."
    return "Thanks — logged."


def _stats_note(total: int, histogram: dict[int, int], rho: Optional[float]) -> str:
    """Say what the dataset still needs, rather than only what it has."""
    if rho is None:
        return (
            f"{total} rating(s). Need {_MIN_PAIRS_FOR_SPEARMAN} before a rank "
            "correlation means anything."
        )
    low = sum(histogram.get(k, 0) for k in (1, 2))
    high = sum(histogram.get(k, 0) for k in (4, 5))
    if low == 0 or high == 0:
        return (
            f"rho={rho:.2f}, but the ratings are one-sided. A model cannot learn "
            "'bad' from a set with no bad nights — keep rating the dull evenings too."
        )
    return f"rho={rho:.2f} over {total} ratings ({low} poor, {high} good)."
