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
from app.services.rating_store import RATING_SCHEMA_VERSION, band_of, dedupe_latest, label_0_100
from app.services.weather_service import WeatherUnavailableError
from app.utils.call_budget import OTHER, TONIGHT, priority
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
    #
    # The label is the scarce part: an evening can only be rated by someone
    # who saw it, while its weather can be refetched from Open-Meteo's archive
    # at any time. So a weather failure here (the shared Render IP refused,
    # the quota spent) costs the raw inputs, never the rating — the record is
    # stored with `context_error` set, and the training export backfills it.
    # Tonight's rating is tonight's work for the call budget.
    # ------------------------------------------------------------------
    prediction = None
    snapshots: list = []
    corridor: list = []
    context_error: Optional[str] = None
    try:
        with priority(TONIGHT if target_date == today else OTHER):
            prediction, snapshots, corridor = await svc.capture_rating_context(
                lat=body.latitude,
                lon=body.longitude,
                target_date=target_date,
                horizon_deg=settings.DEFAULT_HORIZON_OBSTRUCTION_DEG,
            )
    except WeatherUnavailableError as exc:
        context_error = f"weather unavailable: {exc}"[:300]
        logger.warning("Storing rating without weather context: %s", exc)
    except Exception as exc:
        context_error = f"{type(exc).__name__}: {exc}"[:300]
        logger.error("Failed to capture rating context: %s", exc, exc_info=True)

    record = {
        # See RATING_SCHEMA_VERSION for what each version added. Readers must
        # tolerate every earlier one — the store is append-only, never migrated.
        "schema_version": RATING_SCHEMA_VERSION,
        "recorded_at": utcnow().isoformat(),
        "target_date": str(target_date),
        "latitude": body.latitude,
        "longitude": body.longitude,
        "location_name": _sanitize(body.location_name),
        # The canonical label, on the same 0-100 scale the app displays.
        # Always present; derived from the coarse scale when that is all the
        # caller had (see RatingRequest.score_0_100).
        "rating_0_100": body.score_0_100(),
        # True when the number above was actually chosen on 0-100, rather than
        # inferred from a five-way tap. Analysis should be able to tell the
        # difference between a considered 83 and a tap that means "somewhere
        # in the 70s or 80s".
        "rating_is_precise": body.rating_0_100 is not None,
        # The coarse scale, kept for the one-tap UI and for every record
        # written before rating_0_100 existed.
        "rating": body.rating,
        "notes": _sanitize(body.notes),
        # Which sampled moment this rating actually describes, if known — a
        # sunset-lit-cloud evening and an afterglow-gradient one are different
        # events that share a date. None means "the evening as a whole".
        "observed_moment": body.observed_moment,
        # The horizon obstruction the evening was scored with — an input to
        # the engine, so a replay needs it.
        "horizon_deg": settings.DEFAULT_HORIZON_OBSTRUCTION_DEG,
        # Why the fields below are empty, when they are. None when captured.
        "context_error": context_error,
        # What the model said at capture time — the thing we are measuring.
        "predicted_score": prediction.beauty_score_0_100 if prediction else None,
        "predicted_category": prediction.category if prediction else None,
        "predicted_confidence": prediction.confidence_0_100 if prediction else None,
        "raw_physics_score": prediction.raw_physics_score if prediction else None,
        "climatology_percentile": prediction.climatology_percentile if prediction else None,
        "algorithm_version": (
            prediction.algorithm_version if prediction else settings.ALGORITHM_VERSION
        ),
        "ml_model_used": prediction.ml_model_used if prediction else None,
        "physics_breakdown": (
            prediction.physics_component_breakdown.model_dump() if prediction else None
        ),
        "window_scores": prediction.window_scores if prediction else {},
        "best_window_point": prediction.best_window_point if prediction else None,
        # What the model scored at the SPECIFIC moment this rating describes,
        # when that's known — the number this rating should actually be
        # compared against. Falls back to the evening's aggregated score when
        # the moment wasn't given.
        "predicted_score_at_observed_moment": (
            prediction.window_scores.get(body.observed_moment)
            if prediction and body.observed_moment else None
        ),
        # Raw inputs — lets any future scoring change, or a trained model, be
        # replayed offline against this label.
        "window_snapshots": [s.model_dump(mode="json") for s in snapshots],
        # (distance_km, cloud_low_pct, cloud_mid_pct) toward the setting sun.
        "corridor_samples": [list(c) for c in corridor],
    }

    try:
        total = await store.append(record)
    except Exception as exc:
        # Never acknowledge a rating that was not saved.
        logger.error("Failed to store rating: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not save that rating right now — please try again shortly.",
        ) from exc

    predicted = prediction.beauty_score_0_100 if prediction else None
    logger.info(
        "Stored rating %.0f/100 for %s at (%.3f, %.3f); model said %s. Total ratings: %d",
        body.score_0_100(), target_date, body.latitude, body.longitude,
        f"{predicted:.1f}" if predicted is not None else "n/a (no weather)", total,
    )

    return RatingResponse(
        success=True,
        message=(
            _thanks_message(body.score_0_100(), predicted)
            if predicted is not None else "Thanks — logged."
        ),
        rated_date=target_date,
        predicted_score=predicted,
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
    # Deduplicated: a changed mind is one observation, not two. Raw inputs
    # stay in the database — nothing here reads them.
    records = dedupe_latest(await store.records(with_raw=False))

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
        label = label_0_100(rec)
        if label is not None:
            # Histogram stays on the coarse 1-5 bands: 100 buckets over a
            # handful of ratings is not a distribution anyone can read, and
            # the question it answers ("are both ends represented?") is a
            # question about bands.
            histogram[band_of(label)] = histogram.get(band_of(label), 0) + 1
        locations.add((round(rec.get("latitude", 0.0), 2), round(rec.get("longitude", 0.0), 2)))
        dates.add(str(rec.get("target_date")))
        # Prefer the score at the specific moment this rating describes — a
        # sunset-lit-cloud reading and an afterglow-gradient reading of the
        # same evening are different physical events. Falls back to the
        # aggregated evening score for ratings that predate this field.
        pred = rec.get("predicted_score_at_observed_moment")
        if pred is None:
            pred = rec.get("predicted_score")
        if label is not None and isinstance(pred, (int, float)):
            human.append(label)
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


def _thanks_message(rating_0_100: float, predicted: float) -> str:
    """Acknowledge the rating, and say plainly when the model disagreed.

    Both numbers are already on the same 0-100 scale, so the gap is a direct
    subtraction rather than a conversion — which is the point of the scale.
    """
    gap = predicted - rating_0_100
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
