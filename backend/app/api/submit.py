"""POST /submit-photo endpoint — receive a sunset photo and email it to the developer."""
from __future__ import annotations

import re
from datetime import date

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi import status as http_status

from app.core.logging import get_logger
from app.schemas.prediction import PredictRequest
from app.schemas.submission import SubmitPhotoResponse
from app.services.email_service import EmailNotConfiguredError, EmailService

logger = get_logger(__name__)
router = APIRouter(tags=["submission"])

# Accepted image MIME types
_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}


@router.post(
    "/submit-photo",
    response_model=SubmitPhotoResponse,
    summary="Submit a sunset photo",
    description=(
        "Upload a sunset photo with its date and location. "
        "The server emails it to the developer together with the algorithm's stats "
        "for that location and date."
    ),
)
async def submit_photo(
    request: Request,
    photo: UploadFile,
    latitude: float = Form(..., ge=-90, le=90, description="Decimal latitude of the photo location"),
    longitude: float = Form(..., ge=-180, le=180, description="Decimal longitude of the photo location"),
    photo_date: date = Form(..., description="Date the photo was taken (YYYY-MM-DD)"),
    location_name: str = Form(default="", description="Human-readable location name"),
    user_message: str = Form(default="", max_length=1000, description="Optional message from the user"),
) -> SubmitPhotoResponse:
    # ------------------------------------------------------------------
    # Validate file type
    # ------------------------------------------------------------------
    content_type = (photo.content_type or "").lower()
    if content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=http_status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type '{content_type}'. Accepted: JPEG, PNG, WebP, HEIC.",
        )

    # ------------------------------------------------------------------
    # Read image bytes (size guard inside EmailService)
    # ------------------------------------------------------------------
    image_bytes = await photo.read()

    # ------------------------------------------------------------------
    # Fetch prediction stats for the submitted date/location
    # ------------------------------------------------------------------
    svc = request.app.state.prediction_service
    try:
        prediction = await svc.predict(
            PredictRequest(
                latitude=latitude,
                longitude=longitude,
                target_date=photo_date,
            )
        )
    except Exception as exc:
        logger.warning("Could not fetch stats for submission (date=%s): %s", photo_date, exc)
        raise HTTPException(
            status_code=http_status.HTTP_502_BAD_GATEWAY,
            detail="Could not retrieve weather stats for that date. Please try again.",
        ) from exc

    # ------------------------------------------------------------------
    # Send email
    # ------------------------------------------------------------------
    email_svc = EmailService(settings=request.app.state.settings)
    try:
        await email_svc.send_submission(
            image_bytes=image_bytes,
            image_filename=photo.filename or "sunset.jpg",
            submission_date=photo_date,
            latitude=latitude,
            longitude=longitude,
            location_name=_sanitize(location_name),
            user_message=_sanitize(user_message),
            prediction=prediction,
        )
    except EmailNotConfiguredError as exc:
        logger.error("Email not configured: %s", exc)
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Photo submission is not available on this server.",
        ) from exc
    except ValueError as exc:
        # e.g. image too large
        raise HTTPException(status_code=http_status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to send submission email: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send the submission. Please try again later.",
        ) from exc

    return SubmitPhotoResponse(
        success=True,
        message="Your sunset photo has been submitted. Thank you!",
    )


def _sanitize(text: str) -> str:
    """Strip control characters from user-supplied text."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()
