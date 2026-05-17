"""
Email service — sends sunset photo submissions to the developer via Resend.

Uses the Resend Python SDK (HTTPS API call) instead of SMTP, which avoids
port-blocking issues on cloud hosting platforms.

The service is disabled (raises EmailNotConfiguredError) when RESEND_API_KEY
is not configured, so the endpoint returns a clear 503 rather than a cryptic error.
"""
from __future__ import annotations

import asyncio
import base64
from datetime import date
from functools import partial

import resend

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.prediction import PredictResponse

logger = get_logger(__name__)

# Maximum image size accepted (bytes). Checked before any API call.
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


class EmailNotConfiguredError(RuntimeError):
    """Raised when Resend settings are missing."""


class EmailService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def is_configured(self) -> bool:
        s = self._settings
        return bool(s.RESEND_API_KEY and s.DEVELOPER_EMAIL)

    def _require_configured(self) -> None:
        if not self.is_configured:
            raise EmailNotConfiguredError(
                "Email submission is not configured on this server. "
                "Set RESEND_API_KEY and DEVELOPER_EMAIL."
            )

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def send_submission(
        self,
        *,
        image_bytes: bytes,
        image_filename: str,
        submission_date: date,
        latitude: float,
        longitude: float,
        location_name: str,
        user_message: str,
        prediction: PredictResponse,
    ) -> None:
        """
        Build and send the submission email via Resend.

        Runs the blocking SDK call in a thread so the event loop is not stalled.
        """
        self._require_configured()

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image is too large ({len(image_bytes) / 1_048_576:.1f} MB). "
                f"Maximum allowed size is {MAX_IMAGE_BYTES // 1_048_576} MB."
            )

        params = self._build_params(
            image_bytes=image_bytes,
            image_filename=image_filename,
            submission_date=submission_date,
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            user_message=user_message,
            prediction=prediction,
        )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(self._send, params))

        logger.info(
            "Photo submission sent to %s (date=%s, location=%.4f,%.4f, score=%.1f)",
            self._settings.DEVELOPER_EMAIL,
            submission_date,
            latitude,
            longitude,
            prediction.beauty_score_0_100,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_params(
        self,
        *,
        image_bytes: bytes,
        image_filename: str,
        submission_date: date,
        latitude: float,
        longitude: float,
        location_name: str,
        user_message: str,
        prediction: PredictResponse,
    ) -> resend.Emails.SendParams:
        s = self._settings
        p = prediction
        ws = p.weather_summary

        subject = (
            f"[Afterglow] Photo submission — "
            f"{location_name or f'{latitude:.4f}, {longitude:.4f}'}, {submission_date}"
        )

        reasons_text = "\n".join(f"  • {r}" for r in p.reasons)
        window_scores_text = (
            "\n".join(f"  {label}: {score:.1f}" for label, score in p.window_scores.items())
            if p.window_scores
            else "  (not available)"
        )

        body = f"""\
Sunset photo submitted via Afterglow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Date:          {submission_date}
Location:      {location_name or '(unnamed)'}  ({latitude:.6f}, {longitude:.6f})
User message:  {user_message.strip() or '(none)'}

━━  Algorithm Results  ━━━━━━━━━━━━━━━━━━━━━━━━━

Score:         {p.beauty_score_0_100:.1f} / 100  ({p.category})
Confidence:    {p.confidence_0_100:.0f}%
Best window:   {p.best_window_point}
Sunset time:   {p.sunset_time}

Window scores:
{window_scores_text}

Why:
{reasons_text}

━━  Weather at Sunset  ━━━━━━━━━━━━━━━━━━━━━━━━━

Cloud total:   {ws.cloud_total_pct:.0f}%
Cloud high:    {ws.cloud_high_pct:.0f}%
Cloud mid:     {ws.cloud_mid_pct:.0f}%
Cloud low:     {ws.cloud_low_pct:.0f}%
Visibility:    {ws.visibility_km:.1f} km
Humidity:      {ws.humidity_pct:.0f}%
Precipitation: {ws.precipitation_mm:.1f} mm
Temperature:   {ws.temperature_c:.1f}°C
Wind:          {ws.wind_speed_kmh:.1f} km/h
{f"AOD:           {ws.aerosol_optical_depth:.3f}{' (estimated)' if ws.aerosol_is_estimated else ''}" if ws.aerosol_optical_depth is not None else ""}

━━  Meta  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Algorithm:     v{p.algorithm_version}
ML model used: {p.ml_model_used}
ML adjustment: {p.ml_adjustment if p.ml_adjustment is not None else 'N/A'}
"""

        filename = image_filename or "sunset.jpg"
        encoded_content = base64.b64encode(image_bytes).decode()

        return {
            "from": s.RESEND_FROM_EMAIL,
            "to": [s.DEVELOPER_EMAIL],
            "subject": subject,
            "text": body,
            "attachments": [
                {
                    "filename": filename,
                    "content": encoded_content,
                }
            ],
        }

    def _send(self, params: resend.Emails.SendParams) -> None:
        """Blocking Resend API call — invoke via run_in_executor."""
        resend.api_key = self._settings.RESEND_API_KEY
        resend.Emails.send(params)
