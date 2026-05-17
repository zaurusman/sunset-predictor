"""
Email service — sends sunset photo submissions to the developer.

Uses stdlib smtplib so there are no extra dependencies beyond
python-multipart (already added for file upload parsing).

The service is disabled (raises ServiceUnavailableError) when SMTP_HOST
is not configured, so the endpoint can return a clear 503 rather than a
confusing SMTP connection error.
"""
from __future__ import annotations

import asyncio
import smtplib
import ssl
from datetime import date
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import partial

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.prediction import PredictResponse

logger = get_logger(__name__)

# Maximum image size accepted (bytes).  Checked before any SMTP work.
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


class EmailNotConfiguredError(RuntimeError):
    """Raised when SMTP settings are missing."""


class EmailService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def is_configured(self) -> bool:
        s = self._settings
        return bool(s.SMTP_HOST and s.SMTP_USERNAME and s.SMTP_PASSWORD and s.DEVELOPER_EMAIL)

    def _require_configured(self) -> None:
        if not self.is_configured:
            raise EmailNotConfiguredError(
                "Email submission is not configured on this server. "
                "Set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, and DEVELOPER_EMAIL."
            )

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def send_submission(
        self,
        *,
        image_bytes: bytes,
        image_filename: str,
        content_type: str,
        submission_date: date,
        latitude: float,
        longitude: float,
        location_name: str,
        user_message: str,
        prediction: PredictResponse,
    ) -> None:
        """
        Build and send the submission email.

        Runs the blocking smtplib call in a thread so the event loop is
        not stalled.
        """
        self._require_configured()

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image is too large ({len(image_bytes) / 1_048_576:.1f} MB). "
                f"Maximum allowed size is {MAX_IMAGE_BYTES // 1_048_576} MB."
            )

        msg = self._build_message(
            image_bytes=image_bytes,
            image_filename=image_filename,
            content_type=content_type,
            submission_date=submission_date,
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            user_message=user_message,
            prediction=prediction,
        )

        # Run blocking SMTP call in a thread executor
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(self._send_smtp, msg))

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

    def _build_message(
        self,
        *,
        image_bytes: bytes,
        image_filename: str,
        content_type: str,
        submission_date: date,
        latitude: float,
        longitude: float,
        location_name: str,
        user_message: str,
        prediction: PredictResponse,
    ) -> MIMEMultipart:
        s = self._settings
        from_addr = s.SMTP_FROM or s.SMTP_USERNAME
        to_addr = s.DEVELOPER_EMAIL

        subject = (
            f"[Sunset Predictor] Photo submission — "
            f"{location_name or f'{latitude:.4f}, {longitude:.4f}'}, {submission_date}"
        )

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr

        # ── Text body ─────────────────────────────────────────────────────────
        p = prediction
        ws = p.weather_summary
        reasons_text = "\n".join(f"  • {r}" for r in p.reasons)
        window_scores_text = (
            "\n".join(f"  {label}: {score:.1f}" for label, score in p.window_scores.items())
            if p.window_scores
            else "  (not available)"
        )

        body = f"""\
Sunset photo submitted via Sunset Predictor
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
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # ── Image attachment ───────────────────────────────────────────────────
        main_type, sub_type = content_type.split("/", 1) if "/" in content_type else ("image", "jpeg")
        attachment = MIMEBase(main_type, sub_type)
        attachment.set_payload(image_bytes)
        encoders.encode_base64(attachment)
        attachment.add_header(
            "Content-Disposition",
            "attachment",
            filename=image_filename or "sunset.jpg",
        )
        msg.attach(attachment)

        return msg

    def _send_smtp(self, msg: MIMEMultipart) -> None:
        """Blocking SMTP send — call via run_in_executor."""
        s = self._settings
        context = ssl.create_default_context()

        if s.SMTP_PORT == 465:
            # SSL from the start
            with smtplib.SMTP_SSL(s.SMTP_HOST, s.SMTP_PORT, context=context) as server:
                server.login(s.SMTP_USERNAME, s.SMTP_PASSWORD)
                server.send_message(msg)
        else:
            # STARTTLS (port 587 is standard)
            with smtplib.SMTP(s.SMTP_HOST, s.SMTP_PORT) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(s.SMTP_USERNAME, s.SMTP_PASSWORD)
                server.send_message(msg)
