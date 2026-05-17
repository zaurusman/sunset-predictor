"""Schema for the POST /submit-photo response."""
from __future__ import annotations

from pydantic import BaseModel


class SubmitPhotoResponse(BaseModel):
    """Response returned after a successful photo submission."""

    success: bool
    message: str
