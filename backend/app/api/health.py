"""GET /health endpoint."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health", summary="Health check")
async def health(request: Request) -> dict:
    """
    Returns application health status, algorithm version, and ML model info.
    """
    ml_model = request.app.state.ml_model
    settings = request.app.state.settings

    return {
        "status": "ok",
        "algorithm_version": settings.ALGORITHM_VERSION,
        "environment": settings.APP_ENV,
        "ml_model_loaded": ml_model.is_loaded(),
        "model_metadata": ml_model.get_metadata(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
