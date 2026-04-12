"""GET /model/info endpoint — ML model metadata."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["model"])


@router.get("/model/info", summary="ML model metadata")
async def model_info(request: Request) -> dict:
    """
    Return full ML model metadata including training date, dataset statistics,
    feature importances, validation metrics, and blending configuration.

    Returns a minimal `{loaded: false}` object when no model has been trained.
    """
    ml_model = request.app.state.ml_model
    settings = request.app.state.settings

    metadata = ml_model.get_metadata()
    metadata["blend_alpha"] = settings.ML_BLEND_ALPHA
    metadata["algorithm_version"] = settings.ALGORITHM_VERSION

    return metadata
