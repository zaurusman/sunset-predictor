"""Model artifact path management and metadata loading."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Manages the on-disk locations of trained model artifacts.

    Paths are resolved relative to the current working directory (the backend/
    folder when running the server normally, or an overridden base path for tests).
    """

    def __init__(self, settings: Settings, base_dir: Optional[Path] = None) -> None:
        self._settings = settings
        self._base = base_dir or Path.cwd()

    @property
    def model_path(self) -> Path:
        return self._resolve(self._settings.MODEL_PATH)

    @property
    def metadata_path(self) -> Path:
        return self._resolve(self._settings.MODEL_METADATA_PATH)

    def model_exists(self) -> bool:
        return self.model_path.exists()

    def metadata_exists(self) -> bool:
        return self.metadata_path.exists()

    def load_metadata(self) -> dict[str, Any]:
        """Load model_metadata.json; return an empty dict if absent."""
        if not self.metadata_exists():
            return {}
        try:
            with open(self.metadata_path) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load model metadata: %s", exc)
            return {}

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Saved model metadata to %s", self.metadata_path)

    def _resolve(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self._base / p
