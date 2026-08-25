"""Persistent store for Web Push subscriptions.

STORAGE FORMAT
--------------
A single JSON object, endpoint-hash → record, rewritten in full on every
change. Unlike an append-only log this needs updates and deletes (a
subscription's location changes, and a dead endpoint must be removed), and the
volume — one record per device — never justifies a database.

Writes go through a temp file and os.replace() so a crash mid-write cannot
leave a truncated file that loses every subscriber at once.

DEPLOYMENT CAVEAT
-----------------
Render's free tier has an EPHEMERAL filesystem: this file is wiped on every
redeploy and on the restart that follows the free tier's idle sleep. Everyone
silently stops receiving alerts and has to opt in again, with nothing in the UI
to tell them. Point SUBSCRIPTIONS_PATH at a mounted persistent disk before
treating notifications as a real feature.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# A push endpoint is a long URL; hash it for a stable, filename-safe key.
_KEY_LENGTH = 32


def endpoint_key(endpoint: str) -> str:
    """Stable id for a push endpoint."""
    return hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:_KEY_LENGTH]


class SubscriptionStore:
    """JSON-file store of push subscriptions. Serialised through one lock."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()
        self._records: dict[str, dict[str, Any]] = self._read()

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            # Losing subscribers is bad, but refusing to boot is worse: the rest
            # of the API must keep serving predictions.
            logger.error(
                "Could not read subscriptions from %s (%s) — starting empty",
                self._path,
                exc,
            )
            return {}
        if not isinstance(data, dict):
            logger.error("Subscriptions file %s is not an object — starting empty", self._path)
            return {}
        return {k: v for k, v in data.items() if isinstance(v, dict)}

    def all(self) -> list[dict[str, Any]]:
        """Every stored record. Copies, so callers cannot mutate the store."""
        return [dict(rec) for rec in self._records.values()]

    def get(self, endpoint: str) -> Optional[dict[str, Any]]:
        rec = self._records.get(endpoint_key(endpoint))
        return dict(rec) if rec else None

    def count(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert(self, record: dict[str, Any]) -> dict[str, Any]:
        """Insert or replace the record for `record["endpoint"]`.

        Re-subscribing keeps `created_at` from the original record so the
        history of a long-standing subscriber survives a location change.
        """
        key = endpoint_key(record["endpoint"])
        async with self._lock:
            existing = self._records.get(key)
            if existing and existing.get("created_at"):
                record = {**record, "created_at": existing["created_at"]}
            self._records[key] = record
            self._flush_unlocked()
            return dict(record)

    async def update_fields(self, endpoint: str, **fields: Any) -> None:
        """Patch named fields on one record; a no-op if it is already gone."""
        key = endpoint_key(endpoint)
        async with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return
            rec.update(fields)
            self._flush_unlocked()

    async def delete(self, endpoint: str) -> bool:
        """Remove one subscription. Returns whether anything was removed."""
        key = endpoint_key(endpoint)
        async with self._lock:
            if self._records.pop(key, None) is None:
                return False
            self._flush_unlocked()
            return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_unlocked(self) -> None:
        """Atomically rewrite the whole file. Caller holds the lock."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".subscriptions-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._records, f, default=str, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except Exception:
            # Never leave the temp file behind on a failed write.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
