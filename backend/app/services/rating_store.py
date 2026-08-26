"""Append-only store for human sunset ratings (ML training labels).

STORAGE FORMAT
--------------
Newline-delimited JSON, one record per rating. Chosen over a database because:
  - the write volume is one row per evening per user,
  - the read pattern is "load the whole thing into pandas at training time",
  - it needs no migration, no schema, and no extra service.

Each record carries the RAW window snapshots, not just the score, so that a
future scoring change can be replayed offline against the same labels without
refetching a year of weather history.

DEPLOYMENT CAVEAT
-----------------
The Render free tier has an EPHEMERAL filesystem — anything written here is
lost on redeploy or restart. Point RATINGS_PATH at a mounted persistent disk
before relying on this in production, or treat the deployed copy as
best-effort and collect locally. See docs/scoring-v2-plan.md.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


class RatingStore:
    """Append-only JSONL store. Safe for concurrent writes within one process."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def append(self, record: dict[str, Any]) -> int:
        """Append *record* and return the new total row count.

        The write is serialised through an asyncio lock and flushed to disk so
        a crash between ratings cannot lose an acknowledged row.
        """
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, default=str, ensure_ascii=False)
            # Open in append mode per write: the file stays consistent even if
            # the process dies, and the volume never justifies a held handle.
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            return self._count_unlocked()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield stored records, skipping any corrupt lines.

        A truncated final line (power loss mid-write) must not make the whole
        dataset unreadable, so parse failures are logged and skipped.
        """
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping corrupt rating record at %s:%d", self._path, lineno
                    )

    def count(self) -> int:
        return self._count_unlocked()

    def _count_unlocked(self) -> int:
        return sum(1 for _ in self.iter_records())

    def latest_per_evening(self) -> list[dict[str, Any]]:
        """One record per (date, location) — the last rating given.

        The store is append-only, so a user who taps "dull", reconsiders and
        taps "pleasant" leaves TWO records behind. find() already resolves that
        with last-write-wins, but anything reading the file in bulk — the stats
        endpoint, the accuracy check in scripts/evaluate.py — was counting both
        and treating a changed mind as two independent observations.

        With a handful of labels that is not a rounding error: it inflates the
        count that gates the correlation, and it double-weights exactly the
        evenings someone was uncertain about.
        """
        latest: dict[tuple[str, float, float], dict[str, Any]] = {}
        for rec in self.iter_records():
            key = (
                str(rec.get("target_date")),
                round(float(rec.get("latitude", 0.0)), 2),
                round(float(rec.get("longitude", 0.0)), 2),
            )
            latest[key] = rec  # later lines win
        return list(latest.values())

    def find(
        self, latitude: float, longitude: float, target_date: str, tolerance_deg: float = 0.05
    ) -> Optional[dict[str, Any]]:
        """Return the most recent rating for this date near this location, if any.

        Coordinates are matched within *tolerance_deg* (~5 km at the default)
        so a slightly jittery GPS fix still counts as "already rated tonight".
        """
        match: Optional[dict[str, Any]] = None
        for rec in self.iter_records():
            if rec.get("target_date") != target_date:
                continue
            if abs(rec.get("latitude", 999) - latitude) > tolerance_deg:
                continue
            if abs(rec.get("longitude", 999) - longitude) > tolerance_deg:
                continue
            match = rec  # keep scanning; last write wins
        return match
