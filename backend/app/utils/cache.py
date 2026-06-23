"""Thread-safe in-memory TTL cache for weather lookups, with optional disk persistence."""
from __future__ import annotations

import hashlib
import os
import pickle
import threading
import time
from typing import Any, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


class TTLCache:
    """
    Key-value cache with per-entry TTL expiry.

    Thread-safe via a reentrant lock. Expired entries are evicted lazily
    on access and proactively on every 100th set() call.

    When *persist_path* is provided the store is mirrored to disk so cached
    weather survives process restarts (e.g. ``uvicorn --reload``), avoiding a
    full re-fetch — and the Open-Meteo rate-limit pressure that comes with it.
    Expiry uses wall-clock time so TTLs remain meaningful across restarts.
    """

    def __init__(self, ttl_seconds: int = 900, persist_path: Optional[str] = None) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)
        self._lock = threading.RLock()
        self._set_count = 0
        self._persist_path = persist_path or None
        if self._persist_path:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if missing / expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_override: Optional[int] = None) -> None:
        """Store *value* under *key* for TTL seconds (or ttl_override if given)."""
        ttl = ttl_override if ttl_override is not None else self._ttl
        with self._lock:
            expires_at = time.time() + ttl
            self._store[key] = (value, expires_at)
            self._set_count += 1
            if self._set_count % 100 == 0:
                self._evict_expired()
            self._persist()

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)
            self._persist()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._persist()

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(*args: Any) -> str:
        """
        Create a stable string cache key from arbitrary arguments.

        Example:
            key = TTLCache.make_key("weather", 37.77, -122.41, "2024-06-21")
        """
        raw = "|".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]

    def _load(self) -> None:
        """Load persisted entries from disk, dropping any already expired.

        Best-effort: a missing or corrupt cache file is treated as an empty
        cache rather than a fatal error.
        """
        try:
            with open(self._persist_path, "rb") as fh:  # type: ignore[arg-type]
                data: dict[str, tuple[Any, float]] = pickle.load(fh)
        except FileNotFoundError:
            return
        except Exception as exc:  # corrupt file, unpickling error, etc.
            logger.warning("Could not load weather cache from %s: %s", self._persist_path, exc)
            return
        now = time.time()
        self._store = {k: (v, exp) for k, (v, exp) in data.items() if exp > now}
        logger.info("Loaded %d cached weather entries from %s", len(self._store), self._persist_path)

    def _persist(self) -> None:
        """Atomically write the store to disk (best-effort; caller holds the lock)."""
        if not self._persist_path:
            return
        tmp = f"{self._persist_path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(self._store, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self._persist_path)
        except Exception as exc:
            logger.warning("Could not persist weather cache to %s: %s", self._persist_path, exc)
            try:
                os.remove(tmp)
            except OSError:
                pass
