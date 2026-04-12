"""Simple thread-safe in-memory TTL cache for weather lookups."""
from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Optional


class TTLCache:
    """
    In-memory key-value cache with per-entry TTL expiry.

    Thread-safe via a reentrant lock. Expired entries are evicted lazily
    on access and proactively on every 100th set() call.
    """

    def __init__(self, ttl_seconds: int = 900) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)
        self._lock = threading.RLock()
        self._set_count = 0

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
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key* for TTL seconds."""
        with self._lock:
            expires_at = time.monotonic() + self._ttl
            self._store[key] = (value, expires_at)
            self._set_count += 1
            if self._set_count % 100 == 0:
                self._evict_expired()

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

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
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
