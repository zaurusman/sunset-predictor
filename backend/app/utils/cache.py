"""Thread-safe in-memory TTL cache for weather lookups, with optional disk persistence."""
from __future__ import annotations

import asyncio
import hashlib
import os
import pickle
import threading
import time
from typing import Any, Optional

from app.core.logging import get_logger
from app.utils.durable_cache import DURABLE_MIN_TTL_SECONDS

logger = get_logger(__name__)


class TTLCache:
    """
    Key-value cache with per-entry TTL expiry.

    Thread-safe via a reentrant lock. Expired entries are evicted lazily
    on access and proactively on every 100th set() call.

    An expired entry is not dropped straight away: it is kept for a further
    *stale_grace_seconds* so :meth:`get_stale` can hand it back when a fresh
    fetch fails. ``get()`` never returns stale data.

    When *persist_path* is provided the store is mirrored to disk so cached
    weather survives process restarts (e.g. ``uvicorn --reload``), avoiding a
    full re-fetch — and the Open-Meteo rate-limit pressure that comes with it.
    Expiry uses wall-clock time so TTLs remain meaningful across restarts.

    Render's disk is ephemeral, so /tmp alone doesn't survive a deploy. An
    optional durable tier (:meth:`attach_durable`) keeps the long-lived
    entries — TTL of a day or more — in Postgres as well.
    """

    def __init__(
        self,
        ttl_seconds: int = 900,
        persist_path: Optional[str] = None,
        stale_grace_seconds: int = 0,
    ) -> None:
        self._ttl = ttl_seconds
        self._grace = stale_grace_seconds
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)
        self._lock = threading.RLock()
        self._set_count = 0
        self._persist_path = persist_path or None
        if self._persist_path:
            self._load()
        # Durable tier (see attach_durable). None until attached.
        self._durable = None
        self._durable_loop: Optional[asyncio.AbstractEventLoop] = None
        self._durable_queue: Optional[asyncio.Queue] = None
        self._durable_task: Optional[asyncio.Task] = None
        self._durable_linger = 0.0
        self._last_purge = 0.0

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
            now = time.time()
            if now > expires_at:
                if now > expires_at + self._grace:
                    del self._store[key]
                return None
            return value

    def get_stale(self, key: str) -> Optional[Any]:
        """Return the value even if expired, as long as it is within the
        stale grace period. For fallback use only, when a fresh fetch failed."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at + self._grace:
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
        if ttl >= DURABLE_MIN_TTL_SECONDS:
            self._enqueue_durable(key, value, expires_at)

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
    # Durable tier
    # ------------------------------------------------------------------

    async def attach_durable(self, tier, linger_seconds: float = 2.0) -> int:
        """Bulk-load *tier*'s unexpired rows and start writing long-lived
        entries through to it. Returns the number of entries loaded.

        Best-effort throughout: if the database is unreachable the cache keeps
        working exactly as it would without a durable tier.
        """
        loaded = 0
        try:
            rows = await tier.load_all(self._grace)
        except Exception as exc:
            logger.warning("Durable cache load failed; starting without it: %s", exc)
            rows = []
        with self._lock:
            for key, blob, expires_at in rows:
                current = self._store.get(key)
                if current is not None and current[1] >= expires_at:
                    continue
                try:
                    value = pickle.loads(blob)
                except Exception as exc:
                    logger.warning("Skipping undecodable durable cache row %s: %s", key, exc)
                    continue
                self._store[key] = (value, expires_at)
                loaded += 1
            if loaded:
                self._persist()
        logger.info("Loaded %d long-lived cache entries from the durable tier", loaded)

        await self._purge(tier)
        self._durable = tier
        self._durable_linger = linger_seconds
        self._durable_loop = asyncio.get_running_loop()
        self._durable_queue = asyncio.Queue()
        self._durable_task = asyncio.create_task(self._durable_writer())
        return loaded

    async def close_durable(self) -> None:
        """Flush pending writes and stop the writer (call on shutdown)."""
        if self._durable_task is None:
            return
        self._durable_queue.put_nowait(None)
        try:
            await self._durable_task
        finally:
            self._durable = self._durable_task = self._durable_queue = self._durable_loop = None

    def _enqueue_durable(self, key: str, value: Any, expires_at: float) -> None:
        if self._durable_queue is None:
            return
        try:
            if asyncio.get_running_loop() is not self._durable_loop:
                return
        except RuntimeError:  # sync caller with no loop (scripts): memory only
            return
        try:
            blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.warning("Not persisting unpicklable cache entry %s: %s", key, exc)
            return
        self._durable_queue.put_nowait((key, blob, expires_at))

    async def _durable_writer(self) -> None:
        """One background writer. A cold location sets ~100 archive months in
        one burst; the linger lets them collect so the burst costs one
        executemany (and one Neon wake-up) instead of ~100 separate writes."""
        queue = self._durable_queue
        while True:
            item = await queue.get()
            if item is None:
                return
            if self._durable_linger:
                await asyncio.sleep(self._durable_linger)
            batch = {item[0]: item}
            stop = False
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    stop = True
                else:
                    batch[item[0]] = item  # later set of the same key wins
            try:
                await self._durable.write_many(list(batch.values()))
            except Exception as exc:
                logger.warning("Durable cache write of %d entries failed: %s", len(batch), exc)
            else:
                # Purge lazily, only when a write has already woken the database.
                if time.time() - self._last_purge > DURABLE_MIN_TTL_SECONDS:
                    await self._purge(self._durable)
            if stop:
                return

    async def _purge(self, tier) -> None:
        self._last_purge = time.time()
        try:
            await tier.purge_expired(self._grace)
        except Exception as exc:
            logger.warning("Durable cache purge failed: %s", exc)

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
        expired = [k for k, (_, exp) in self._store.items() if now > exp + self._grace]
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
        self._store = {k: (v, exp) for k, (v, exp) in data.items() if exp + self._grace > now}
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
