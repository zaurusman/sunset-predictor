"""Thread-safe in-memory TTL cache for weather lookups, with optional disk persistence."""
from __future__ import annotations

import asyncio
import hashlib
import os
import pickle
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from app.core.logging import get_logger
from app.utils.client_fetch import client_data
from app.utils.durable_cache import (
    DURABLE_MIN_TTL_SECONDS,
    decode_value,
    encode_value,
    normalize_blob,
)

logger = get_logger(__name__)


class TTLCache:
    """
    Key-value cache with per-entry TTL expiry.

    Thread-safe via a reentrant lock. Expired entries are evicted lazily
    on access and proactively on every 100th set() call.

    An expired entry is not dropped straight away: it is kept for a further
    *stale_grace_seconds* so :meth:`get_stale` can hand it back when a fresh
    fetch fails. ``get()`` never returns stale data.

    Memory: values are held pickled and zlib-compressed (see encode_value) —
    live Python objects for this data take ~6x their pickled size and the
    pickle compresses ~6x, so this is ~35x smaller. The last *hot_entries*
    values read or written are also kept decoded, so hot reads don't pay for
    decompression. With *memory_budget_bytes* set, the least recently used
    entries are dropped once the compressed total exceeds it. Before this the
    cache kept every location for 30 days as live objects and ran Render's
    512 MB instance out of memory at ~60 locations (scripts/capacity).

    When *persist_path* is provided the store is mirrored to disk so cached
    weather survives process restarts (e.g. ``uvicorn --reload``), avoiding a
    full re-fetch — and the Open-Meteo rate-limit pressure that comes with it.
    Writes are debounced: at most one every *persist_interval* seconds, from a
    background thread — call :meth:`flush` to write now. (Re-pickling the
    whole store on every set() used to freeze the event loop for seconds per
    write once the cache held a few dozen locations.) Expiry uses wall-clock
    time so TTLs remain meaningful across restarts.

    Render's disk is ephemeral, so /tmp alone doesn't survive a deploy. An
    optional durable tier (:meth:`attach_durable`) keeps the long-lived
    entries — TTL of a day or more — in Postgres as well.
    """

    def __init__(
        self,
        ttl_seconds: int = 900,
        persist_path: Optional[str] = None,
        stale_grace_seconds: int = 0,
        memory_budget_bytes: Optional[int] = None,
        persist_interval: float = 30.0,
        hot_entries: int = 64,
    ) -> None:
        self._ttl = ttl_seconds
        self._grace = stale_grace_seconds
        self._budget = memory_budget_bytes
        self._hot_max = hot_entries
        # key -> (packed value, expires_at); order is recency, oldest first.
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._bytes = 0
        self._hot: OrderedDict[str, Any] = OrderedDict()  # decoded values
        # When each entry was set in this process (see get_fresh). Not
        # persisted: an entry loaded from disk or the durable tier has no
        # known age, so get_fresh treats it as outdated.
        self._stored_at: dict[str, float] = {}
        self._lock = threading.RLock()
        self._set_count = 0
        self._persist_path = persist_path or None
        self._persist_interval = persist_interval
        self._dirty = False
        self._timer: Optional[threading.Timer] = None
        self._io_lock = threading.Lock()
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
            packed, expires_at = entry
            now = time.time()
            if now > expires_at:
                if now > expires_at + self._grace:
                    self._drop(key)
                return None
            return self._value(key, packed)

    def get_stale(self, key: str) -> Optional[Any]:
        """Return the value even if expired, as long as it is within the
        stale grace period. For fallback use only, when a fresh fetch failed."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            packed, expires_at = entry
            if time.time() > expires_at + self._grace:
                self._drop(key)
                return None
            return self._value(key, packed)

    def get_fresh(self, key: str, since: float) -> Optional[Any]:
        """Like get(), but only an entry set at or after *since* — e.g. after
        the latest weather-model run became available (see model_runs)."""
        with self._lock:
            stored = self._stored_at.get(key)
            if stored is None or stored < since:
                return None
            return self.get(key)

    def set(self, key: str, value: Any, ttl_override: Optional[int] = None) -> None:
        """Store *value* under *key* for TTL seconds (or ttl_override if given)."""
        if client_data.get() is not None:
            return  # browser-supplied data is never shared (see client_fetch)
        ttl = ttl_override if ttl_override is not None else self._ttl
        try:
            packed: Any = encode_value(value)
        except Exception as exc:
            logger.warning("Caching unpicklable value for %s in memory only: %s", key, exc)
            packed = _Unpacked(value)
        with self._lock:
            expires_at = time.time() + ttl
            self._drop(key)
            self._store[key] = (packed, expires_at)
            self._stored_at[key] = expires_at - ttl
            self._bytes += _size(packed)
            self._remember(key, value)
            self._set_count += 1
            if self._set_count % 100 == 0:
                self._evict_expired()
            self._enforce_budget()
            self._mark_dirty()
        if ttl >= DURABLE_MIN_TTL_SECONDS and isinstance(packed, bytes):
            self._enqueue_durable(key, packed, expires_at, expires_at - ttl)

    def delete(self, key: str) -> None:
        with self._lock:
            self._drop(key)
            self._mark_dirty()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hot.clear()
            self._stored_at.clear()
            self._bytes = 0
            self._mark_dirty()

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def memory_bytes(self) -> int:
        """Compressed bytes held (what the memory budget counts)."""
        with self._lock:
            return self._bytes

    def flush(self) -> None:
        """Write the store to disk now if it changed since the last write."""
        if not self._persist_path:
            return
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False
            snapshot = {k: e for k, e in self._store.items() if isinstance(e[0], bytes)}
        with self._io_lock:
            self._write(snapshot)

    def close(self) -> None:
        """Stop the background writer and flush (call on shutdown)."""
        with self._lock:
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
        self.flush()

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
            rows = await tier.load_all(self._grace, budget_bytes=self._budget)
        except Exception as exc:
            logger.warning("Durable cache load failed; starting without it: %s", exc)
            rows = []
        with self._lock:
            # Rows arrive latest-expiry first; insert in reverse so those end
            # up most recently used — the last to go if the budget is tight.
            # Values stay compressed: nothing is decoded until it is read.
            for key, blob, expires_at, stored_at in reversed(rows):
                current = self._store.get(key)
                if current is not None and current[1] >= expires_at:
                    continue
                self._drop(key)
                packed = normalize_blob(blob)
                self._store[key] = (packed, expires_at)
                if stored_at is not None:
                    # Lets get_fresh judge it against model runs, as before the restart.
                    self._stored_at[key] = stored_at
                self._bytes += len(packed)
                loaded += 1
            if loaded:
                self._enforce_budget()
                self._mark_dirty()
        logger.info(
            "Loaded %d long-lived cache entries from the durable tier (%.1f MB compressed in memory)",
            loaded, self._bytes / 1e6,
        )

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

    def _enqueue_durable(self, key: str, blob: bytes, expires_at: float, stored_at: float) -> None:
        if self._durable_queue is None:
            return
        try:
            if asyncio.get_running_loop() is not self._durable_loop:
                return
        except RuntimeError:  # sync caller with no loop (scripts): memory only
            return
        self._durable_queue.put_nowait((key, blob, expires_at, stored_at))

    async def _durable_writer(self) -> None:
        """One background writer. A cold location sets its archive months and
        climatology curve in one burst; the linger lets them collect so the
        burst costs one executemany (and one Neon wake-up), not one per key."""
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

    def _value(self, key: str, packed: Any) -> Optional[Any]:
        """Decoded value for a live entry (caller holds the lock)."""
        self._store.move_to_end(key)
        if key in self._hot:
            self._hot.move_to_end(key)
            return self._hot[key]
        if isinstance(packed, _Unpacked):
            return packed.value
        try:
            value = decode_value(packed)
        except Exception as exc:
            logger.warning("Dropping undecodable cache entry %s: %s", key, exc)
            self._drop(key)
            return None
        self._remember(key, value)
        return value

    def _remember(self, key: str, value: Any) -> None:
        self._hot[key] = value
        self._hot.move_to_end(key)
        while len(self._hot) > self._hot_max:
            self._hot.popitem(last=False)

    def _drop(self, key: str) -> None:
        entry = self._store.pop(key, None)
        if entry is not None:
            self._bytes -= _size(entry[0])
        self._hot.pop(key, None)
        self._stored_at.pop(key, None)

    def _enforce_budget(self) -> None:
        """Drop least recently used entries until within the memory budget."""
        if self._budget is None or self._bytes <= self._budget:
            return
        dropped = 0
        while self._bytes > self._budget and len(self._store) > 1:
            key = next(iter(self._store))
            self._drop(key)
            dropped += 1
        logger.info(
            "Cache over its %.0f MB budget: dropped %d least recently used entries",
            self._budget / 1e6, dropped,
        )

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now > exp + self._grace]
        for k in expired:
            self._drop(k)

    def _mark_dirty(self) -> None:
        """Schedule a disk write (caller holds the lock)."""
        if not self._persist_path:
            return
        self._dirty = True
        if self._timer is None:
            self._timer = threading.Timer(self._persist_interval, self._timer_flush)
            self._timer.daemon = True
            self._timer.start()

    def _timer_flush(self) -> None:
        with self._lock:
            self._timer = None
        self.flush()

    def _load(self) -> None:
        """Load persisted entries from disk, dropping any already expired.

        Best-effort: a missing or corrupt cache file is treated as an empty
        cache rather than a fatal error. A file in the older format (live
        objects rather than compressed values) is converted.
        """
        try:
            with open(self._persist_path, "rb") as fh:  # type: ignore[arg-type]
                data = pickle.load(fh)
        except FileNotFoundError:
            return
        except Exception as exc:  # corrupt file, unpickling error, etc.
            logger.warning("Could not load weather cache from %s: %s", self._persist_path, exc)
            return
        if isinstance(data, dict) and data.get("format") == _FILE_FORMAT:
            entries = data["entries"]
        elif isinstance(data, dict):
            entries = {}
            for k, (v, exp) in data.items():
                try:
                    entries[k] = (encode_value(v), exp)
                except Exception:
                    continue
        else:
            logger.warning("Ignoring weather cache file %s of unknown format", self._persist_path)
            return
        now = time.time()
        for k, (packed, exp) in entries.items():
            if exp + self._grace > now:
                self._store[k] = (packed, exp)
                self._bytes += len(packed)
        self._enforce_budget()
        logger.info("Loaded %d cached weather entries from %s", len(self._store), self._persist_path)

    def _write(self, snapshot: dict[str, tuple[bytes, float]]) -> None:
        """Atomically write *snapshot* to disk (best-effort)."""
        tmp = f"{self._persist_path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "wb") as fh:
                pickle.dump({"format": _FILE_FORMAT, "entries": snapshot}, fh,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self._persist_path)
        except Exception as exc:
            logger.warning("Could not persist weather cache to %s: %s", self._persist_path, exc)
            try:
                os.remove(tmp)
            except OSError:
                pass


# Version of the on-disk layout: {"format": 2, "entries": {key: (blob, expires_at)}}.
_FILE_FORMAT = 2


class _Unpacked:
    """A value that could not be pickled: served from memory, never persisted."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


def _size(packed: Any) -> int:
    return len(packed) if isinstance(packed, bytes) else 0
