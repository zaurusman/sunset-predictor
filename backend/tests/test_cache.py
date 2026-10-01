"""Tests for TTLCache, focusing on disk persistence across process restarts.

Persistence is what lets the weather cache survive `uvicorn --reload`, so a
code change no longer wipes the cache and forces a full Open-Meteo re-fetch.
"""
from __future__ import annotations

import time

from app.utils.cache import TTLCache


def test_persists_across_instances(tmp_path):
    """A value written by one instance is readable by a fresh instance."""
    path = str(tmp_path / "cache.pkl")

    a = TTLCache(ttl_seconds=900, persist_path=path)
    a.set("key", {"score": 42})
    a.flush()

    b = TTLCache(ttl_seconds=900, persist_path=path)
    assert b.get("key") == {"score": 42}


def test_expired_entries_dropped_on_load(tmp_path):
    """Entries past their TTL are not resurrected when a new instance loads."""
    path = str(tmp_path / "cache.pkl")

    a = TTLCache(ttl_seconds=900, persist_path=path)
    a.set("fresh", 1)
    a.set("stale", 2, ttl_override=-1)  # already expired
    a.flush()

    b = TTLCache(ttl_seconds=900, persist_path=path)
    assert b.get("fresh") == 1
    assert b.get("stale") is None


def test_no_persistence_when_path_disabled():
    """With persistence off, a new instance starts empty."""
    a = TTLCache(ttl_seconds=900, persist_path=None)
    a.set("key", "value")

    b = TTLCache(ttl_seconds=900, persist_path=None)
    assert b.get("key") is None


def test_corrupt_cache_file_is_ignored(tmp_path):
    """A corrupt cache file is treated as empty, not a fatal error."""
    path = tmp_path / "cache.pkl"
    path.write_bytes(b"not a valid pickle")

    cache = TTLCache(ttl_seconds=900, persist_path=str(path))
    assert cache.get("anything") is None
    # And it can still be used normally afterwards.
    cache.set("k", 1)
    assert cache.get("k") == 1


def test_clear_empties_persisted_store(tmp_path):
    """clear() removes entries from disk too."""
    path = str(tmp_path / "cache.pkl")

    a = TTLCache(ttl_seconds=900, persist_path=path)
    a.set("key", 1)
    a.flush()
    a.clear()
    a.flush()

    b = TTLCache(ttl_seconds=900, persist_path=path)
    assert b.get("key") is None


def test_get_stale_returns_expired_entry_within_grace():
    """An expired entry is hidden from get() but still offered as a fallback."""
    c = TTLCache(ttl_seconds=900, stale_grace_seconds=3600)
    c.set("k", "v", ttl_override=-1)  # already expired

    assert c.get("k") is None          # never served as fresh
    assert c.get_stale("k") == "v"     # but still there for a fallback


def test_get_stale_drops_entry_past_grace():
    c = TTLCache(ttl_seconds=900, stale_grace_seconds=10)
    c.set("k", "v", ttl_override=-60)  # expired a minute ago, grace is 10s

    assert c.get_stale("k") is None
    assert c.size() == 0


def test_no_grace_means_no_stale_fallback():
    """Default grace of 0 keeps the old behaviour: expired means gone."""
    c = TTLCache(ttl_seconds=900)
    c.set("k", "v", ttl_override=-1)

    assert c.get_stale("k") is None


def test_stale_entries_survive_reload_within_grace(tmp_path):
    path = str(tmp_path / "cache.pkl")
    a = TTLCache(ttl_seconds=900, persist_path=path, stale_grace_seconds=3600)
    a.set("k", "v", ttl_override=-1)
    a.flush()

    b = TTLCache(ttl_seconds=900, persist_path=path, stale_grace_seconds=3600)
    assert b.get("k") is None
    assert b.get_stale("k") == "v"


# ---------------------------------------------------------------------------
# Memory: compressed storage + LRU budget (scripts/capacity found the old
# live-object cache ran Render's 512 MB out of memory at ~60 locations)
# ---------------------------------------------------------------------------

def _weather_like(seed: int) -> dict:
    """Shaped like a cached Open-Meteo month: long float lists."""
    return {"hourly": {"cloud_cover": [float((i * seed) % 100) for i in range(3000)]}}


def test_values_are_held_compressed():
    import pickle

    c = TTLCache(ttl_seconds=900, hot_entries=0)
    value = _weather_like(7)
    c.set("month", value)
    assert c.get("month") == value
    assert c.memory_bytes() < len(pickle.dumps(value)) / 3


def test_budget_drops_least_recently_used():
    c = TTLCache(ttl_seconds=900, hot_entries=0)
    c.set("probe", _weather_like(1))
    one = c.memory_bytes()
    c = TTLCache(ttl_seconds=900, memory_budget_bytes=int(one * 3.5), hot_entries=0)
    for k in ("a", "b", "c"):
        c.set(k, _weather_like(1))
    assert c.get("a") is not None      # touch a: b is now the oldest
    c.set("d", _weather_like(1))

    assert c.get("b") is None
    assert all(c.get(k) is not None for k in ("a", "c", "d"))
    assert c.memory_bytes() <= one * 3.5


def test_overwriting_a_key_does_not_leak_budget():
    c = TTLCache(ttl_seconds=900)
    for _ in range(20):
        c.set("same", _weather_like(3))
    c2 = TTLCache(ttl_seconds=900)
    c2.set("same", _weather_like(3))
    assert c.memory_bytes() == c2.memory_bytes()
    c.delete("same")
    assert c.memory_bytes() == 0


def test_unpicklable_value_is_served_from_memory():
    c = TTLCache(ttl_seconds=900)
    fn = lambda: 1  # noqa: E731
    c.set("fn", fn)
    assert c.get("fn") is fn


# ---------------------------------------------------------------------------
# Persistence is debounced: set() must not rewrite the whole store
# ---------------------------------------------------------------------------

def test_set_does_not_write_to_disk_immediately(tmp_path):
    path = tmp_path / "cache.pkl"
    c = TTLCache(ttl_seconds=900, persist_path=str(path), persist_interval=60)
    for i in range(50):
        c.set(f"k{i}", i)
    assert not path.exists()
    c.close()
    assert TTLCache(ttl_seconds=900, persist_path=str(path)).get("k49") == 49


def test_background_writer_flushes_after_interval(tmp_path):
    path = tmp_path / "cache.pkl"
    c = TTLCache(ttl_seconds=900, persist_path=str(path), persist_interval=0.05)
    c.set("k", "v")
    deadline = time.time() + 2
    while not path.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert TTLCache(ttl_seconds=900, persist_path=str(path)).get("k") == "v"


def test_legacy_cache_file_is_converted(tmp_path):
    import pickle

    path = tmp_path / "cache.pkl"
    path.write_bytes(pickle.dumps({"old": ({"score": 5}, time.time() + 900)}))
    assert TTLCache(ttl_seconds=900, persist_path=str(path)).get("old") == {"score": 5}
