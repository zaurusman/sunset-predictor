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

    b = TTLCache(ttl_seconds=900, persist_path=path)
    assert b.get("key") == {"score": 42}


def test_expired_entries_dropped_on_load(tmp_path):
    """Entries past their TTL are not resurrected when a new instance loads."""
    path = str(tmp_path / "cache.pkl")

    a = TTLCache(ttl_seconds=900, persist_path=path)
    a.set("fresh", 1)
    a.set("stale", 2, ttl_override=-1)  # already expired

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
    a.clear()

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

    b = TTLCache(ttl_seconds=900, persist_path=path, stale_grace_seconds=3600)
    assert b.get("k") is None
    assert b.get_stale("k") == "v"
