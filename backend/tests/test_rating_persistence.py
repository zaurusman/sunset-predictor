"""Ratings survive, and come back out ready to train on.

Render's disk is wiped on every deploy, so production ratings live in
Postgres. These tests hold both stores to one contract, check that POST /rate
never loses a label to a weather outage or acknowledges one it did not save,
and that a stored record turns into a training row the current engine can
replay without any network.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.weather import WeatherSnapshot
from app.services.rating_dataset import rescore, training_row, training_rows
from app.services.rating_store import (
    RATING_SCHEMA_VERSION,
    PostgresRatingStore,
    RatingStore,
    load_records,
    record_hash,
)
from app.services.scoring_engine import ScoringEngine
from app.services.weather_service import WeatherUnavailableError

TLV = (32.08, 34.78)


def _snap(label: str, low: float = 20.0, high: float = 40.0) -> WeatherSnapshot:
    return WeatherSnapshot(
        cloud_low=low, cloud_mid=15.0, cloud_high=high, cloud_total=55.0,
        visibility_m=20_000.0, relative_humidity=60.0, dewpoint_c=12.0,
        temperature_c=22.0, precipitation_mm=0.0, wind_speed_kmh=12.0,
        pressure_hpa=1012.0, aerosol_optical_depth=0.15, sun_elevation_deg=0.5,
        timestamp_label=label,
    )


CORRIDOR = [(100.0, 5.0, 10.0), (250.0, 0.0, 5.0), (400.0, 10.0, 0.0)]


def _full_record(day: str = "2026-08-20", rating: float = 70.0, **extra) -> dict:
    rec = {
        "schema_version": RATING_SCHEMA_VERSION,
        "recorded_at": f"{day}T17:30:00+00:00",
        "target_date": day,
        "latitude": TLV[0], "longitude": TLV[1],
        "rating_0_100": rating, "rating_is_precise": False, "rating": 4,
        "observed_moment": None, "horizon_deg": 2.0, "context_error": None,
        "predicted_score": 55.0, "algorithm_version": "1.0.0",
        "window_scores": {"sunset": 55.0},
        "window_snapshots": [_snap(m).model_dump(mode="json") for m in ("-15m", "sunset", "+15m", "+30m")],
        "corridor_samples": [list(c) for c in CORRIDOR],
    }
    rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# Store contract — the same behaviour from the file and the database
# ---------------------------------------------------------------------------


async def _exercise(store) -> None:
    assert await store.total() == 0
    assert await store.records() == []

    first = _full_record("2026-08-20", 30.0)
    assert await store.append(first) == 1
    assert await store.append(_full_record("2026-08-21", 80.0)) == 2
    # A legacy (schema 1) record with almost nothing in it must still store.
    assert await store.append({"rating": 3, "target_date": "2026-08-22"}) == 3

    records = await store.records()
    assert [r["target_date"] for r in records] == ["2026-08-20", "2026-08-21", "2026-08-22"], \
        "oldest first, so dedupe_latest's last-write-wins holds"
    assert records[0] == json.loads(json.dumps(first)), "the record comes back whole"
    assert records[0]["window_snapshots"] and records[0]["corridor_samples"]

    light = await store.records(with_raw=False)
    assert [r["target_date"] for r in light] == ["2026-08-20", "2026-08-21", "2026-08-22"]
    assert light[0]["rating_0_100"] == 30.0


def test_jsonl_store_contract(tmp_path):
    asyncio.run(_exercise(RatingStore(str(tmp_path / "r.jsonl"))))


requires_pg = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to run"
)


async def _fresh_pg() -> PostgresRatingStore:
    store = await PostgresRatingStore.connect(os.environ["TEST_DATABASE_URL"])
    await store.ensure_schema()  # idempotent
    await store._pool.execute("TRUNCATE sunset_ratings")
    return store


@requires_pg
def test_postgres_store_contract():
    async def go():
        store = await _fresh_pg()
        try:
            await _exercise(store)
            light = await store.records(with_raw=False)
            assert "window_snapshots" not in light[0] and "corridor_samples" not in light[0], \
                "stats readers must not pull the raw inputs over the wire"
        finally:
            await store.close()
    asyncio.run(go())


@requires_pg
def test_postgres_typed_columns_and_idempotent_import():
    async def go():
        store = await _fresh_pg()
        try:
            recs = [_full_record("2026-08-20", 30.0), _full_record("2026-08-21", 80.0)]
            assert await store.append_many(recs) == 2
            assert await store.append_many(recs) == 0, "re-importing a file adds nothing"
            assert await store.total() == 2

            row = await store._pool.fetchrow(
                "SELECT target_date, rating_0_100, schema_version, has_raw_inputs, record_hash"
                " FROM sunset_ratings ORDER BY id LIMIT 1"
            )
            assert row["target_date"] == date(2026, 8, 20)
            assert row["rating_0_100"] == 30.0
            assert row["schema_version"] == RATING_SCHEMA_VERSION
            assert row["has_raw_inputs"] is True
            assert row["record_hash"] == record_hash(recs[0])
        finally:
            await store.close()
    asyncio.run(go())


@requires_pg
def test_postgres_store_survives_nan():
    """Postgres JSONB rejects NaN; one bad float must not cost the label."""
    async def go():
        store = await _fresh_pg()
        try:
            await store.append(_full_record(predicted_score=float("nan")))
            (rec,) = await store.records()
            assert rec["predicted_score"] is None
        finally:
            await store.close()
    asyncio.run(go())


@requires_pg
def test_load_records_reads_a_database_url():
    async def go():
        store = await _fresh_pg()
        try:
            await store.append(_full_record())
        finally:
            await store.close()
        return await load_records(os.environ["TEST_DATABASE_URL"])
    (rec,) = asyncio.run(go())
    assert rec["target_date"] == "2026-08-20"


def test_load_records_reads_a_jsonl_path(tmp_path):
    path = str(tmp_path / "r.jsonl")
    asyncio.run(RatingStore(path).append(_full_record()))
    assert len(asyncio.run(load_records(path))) == 1


# ---------------------------------------------------------------------------
# POST /rate — never lose a label, never fake a save
# ---------------------------------------------------------------------------


@pytest.fixture
def rate_client(tmp_path):
    with TestClient(app) as c:
        original_store = app.state.rating_store
        app.state.rating_store = RatingStore(path=str(tmp_path / "ratings.jsonl"))
        svc = app.state.prediction_service
        yield c, svc
        app.state.rating_store = original_store


def _yesterday() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


def test_rate_stores_raw_inputs_including_corridor(rate_client, monkeypatch):
    client, svc = rate_client

    class FakePrediction:
        beauty_score_0_100 = 61.0
        category = "Good"
        confidence_0_100 = 70.0
        raw_physics_score = 61.0
        climatology_percentile = 0.7
        algorithm_version = "1.0.0"
        ml_model_used = False
        window_scores = {"sunset": 61.0}
        best_window_point = "sunset"

        class physics_component_breakdown:
            @staticmethod
            def model_dump():
                return {"cloud_quality": 70.0}

    async def fake_capture(**_):
        return FakePrediction(), [_snap("sunset")], CORRIDOR

    monkeypatch.setattr(svc, "capture_rating_context", fake_capture)
    resp = client.post("/rate", json={
        "latitude": TLV[0], "longitude": TLV[1], "rating": 4, "target_date": _yesterday(),
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["predicted_score"] == 61.0

    (rec,) = list(app.state.rating_store.iter_records())
    assert rec["schema_version"] == RATING_SCHEMA_VERSION
    assert rec["corridor_samples"] == [list(c) for c in CORRIDOR]
    assert rec["window_snapshots"][0]["timestamp_label"] == "sunset"
    assert rec["horizon_deg"] == 2.0
    assert rec["context_error"] is None
    assert rec["raw_physics_score"] == 61.0


def test_rate_keeps_the_label_when_weather_is_unavailable(rate_client, monkeypatch):
    """The label is the scarce part — weather can be backfilled from the
    archive later; the person who saw the sky cannot be asked again."""
    client, svc = rate_client

    async def refused(**_):
        raise WeatherUnavailableError("Open-Meteo daily limit")

    monkeypatch.setattr(svc, "capture_rating_context", refused)
    resp = client.post("/rate", json={
        "latitude": TLV[0], "longitude": TLV[1], "rating": 2, "target_date": _yesterday(),
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True and body["predicted_score"] is None

    (rec,) = list(app.state.rating_store.iter_records())
    assert rec["rating_0_100"] == 30.0
    assert "unavailable" in rec["context_error"]
    assert rec["window_snapshots"] == [] and rec["predicted_score"] is None


def test_rate_reports_failure_when_the_store_does(rate_client, monkeypatch):
    client, svc = rate_client

    async def refused(**_):
        raise WeatherUnavailableError("down")

    class BrokenStore:
        async def append(self, record):
            raise ConnectionError("database unreachable")

    monkeypatch.setattr(svc, "capture_rating_context", refused)
    app.state.rating_store = BrokenStore()
    resp = client.post("/rate", json={
        "latitude": TLV[0], "longitude": TLV[1], "rating": 3, "target_date": _yesterday(),
    })
    assert resp.status_code == 503
    assert "Could not save" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Training table
# ---------------------------------------------------------------------------


def test_rescore_uses_the_stored_corridor_offline():
    engine = ScoringEngine()
    rec = _full_record()
    result = rescore(engine, rec)
    assert result is not None
    expected = engine.score(_snap("sunset"), 2.0, corridor_samples=CORRIDOR).physics_score
    assert result["by_moment"]["sunset"] == pytest.approx(expected)
    assert result["comparable"] == result["final"], "whole-evening label -> window score"


def test_rescore_compares_against_the_observed_moment():
    result = rescore(ScoringEngine(), _full_record(observed_moment="+15m"))
    assert result["comparable"] == result["by_moment"]["+15m"]


def test_rescore_refuses_to_guess_a_missing_corridor():
    rec = _full_record()
    del rec["corridor_samples"]  # a schema-1 record
    assert rescore(ScoringEngine(), rec) is None
    assert rescore(ScoringEngine(), rec, corridor=CORRIDOR) is not None


def test_training_row_has_label_features_and_engine_baseline():
    row = training_row(_full_record(rating=72.0), ScoringEngine())
    assert row["label_rating_0_100"] == 72.0
    assert row["label_band"] == 4
    assert row["sunset__cloud_low"] == 20.0 and row["+30m__cloud_high"] == 40.0
    assert row["corridor_n"] == 3.0
    assert row["corridor_low_max"] == 10.0
    assert json.loads(row["corridor_json"]) == [list(c) for c in CORRIDOR]
    assert isinstance(row["engine_score"], float)
    assert row["month"] == 8.0
    assert math.isclose(row["month_sin"], math.sin(8 * 2 * math.pi / 12))
    assert row["meta_has_raw_inputs"] is True


def test_training_row_for_a_label_without_weather():
    rec = _full_record(window_snapshots=[], corridor_samples=None,
                       context_error="weather unavailable", predicted_score=None)
    row = training_row(rec, ScoringEngine())
    assert row["label_rating_0_100"] == 70.0
    assert row["engine_score"] is None
    assert row["meta_has_raw_inputs"] is False
    assert row["meta_context_error"] == "weather unavailable"


def test_training_rows_dedupe_a_changed_mind():
    rows = training_rows([
        _full_record("2026-08-20", 30.0),
        _full_record("2026-08-20", 60.0),   # same evening, re-rated
        _full_record("2026-08-21", 90.0),
        {"target_date": "2026-08-22"},      # no label at all
    ])
    assert [(r["meta_target_date"], r["label_rating_0_100"]) for r in rows] == [
        ("2026-08-20", 60.0), ("2026-08-21", 90.0),
    ]
