"""Tests for the rating capture path (POST /rate, GET /ratings/stats).

These labels are the input to every future scoring change, so the properties
that matter are: nothing is silently lost, raw inputs are stored alongside the
label, and rubbish (future dates, out-of-range stars) never enters the file.
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.rating_store import RatingStore
from app.utils.math_utils import spearman


# ---------------------------------------------------------------------------
# RatingStore — durability and tolerance to damage
# ---------------------------------------------------------------------------


def test_store_roundtrips_records(tmp_path):
    store = RatingStore(path=str(tmp_path / "r.jsonl"))
    asyncio.run(store.append({"rating": 4, "target_date": "2026-08-20"}))
    asyncio.run(store.append({"rating": 2, "target_date": "2026-08-21"}))

    records = list(store.iter_records())
    assert [r["rating"] for r in records] == [4, 2]
    assert store.count() == 2


def test_store_append_returns_running_total(tmp_path):
    store = RatingStore(path=str(tmp_path / "r.jsonl"))
    assert asyncio.run(store.append({"rating": 1})) == 1
    assert asyncio.run(store.append({"rating": 5})) == 2


def test_store_creates_missing_parent_directory(tmp_path):
    store = RatingStore(path=str(tmp_path / "nested" / "deeper" / "r.jsonl"))
    asyncio.run(store.append({"rating": 3}))
    assert store.count() == 1


def test_store_survives_a_corrupt_line(tmp_path):
    """A truncated write must not make the rest of the dataset unreadable."""
    path = tmp_path / "r.jsonl"
    store = RatingStore(path=str(path))
    asyncio.run(store.append({"rating": 5}))
    with open(path, "a") as f:
        f.write('{"rating": 3, "truncated"\n')   # power-loss mid-write
    asyncio.run(store.append({"rating": 1}))

    ratings = [r["rating"] for r in store.iter_records()]
    assert ratings == [5, 1], "good records must survive a corrupt neighbour"


def test_store_count_is_zero_when_file_absent(tmp_path):
    assert RatingStore(path=str(tmp_path / "nope.jsonl")).count() == 0


def test_store_find_matches_within_gps_jitter(tmp_path):
    store = RatingStore(path=str(tmp_path / "r.jsonl"))
    asyncio.run(store.append(
        {"rating": 4, "target_date": "2026-08-20", "latitude": 32.08, "longitude": 34.78}
    ))
    assert store.find(32.081, 34.779, "2026-08-20") is not None, "jittery GPS should still match"
    assert store.find(35.00, 34.78, "2026-08-20") is None, "a different city should not match"
    assert store.find(32.08, 34.78, "2026-08-19") is None, "a different date should not match"


# ---------------------------------------------------------------------------
# Spearman helper — the number the whole exercise reports
# ---------------------------------------------------------------------------


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_linear():
    """Monotone but wildly non-linear should still be a perfect rank match."""
    assert spearman([1, 2, 3, 4], [1, 4, 900, 10_000]) == pytest.approx(1.0)


def test_spearman_handles_ties():
    rho = spearman([1, 1, 2, 3], [5, 5, 7, 9])
    assert rho is not None and rho == pytest.approx(1.0)


def test_spearman_undefined_cases():
    assert spearman([1, 2], [3, 4]) is None, "too few pairs"
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None, "constant series has no rank variance"


def test_spearman_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def rating_client(tmp_path):
    """TestClient with the rating store redirected to a temp file."""
    with TestClient(app) as c:
        original = app.state.rating_store
        app.state.rating_store = RatingStore(path=str(tmp_path / "ratings.jsonl"))
        yield c
        app.state.rating_store = original


def test_rate_rejects_future_dates(rating_client):
    """You cannot have seen a sunset that hasn't happened — those would poison training."""
    future = (date.today() + timedelta(days=3)).isoformat()
    resp = rating_client.post("/rate", json={
        "latitude": 32.08, "longitude": 34.78, "rating": 5, "target_date": future,
    })
    assert resp.status_code == 400
    assert "hasn't happened" in resp.json()["detail"]


@pytest.mark.parametrize("bad", [0, 6, -1])
def test_rate_rejects_out_of_range_stars(rating_client, bad):
    resp = rating_client.post("/rate", json={
        "latitude": 32.08, "longitude": 34.78, "rating": bad,
    })
    assert resp.status_code == 422


def test_rate_rejects_impossible_coordinates(rating_client):
    resp = rating_client.post("/rate", json={
        "latitude": 200.0, "longitude": 34.78, "rating": 3,
    })
    assert resp.status_code == 422


def test_stats_empty_dataset_is_not_an_error(rating_client):
    resp = rating_client.get("/ratings/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_ratings"] == 0
    assert body["spearman_vs_model"] is None
    assert "No ratings yet" in body["note"]


def test_stats_withholds_correlation_until_enough_pairs(rating_client, tmp_path):
    """A rho computed from four ratings is noise; better to show nothing."""
    store = app.state.rating_store
    for i in range(4):
        asyncio.run(store.append({
            "rating": (i % 5) + 1, "predicted_score": 50.0 + i,
            "target_date": f"2026-08-0{i+1}", "latitude": 32.08, "longitude": 34.78,
        }))

    body = rating_client.get("/ratings/stats").json()
    assert body["total_ratings"] == 4
    assert body["spearman_vs_model"] is None
    assert "Need 15" in body["note"]


def test_stats_reports_correlation_and_flags_one_sided_data(rating_client):
    """With only good nights rated, rho is reported but called out as one-sided."""
    store = app.state.rating_store
    for i in range(20):
        asyncio.run(store.append({
            "rating": 4 + (i % 2),          # only 4s and 5s — no bad evenings
            "predicted_score": 60.0 + i,
            "target_date": f"2026-07-{i+1:02d}", "latitude": 32.08, "longitude": 34.78,
        }))

    body = rating_client.get("/ratings/stats").json()
    assert body["total_ratings"] == 20
    assert body["spearman_vs_model"] is not None
    assert "one-sided" in body["note"], "must warn that there is no negative class"


def test_stats_counts_distinct_locations_and_dates(rating_client):
    store = app.state.rating_store
    for lat, lon, d in [
        (32.08, 34.78, "2026-08-01"),
        (32.08, 34.78, "2026-08-02"),
        (51.51, -0.13, "2026-08-01"),
    ]:
        asyncio.run(store.append({
            "rating": 3, "predicted_score": 55.0,
            "target_date": d, "latitude": lat, "longitude": lon,
        }))

    body = rating_client.get("/ratings/stats").json()
    assert body["distinct_locations"] == 2
    assert body["distinct_dates"] == 2
    assert body["rating_histogram"] == {"3": 3}
