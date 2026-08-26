

def test_changed_mind_counts_once(tmp_path):
    """The store is append-only, so re-rating an evening leaves both records.
    Bulk readers must treat that as one observation, not two — otherwise a
    moment's hesitation double-weights that evening and inflates the label
    count that gates the correlation."""
    import asyncio

    from app.services.rating_store import RatingStore

    store = RatingStore(str(tmp_path / "r.jsonl"))

    async def go():
        for value in (2, 3, 4):
            await store.append({
                "target_date": "2026-08-23", "latitude": 32.08,
                "longitude": 34.78, "rating": value, "predicted_score": 50.0,
            })
        await store.append({
            "target_date": "2026-08-24", "latitude": 32.08,
            "longitude": 34.78, "rating": 5, "predicted_score": 60.0,
        })

    asyncio.run(go())

    assert len(list(store.iter_records())) == 4, "every append is still on disk"
    latest = store.latest_per_evening()
    assert len(latest) == 2, "one row per evening"
    by_date = {r["target_date"]: r["rating"] for r in latest}
    assert by_date["2026-08-23"] == 4, "the last rating wins"
    assert by_date["2026-08-24"] == 5


def test_same_evening_nearby_coords_counts_once():
    """Rating one evening from two spots a couple of km apart is ONE observation.

    This was a real defect, not a hypothetical: the store deduped on
    coordinates rounded to 2 dp (~1.1 km) while find() matched within 0.05 deg
    (~5 km), so the same evening rated from home and then from the beach was
    "already rated" for the UI but two independent labels for the accuracy
    check. In the real label set it kept a score from an engine several commits
    stale alongside the current one, for the same sky.
    """
    from app.services.rating_store import dedupe_latest

    records = [
        # Same evening, 1.7 km apart — the real case from data/ratings.jsonl.
        {"target_date": "2026-08-23", "latitude": 32.0853, "longitude": 34.7818,
         "rating": 4, "predicted_score": 17.3},
        {"target_date": "2026-08-23", "latitude": 32.0775, "longitude": 34.7663,
         "rating": 4, "predicted_score": 48.3},
    ]

    out = dedupe_latest(records)
    assert len(out) == 1, "same evening, same place — one observation"
    assert out[0]["predicted_score"] == 48.3, "the later rating wins"


def test_genuinely_distant_places_stay_separate():
    """Same date, different cities is two observations — the dedupe must not
    collapse everything sharing a date."""
    from app.services.rating_store import dedupe_latest

    records = [
        {"target_date": "2026-08-23", "latitude": 32.08, "longitude": 34.78, "rating": 4},
        {"target_date": "2026-08-23", "latitude": 51.51, "longitude": -0.13, "rating": 2},
    ]

    assert len(dedupe_latest(records)) == 2


def test_dedupe_survives_grid_boundary():
    """Two points ~1 km apart that straddle a rounding boundary.

    Grid-rounding put these in different buckets precisely when they were
    closest to each other, which is the bug the rounding looked like it was
    avoiding. Clustering by distance does not care where the boundary falls.
    """
    from app.services.rating_store import dedupe_latest

    records = [
        {"target_date": "2026-08-23", "latitude": 32.0049, "longitude": 34.78, "rating": 3},
        {"target_date": "2026-08-23", "latitude": 32.0051, "longitude": 34.78, "rating": 5},
    ]

    out = dedupe_latest(records)
    assert len(out) == 1, "20 m apart is the same place, whatever rounding says"
    assert out[0]["rating"] == 5
