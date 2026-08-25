

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
