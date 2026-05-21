"""
Live sanity checks — call the real Open-Meteo archive and verify scores
for specific dates where the sky was genuinely exceptional.

These tests are skipped by default. Run with:
    pytest -m live --live

IMPORTANT: archive dates use the 15 km visibility default (real visibility
is never returned by the archive API), so the atmosphere component is
capped at ~59 rather than the 90+ you'd get on a recent day with real data.
The maximum achievable score for an archive date is ~89.

The tests therefore assert ≥ 80 (Epic tier) rather than a specific number.
What we're checking is: does the model recognise these exceptional cloud
conditions as Epic? It should — both dates had 97–100 % high cirrus with
0 % low cloud, which is the best possible cloud structure.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from app.main import app

# Tel Aviv
LAT = 32.0853
LON = 34.7818

# Shared client across all live tests in this module
@pytest.fixture(scope="module")
def live_client():
    with TestClient(app) as c:
        yield c


def _predict(client, date_str: str) -> dict:
    resp = client.post("/predict", json={
        "latitude": LAT,
        "longitude": LON,
        "target_date": date_str,
        "horizon_obstruction_deg": 2.0,
    })
    assert resp.status_code == 200, f"Predict failed for {date_str}: {resp.text}"
    return resp.json()


# ---------------------------------------------------------------------------
# Verified exceptional sunsets — Tel Aviv
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_tel_aviv_jan_4_2026_is_epic(live_client):
    """
    4 January 2026 — Tel Aviv.
    Verified exceptional sunset: 97 % high cirrus, 0 % low cloud.
    Should be Epic (≥ 80).

    Archive limitation: visibility always null → defaults to 15 km.
    Atmosphere is therefore ~59 (not the 90+ of a real-data day).
    Max achievable archive score ≈ 89 — so we check ≥ 80, not 95.
    """
    data = _predict(live_client, "2026-01-04")
    score = data["beauty_score_0_100"]
    category = data["category"]
    clouds = data["weather_summary"]

    assert score >= 80.0, (
        f"Jan 4 2026 Tel Aviv should be Epic (≥ 80), got {score:.1f} ({category})\n"
        f"  clouds: low={clouds['cloud_low_pct']}%, mid={clouds['cloud_mid_pct']}%, "
        f"high={clouds['cloud_high_pct']}%"
    )
    assert category == "Epic", (
        f"Jan 4 2026 Tel Aviv should be Epic, got {category} (score={score:.1f})"
    )


@pytest.mark.live
def test_tel_aviv_nov_17_2025_is_epic(live_client):
    """
    17 November 2025 — Tel Aviv.
    Verified exceptional sunset: 100 % high cirrus, 0 % low cloud.
    Should be Epic (≥ 80).
    """
    data = _predict(live_client, "2025-11-17")
    score = data["beauty_score_0_100"]
    category = data["category"]
    clouds = data["weather_summary"]

    assert score >= 80.0, (
        f"Nov 17 2025 Tel Aviv should be Epic (≥ 80), got {score:.1f} ({category})\n"
        f"  clouds: low={clouds['cloud_low_pct']}%, mid={clouds['cloud_mid_pct']}%, "
        f"high={clouds['cloud_high_pct']}%"
    )
    assert category == "Epic", (
        f"Nov 17 2025 Tel Aviv should be Epic, got {category} (score={score:.1f})"
    )


# ---------------------------------------------------------------------------
# Counter-sanity: a rainy Tel Aviv day should NOT be Epic
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_tel_aviv_dull_day_not_epic(live_client):
    """
    4 February 2025 — Tel Aviv.
    A dull cloudy day: 12 % low cloud, 0 % high cloud, no real colour potential.
    Should NOT be Epic — verifies the model isn't indiscriminate.
    Expected: Good or lower (score < 65).
    """
    data = _predict(live_client, "2025-02-04")
    score = data["beauty_score_0_100"]
    category = data["category"]
    clouds = data["weather_summary"]

    assert category != "Epic", (
        f"A dull Tel Aviv day should not be Epic, got {category} (score={score:.1f})\n"
        f"  clouds: low={clouds['cloud_low_pct']}%, mid={clouds['cloud_mid_pct']}%, "
        f"high={clouds['cloud_high_pct']}%"
    )
    assert score < 65.0, (
        f"A dull day should score < 65, got {score:.1f}"
    )
