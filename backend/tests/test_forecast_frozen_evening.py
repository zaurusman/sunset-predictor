"""An evening that is over reads the same in /forecast as in /predict.

/predict freezes an evening once its viewing window ends (the last reading
from before sunset stays). /forecast used to rescore that evening from the
newest model run, so after sunset Tonight said 71 and the 7-day card 80.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

import app.services.prediction_service as ps
from app.core.config import settings
from app.schemas.forecast import ForecastRequest
from app.schemas.prediction import PhysicsBreakdown, PredictResponse, WeatherSummary
from app.services.astronomy_service import AstronomyService
from app.services.prediction_service import PredictionService

UTC = timezone.utc
TEL_AVIV = (32.08, 34.78)
D0 = date(2026, 10, 2)
D1 = date(2026, 10, 3)


class StubWeather:
    async def get_forecast_range_windows(self, lat, lon, days):
        return [(D0, ["new run"]), (D1, ["new run"])]

    async def get_corridor_samples_map(self, lat, lon, dates):
        return {}

    async def get_ensemble_cloud_spread_map(self, lat, lon, targets):
        return {}


def _prediction(score: float, sunset: datetime) -> PredictResponse:
    return PredictResponse(
        beauty_score_0_100=score, category="Great", confidence_0_100=60.0,
        reasons=["frozen"], sunset_time=sunset,
        best_viewing_window_start=sunset, best_viewing_window_end=sunset,
        best_window_point="sunset", window_scores={"sunset": score},
        go_outside_recommendation=False, raw_physics_score=score,
        climatology_percentile=None, climatology_is_local=False,
        algorithm_version="test", ml_model_used=False, ml_adjustment=None,
        physics_component_breakdown=PhysicsBreakdown(
            cloud_quality_score=58.1, atmosphere_score=83.5, moisture_score=100,
            horizon_score=96.3, weighted_physics_score=score, component_weights={},
        ),
        weather_summary=WeatherSummary(
            cloud_low_pct=20, cloud_mid_pct=27, cloud_high_pct=89, cloud_total_pct=78,
            visibility_km=None, precipitation_mm=0, aerosol_optical_depth=None,
            aerosol_is_estimated=True, temperature_c=24, humidity_pct=60, wind_speed_kmh=8,
        ), location={"latitude": 0, "longitude": 0},
        requested_at=sunset,
    )


@pytest.mark.asyncio
async def test_an_evening_that_is_over_is_the_one_predict_froze(monkeypatch):
    astro = AstronomyService()
    sunset0 = astro.get_sunset_time(*TEL_AVIV, D0)
    svc = PredictionService(
        weather_service=StubWeather(), astro_service=astro, scoring_engine=None,
        explanation_engine=None, ml_model=None, settings=settings,
    )
    # After D0's window, before D1's sunset.
    monkeypatch.setattr(ps, "utcnow", lambda: sunset0 + timedelta(hours=2))

    asked: list[date] = []

    async def predict(request, **_):
        asked.append(request.target_date)
        return _prediction(71.1, astro.get_sunset_time(*TEL_AVIV, request.target_date))

    rescored: list[date] = []

    async def score_day(lat, lon, d, window_snaps, *a, **k):
        rescored.append(d)
        return (await svc._frozen_day(lat, lon, d, 2.0)).model_copy(
            update={"beauty_score_0_100": 80.3}
        )

    monkeypatch.setattr(svc, "predict", predict)
    monkeypatch.setattr(svc, "_score_day", score_day)

    out = await svc.forecast(ForecastRequest(latitude=TEL_AVIV[0], longitude=TEL_AVIV[1], days=2))

    by_date = {d.date: d for d in out.days}
    assert by_date[D0].beauty_score_0_100 == 71.1       # what Tonight shows
    assert by_date[D0].physics_component_breakdown.cloud_quality_score == 58.1
    assert by_date[D1].beauty_score_0_100 == 80.3       # still scored from the new run
    assert rescored == [D1]
