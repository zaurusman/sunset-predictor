"""Alert runs must never trigger a climatology build (extra archive fetches)."""
from __future__ import annotations

from app.core.config import settings
from app.services.prediction_service import PredictionService


class FakeClimatology:
    def __init__(self) -> None:
        self.warmed: list[tuple[float, float]] = []

    def percentile_of(self, lat, lon, raw_score, on_date=None):
        return 0.5, False  # not local → would normally warm

    def warm_in_background(self, lat, lon):
        self.warmed.append((lat, lon))


def _service(clim: FakeClimatology) -> PredictionService:
    return PredictionService(
        weather_service=None, astro_service=None, scoring_engine=None,
        explanation_engine=None, ml_model=None, settings=settings, climatology=clim,
    )


def test_calibrate_warms_by_default():
    clim = FakeClimatology()
    _service(clim)._calibrate(60.0, 32.1, 34.8)
    assert clim.warmed == [(32.1, 34.8)]


def test_calibrate_skips_warm_when_disabled():
    clim = FakeClimatology()
    _service(clim)._calibrate(60.0, 32.1, 34.8, warm=False)
    assert clim.warmed == []
