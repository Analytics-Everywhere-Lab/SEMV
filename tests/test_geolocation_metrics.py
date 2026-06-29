from __future__ import annotations

from src.evaluation.geolocation_metrics import geolocation_summary, haversine_meters


def test_haversine_and_thresholds():
    assert haversine_meters(0, 0, 0, 0) == 0
    summary = geolocation_summary([{"latitude": 0, "longitude": 0}], [{"latitude": 0, "longitude": 0}])
    assert summary["within_100m"] is True
