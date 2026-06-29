from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt
from statistics import mean, median


def haversine_meters(lat1, lon1, lat2, lon2):
    radius = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return radius * c


def geolocation_summary(predicted: list[dict], gold: list[dict]) -> dict:
    errors = []
    for pred in predicted:
        for target in gold:
            if _has_point(pred) and _has_point(target):
                errors.append(haversine_meters(pred["latitude"], pred["longitude"], target["latitude"], target["longitude"]))
    if not errors:
        return {"mean_error_meters": None, "median_error_meters": None, "within_100m": None, "within_500m": None, "within_1km": None}
    best = min(errors)
    return {
        "mean_error_meters": mean(errors),
        "median_error_meters": median(errors),
        "best_error_meters": best,
        "within_100m": best <= 100,
        "within_500m": best <= 500,
        "within_1km": best <= 1000,
    }


def _has_point(row: dict) -> bool:
    return row.get("latitude") is not None and row.get("longitude") is not None
