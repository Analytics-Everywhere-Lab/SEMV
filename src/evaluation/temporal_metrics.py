from __future__ import annotations

from datetime import datetime


def parse_time(value: str | None):
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromisoformat(text[:10])
        except ValueError:
            return None


def temporal_interval_iou(pred: dict, gold: dict) -> float | None:
    p0 = parse_time(pred.get("earliest_possible"))
    p1 = parse_time(pred.get("latest_possible"))
    g0 = parse_time(gold.get("earliest_possible"))
    g1 = parse_time(gold.get("latest_possible"))
    if not all([p0, p1, g0, g1]):
        return None
    latest_start = max(p0, g0)
    earliest_end = min(p1, g1)
    overlap = max(0.0, (earliest_end - latest_start).total_seconds())
    union = max((max(p1, g1) - min(p0, g0)).total_seconds(), 0.0)
    return overlap / union if union else 0.0


def publication_event_confusion(pred: dict, gold: dict) -> bool | None:
    if not gold:
        return None
    return bool(
        gold.get("exact_recording_time_known") is False
        and pred.get("exact_recording_time_known") is True
    )


def temporal_bound_score(pred: dict, gold: dict) -> float | None:
    if not pred or not gold:
        return None
    if pred == gold:
        return 1.0
    iou = temporal_interval_iou(pred, gold)
    if iou is not None:
        return iou
    if pred.get("bound_type") == gold.get("bound_type"):
        return 0.5
    return 0.0
