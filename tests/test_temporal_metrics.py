from __future__ import annotations

from src.evaluation.temporal_metrics import publication_event_confusion, temporal_interval_iou


def test_temporal_interval_iou_and_confusion():
    pred = {"earliest_possible": "2024-01-02T00:00:00", "latest_possible": "2024-01-04T00:00:00", "exact_recording_time_known": True}
    gold = {"earliest_possible": "2024-01-01T00:00:00", "latest_possible": "2024-01-03T00:00:00", "exact_recording_time_known": False}

    assert temporal_interval_iou(pred, gold) > 0
    assert publication_event_confusion(pred, gold) is True
