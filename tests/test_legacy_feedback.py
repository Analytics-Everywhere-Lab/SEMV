from __future__ import annotations

import pytest

from src.main import run_case
from src.schemas.case_schema import MultimediaCase


def test_supported_legacy_feedback_reaches_contestation_batch(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_run_case_bundle(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("src.main.run_case_bundle", fake_run_case_bundle)
    case = MultimediaCase(case_id="c1", claim="claim")
    feedback = {
        "case_id": "c1", "reviewer_id": "r1",
        "contestations": [{
            "contestation_id": "h1", "case_id": "c1", "action": "add",
            "added_subclaim_id": "s1", "added_text": "counter", "added_stance": "attack",
        }],
    }
    assert run_case(case, "inference_only", human_feedback=feedback) is sentinel
    batch = captured["human_review_batch"]
    assert batch.reviewer_id == "r1"
    assert batch.contestations[0].action == "add"


def test_malformed_legacy_feedback_is_actionable():
    case = MultimediaCase(case_id="c1", claim="claim")
    with pytest.raises(ValueError, match="HumanReviewBatch|human_feedback"):
        run_case(case, "inference_only", human_feedback={"human_review_batch": "bad"})


def test_none_legacy_feedback_remains_backward_compatible(monkeypatch):
    captured = {}
    monkeypatch.setattr("src.main.run_case_bundle", lambda **kwargs: captured.update(kwargs) or object())
    run_case(MultimediaCase(case_id="c1", claim="claim"), "inference_only", human_feedback=None)
    assert captured["human_review_batch"] is None
