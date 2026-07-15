from __future__ import annotations

from src.evaluation.memory_metrics import memory_metrics
from src.memory.memory_consolidator import MemoryConsolidator
from src.memory.memory_store import MemoryStore

from tests.memory_test_utils import make_candidate, make_memory_config


def test_memory_metrics_without_store_returns_association_only():
    predictions = [
        {"case_id": "a", "memory_used_ids": ["m1"], "memory_retrieved_ids": ["m1", "m2"]},
        {"case_id": "b", "memory_used_ids": [], "memory_retrieved_ids": ["m1"]},
    ]
    per_case = [
        {"case_id": "a", "final_label_correct": True},
        {"case_id": "b", "final_label_correct": False},
    ]

    metrics = memory_metrics(predictions, per_case)

    assert metrics["memory_citation_rate"] == 0.5
    assert metrics["memory_associated_correctness_rate"] == 1.0
    assert metrics["memory_associated_error_rate"] == 0.0
    # Without a paired memory-off run, negative transfer is not claimed.
    assert metrics["negative_transfer_rate"] is None


def test_paired_negative_transfer_uses_paired_runs():
    with_memory = [
        {"case_id": "a", "final_label_correct": False},
        {"case_id": "b", "final_label_correct": True},
    ]
    without_memory = [
        {"case_id": "a", "final_label_correct": True},
        {"case_id": "b", "final_label_correct": True},
    ]
    metrics = memory_metrics(
        [{"case_id": "a", "memory_used_ids": ["m1"]}, {"case_id": "b", "memory_used_ids": []}],
        with_memory,
        paired_baseline_case_metrics=without_memory,
    )
    assert metrics["negative_transfer_rate"] == 0.5


def test_store_metrics_reflect_lifecycle(tmp_path):
    config = make_memory_config(tmp_path)
    store = MemoryStore(config=config)
    consolidator = MemoryConsolidator(store=store, config=config)
    text = "When reverse search finds an earlier upload, attack the temporal claim."
    consolidator.apply(
        [
            make_candidate(
                case_id="case1", text=text, verified=True, failure_type="weak_provenance"
            ),
            make_candidate(
                case_id="case2", text=text, verified=True, failure_type="weak_provenance"
            ),
        ]
    )
    consolidator.consolidate()

    metrics = memory_metrics([], [], store=store)

    assert metrics["lesson_acceptance_rate"] == 1.0
    assert metrics["stm_to_ltm_promotion_rate"] == 1.0
    assert metrics["active_memory_count"] == 1
    assert metrics["average_independent_support"] == 2.0
    assert metrics["failure_recurrence_rate"] == 0.5
