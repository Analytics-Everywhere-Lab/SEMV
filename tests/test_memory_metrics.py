from __future__ import annotations

from src.evaluation.memory_metrics import memory_metrics, paired_memory_comparison
from src.memory.memory_consolidator import MemoryConsolidator
from src.memory.memory_store import MemoryStore

from tests.memory_test_utils import make_candidate, make_memory_config, make_record
from src.schemas.memory_schema import ConsolidationEvent


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



def test_under_review_generalization_is_not_counted_as_success(tmp_path):
    config = make_memory_config(tmp_path)
    store = MemoryStore(config=config)
    proposal = make_record(
        memory_id="mem_proposal",
        memory_type="semantic_rule",
        status="under_review",
        origin="consolidated",
        support_count=3,
        source_case_ids=["a", "b", "c"],
        source_fingerprints=["fa", "fb", "fc"],
        metadata={
            "generalized_from_stm_ids": ["sa", "sb", "sc"],
            "proposal_only": True,
            "generalization_verified": False,
        },
    )
    store.upsert_long_term([proposal])

    metrics = memory_metrics([], [], store=store)

    assert metrics["semantic_generalization_rate"] == 0.0
    assert metrics["semantic_generalization_proposal_count"] == 1
    assert metrics["under_review_semantic_proposal_count"] == 1
    assert metrics["active_semantic_rule_count"] == 0


def test_recovered_active_generalization_is_counted(tmp_path):
    config = make_memory_config(tmp_path)
    store = MemoryStore(config=config)
    recovered = make_record(
        memory_id="mem_recovered",
        memory_type="semantic_rule",
        status="active",
        origin="consolidated",
        support_count=4,
        source_case_ids=["a", "b", "c", "d"],
        source_fingerprints=["fa", "fb", "fc", "fd"],
        metadata={
            "generalized_from_stm_ids": ["sa", "sb", "sc", "sd"],
            "proposal_only": False,
            "generalization_verified": True,
        },
    )
    store.upsert_long_term([recovered])
    store.append_consolidation_event(
        ConsolidationEvent(
            event_id="evt_recovery",
            event_type="generalization_recovered",
            memory_id=recovered.memory_id,
        )
    )

    metrics = memory_metrics([], [], store=store)

    assert metrics["semantic_generalization_rate"] == 1.0
    assert metrics["semantic_generalization_recovery_count"] == 1
    assert metrics["semantic_generalization_proposal_count"] == 0
    assert metrics["active_semantic_rule_count"] == 1


def test_paired_memory_comparison_reports_all_outcomes_and_unmatched_ids():
    memory_on = [
        {"case_id": "positive", "final_label_correct": True},
        {"case_id": "both_wrong", "final_label_correct": False},
        {"case_id": "negative", "final_label_correct": False},
        {"case_id": "both_correct", "final_label_correct": True},
        {"case_id": "on_only", "final_label_correct": True},
    ]
    memory_off = [
        {"case_id": "both_correct", "final_label_correct": True},
        {"case_id": "negative", "final_label_correct": True},
        {"case_id": "both_wrong", "final_label_correct": False},
        {"case_id": "positive", "final_label_correct": False},
        {"case_id": "off_only", "final_label_correct": True},
    ]

    comparison = paired_memory_comparison(memory_on, memory_off)

    assert comparison["paired_case_count"] == 4
    assert comparison["negative_transfer_rate"] == 0.25
    assert comparison["positive_transfer_rate"] == 0.25
    assert comparison["baseline_correct_memory_wrong_count"] == 1
    assert comparison["memory_correct_baseline_wrong_count"] == 1
    assert comparison["both_correct_count"] == 1
    assert comparison["both_wrong_count"] == 1
    assert comparison["missing_from_memory_on"] == ["off_only"]
    assert comparison["missing_from_memory_off"] == ["on_only"]
    assert comparison["paired_case_ids"] == [
        "both_correct", "both_wrong", "negative", "positive"
    ]


def test_paired_memory_comparison_rejects_duplicate_case_ids():
    import pytest

    duplicate = [
        {"case_id": "same", "final_label_correct": True},
        {"case_id": "same", "final_label_correct": False},
    ]
    with pytest.raises(ValueError, match="Duplicate case_id"):
        paired_memory_comparison(duplicate, [])


def test_no_baseline_returns_no_transfer_claim():
    comparison = paired_memory_comparison(
        [{"case_id": "a", "final_label_correct": False}], None
    )
    assert comparison["negative_transfer_rate"] is None
    assert comparison["positive_transfer_rate"] is None
    assert comparison["paired_case_count"] == 0
