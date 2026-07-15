from __future__ import annotations

from src.reflection.lesson_generator import LessonGenerator
from src.schemas.argument_schema import Argument
from src.schemas.evidence_schema import EvidenceItem
from src.schemas.report_schema import ReflectionLog, SubClaimReport, VerificationReport

from tests.memory_test_utils import MemoryFakeLLM


def _report() -> VerificationReport:
    argument = Argument(
        argument_id="arg1",
        claim_id="c1",
        stance="support",
        text="Reverse search shows the same footage from the claimed date.",
        evidence_ids=["ev1"],
        score=0.8,
    )
    return VerificationReport(
        case_id="case1",
        final_status="false_context",
        final_confidence=0.7,
        subclaim_reports=[
            SubClaimReport(
                claim_id="c1",
                claim_type="where",
                statement="Filmed at the claimed location.",
                score=0.6,
                decision="supported",
                confidence=0.7,
                top_support_arguments=[argument],
            )
        ],
        evidence=[
            EvidenceItem(evidence_id="ev1", content="An archived article describing the event.")
        ],
        metadata={
            "dataset": {"dataset_name": "mv2026", "dataset_split": "train"},
            "task": {"task_type": "multimedia_verification"},
            "input": {"title": "Explosion at the port"},
        },
    )


def _reflection(**overrides) -> ReflectionLog:
    base = dict(
        case_id="case1",
        predicted_label="false_context",
        ground_truth_label="false_context",
        failure_modes=[],
        lessons=[],
    )
    base.update(overrides)
    return ReflectionLog(**base)


def test_structured_generation_uses_one_llm_call_and_no_unconditional_semantic():
    llm = MemoryFakeLLM(
        lessons={
            "episodic": {
                "observation": "Reverse search resolved the location claim.",
                "confidence": 0.9,
                "evidence_ids": ["ev1"],
                "argument_ids": ["arg1"],
            },
            "failures": [],
            "semantic": None,
        }
    )
    candidates = LessonGenerator(llm).generate(_report(), _reflection())

    reflection_calls = [c for c in llm.calls if "You are the reflection module" in c]
    assert len(reflection_calls) == 1
    assert [c.memory_type for c in candidates] == ["episodic"]
    assert candidates[0].grounding_evidence_ids == ["ev1"]
    assert candidates[0].source_fingerprint
    assert candidates[0].dataset_name == "mv2026"
    assert candidates[0].dataset_split == "train"


def test_semantic_candidate_requires_generalizable_pattern():
    llm = MemoryFakeLLM(
        lessons={
            "episodic": {"observation": "obs", "confidence": 0.8},
            "failures": [],
            "semantic": {
                "generalizable_pattern": "",
                "lesson": "Some rule.",
                "trigger_pattern": "trigger",
                "recommended_action": "act",
            },
        }
    )
    candidates = LessonGenerator(llm).generate(_report(), _reflection())
    assert all(c.memory_type != "semantic_rule" for c in candidates)


def test_invented_grounding_ids_mark_candidate_under_review():
    llm = MemoryFakeLLM(
        lessons={
            "episodic": {
                "observation": "obs",
                "confidence": 0.9,
                "evidence_ids": ["ev_invented"],
                "argument_ids": [],
            },
            "failures": [],
            "semantic": None,
        }
    )
    candidates = LessonGenerator(llm).generate(_report(), _reflection())
    episodic = candidates[0]
    assert episodic.verification_status == "under_review"
    assert episodic.rejected_reason == "candidate_referenced_unknown_ids"
    # Grounding fell back to real report IDs, never invented ones.
    assert set(episodic.grounding_evidence_ids) <= {"ev1"}
    assert set(episodic.grounding_argument_ids) <= {"arg1"}


def test_generation_failure_falls_back_to_grounded_episodic_under_review():
    llm = MemoryFakeLLM(raise_on_lessons=True)
    candidates = LessonGenerator(llm).generate(
        _report(), _reflection(failure_modes=["final_label_mismatch"])
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.memory_type == "episodic"
    assert candidate.verification_status == "under_review"
    assert candidate.rejected_reason == "structured_lesson_generation_failed"
    assert candidate.grounding_evidence_ids or candidate.grounding_argument_ids
    # No generic semantic fallback sentence is produced.
    assert all(c.memory_type != "semantic_rule" for c in candidates)


def test_failure_lessons_are_case_specific():
    llm = MemoryFakeLLM(
        lessons={
            "episodic": {"observation": "obs", "confidence": 0.8},
            "failures": [
                {
                    "failure_type": "temporal_publication_event_confusion",
                    "trigger_pattern": "publication time treated as event time",
                    "lesson": "Bound the event time by publication time only.",
                    "recommended_action": "Map When to partially_verified.",
                    "confidence": 0.8,
                    "evidence_ids": ["ev1"],
                }
            ],
            "semantic": None,
        }
    )
    candidates = LessonGenerator(llm).generate(
        _report(), _reflection(failure_modes=["temporal_publication_event_confusion"])
    )
    failure = next(c for c in candidates if c.memory_type == "failure")
    assert failure.failure_type == "temporal_publication_event_confusion"
    assert failure.trigger_pattern
    assert failure.recommended_action
    assert failure.polarity == "negative"
