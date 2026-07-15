from __future__ import annotations

from src.memory.memory_consolidator import MemoryConsolidator
from src.memory.memory_store import MemoryStore
from src.memory.memory_verifier import MemoryVerifier

from tests.memory_test_utils import make_candidate, make_memory_config, make_record


def _verifier(tmp_path, llm, **overrides):
    config = make_memory_config(tmp_path, **overrides)
    store = MemoryStore(config=config)
    return MemoryVerifier(llm, store=store, config=config), store


def test_llm_exception_is_fail_closed(tmp_path):
    from tests.memory_test_utils import MemoryFakeLLM

    verifier, _ = _verifier(tmp_path, MemoryFakeLLM(raise_on_verify=True))
    result = verifier.verify(make_candidate())

    assert result.verified is False
    assert result.verification_status == "under_review"
    assert result.rejected_reason == "llm_verification_unavailable"


def test_llm_invalid_response_is_fail_closed(tmp_path):
    class WeirdLLM:
        def generate_json(self, prompt, **kwargs):
            return {"unexpected": "shape"}

    verifier, _ = _verifier(tmp_path, WeirdLLM())
    result = verifier.verify(make_candidate())

    assert result.verified is False
    assert result.verification_status == "under_review"
    assert result.rejected_reason == "llm_verification_invalid_response"


def test_configured_min_confidence_is_used(tmp_path):
    from tests.memory_test_utils import MemoryFakeLLM

    verifier, _ = _verifier(
        tmp_path, MemoryFakeLLM(), verification={"min_confidence": 0.9}
    )
    result = verifier.verify(make_candidate(confidence=0.8))

    assert result.verified is False
    assert result.rejected_reason == "confidence_below_threshold"


def test_missing_grounding_is_rejected(tmp_path):
    from tests.memory_test_utils import MemoryFakeLLM

    verifier, _ = _verifier(tmp_path, MemoryFakeLLM())
    candidate = make_candidate(grounding_evidence_ids=[], grounding_argument_ids=[])
    result = verifier.verify(candidate)

    assert result.verified is False
    assert result.rejected_reason == "missing_grounding"


def test_invented_grounding_ids_go_under_review(tmp_path):
    from tests.memory_test_utils import MemoryFakeLLM

    verifier, _ = _verifier(tmp_path, MemoryFakeLLM())
    candidate = make_candidate(grounding_evidence_ids=["ev_fake"])
    result = verifier.verify(candidate, valid_evidence_ids={"ev_real"}, valid_argument_ids={"arg1"})

    assert result.verified is False
    assert result.verification_status == "under_review"
    assert result.rejected_reason == "grounding_evidence_ids_not_in_report"


def test_validation_and_test_split_candidates_are_rejected(tmp_path):
    from tests.memory_test_utils import MemoryFakeLLM

    verifier, _ = _verifier(tmp_path, MemoryFakeLLM())
    for split in ["validation", "test"]:
        result = verifier.verify(make_candidate(split=split))
        assert result.verified is False
        assert result.rejected_reason == "non_training_split"


def test_single_case_universal_rule_goes_under_review(tmp_path):
    from tests.memory_test_utils import MemoryFakeLLM

    verifier, _ = _verifier(tmp_path, MemoryFakeLLM())
    candidate = make_candidate(
        memory_type="semantic_rule",
        text="Always reject every caption that mentions a protest.",
    )
    result = verifier.verify(candidate)

    assert result.verified is False
    assert result.verification_status == "under_review"
    assert result.rejected_reason == "single_case_overgeneralization"


def test_duplicate_from_same_source_is_rejected(tmp_path):
    from tests.memory_test_utils import MemoryFakeLLM

    verifier, store = _verifier(tmp_path, MemoryFakeLLM())
    candidate = make_candidate(case_id="case1")
    existing = make_record(
        canonical_key=None,
        source_case_ids=["case1"],
        source_fingerprints=["fp_case1"],
    )
    store.append(existing)

    result = verifier.verify(candidate)

    assert result.verified is False
    assert result.rejected_reason == "duplicate_from_same_source"


def test_equivalent_candidate_from_new_case_is_not_rejected_as_duplicate(tmp_path):
    from tests.memory_test_utils import MemoryFakeLLM

    verifier, store = _verifier(tmp_path, MemoryFakeLLM())
    store.append(
        make_record(source_case_ids=["case1"], source_fingerprints=["fp_case1"])
    )

    result = verifier.verify(make_candidate(case_id="case2"))

    assert result.verified is True
    assert result.verification_status == "verified"


def test_verified_contradiction_reaches_conflict_count(tmp_path):
    from tests.memory_test_utils import MemoryFakeLLM

    verifier, store = _verifier(tmp_path, MemoryFakeLLM())
    store.append(
        make_record(
            text="Trust reverse image search results when locating an event.",
            confidence=0.95,
            support_count=4,
            source_case_ids=["a", "b", "c", "d"],
            source_fingerprints=["fa", "fb", "fc", "fd"],
        )
    )
    candidate = make_candidate(
        case_id="case9",
        text="Do not trust reverse image search results when locating an event.",
        confidence=0.7,
    )

    result = verifier.verify(candidate)

    assert result.verified is True
    assert result.semantic_relation == "contradicts"
    assert result.related_memory_id == "mem_existing"
    consolidator = MemoryConsolidator(store=store, config=verifier.config, similarity=verifier.similarity)
    consolidator.apply([result])
    consolidation = consolidator.consolidate()
    assert store.load_long_term()[0].conflict_count == 1
    assert consolidation.conflicted
