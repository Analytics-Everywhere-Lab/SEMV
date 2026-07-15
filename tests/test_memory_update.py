from __future__ import annotations

from src.memory.memory_consolidator import MemoryConsolidator
from src.memory.memory_store import MemoryStore
from src.memory.memory_verifier import MemoryVerifier
from src.schemas.memory_schema import MemoryUpdateCandidate

from tests.conftest import FakeLLMClient


def _candidate(**overrides) -> MemoryUpdateCandidate:
    base = dict(
        candidate_id="cand1",
        memory_type="semantic_rule",
        text="Keep weak provenance uncertain.",
        source_case_id="case1",
        dataset_split="train",
        grounding_evidence_ids=["ev1"],
        confidence=0.8,
    )
    base.update(overrides)
    return MemoryUpdateCandidate(**base)


def test_verified_candidate_is_staged_in_short_term_not_long_term(tmp_path):
    store = MemoryStore(tmp_path)
    verified = MemoryVerifier(FakeLLMClient(), store).verify(_candidate())
    staged = MemoryConsolidator(store).apply([verified])

    assert verified.verified is True
    assert verified.verification_status == "verified"
    assert len(staged) == 1
    assert staged[0].status == "staged"
    # The candidate lands in the short-term staging store...
    assert (tmp_path / "short_term_memory.jsonl").read_text(encoding="utf-8").strip()
    # ...and is NOT appended to the long-term semantic store.
    assert (tmp_path / "semantic_rules.jsonl").read_text(encoding="utf-8").strip() == ""


def test_memory_verifier_rejects_low_confidence(tmp_path):
    store = MemoryStore(tmp_path)
    candidate = _candidate(
        candidate_id="cand2",
        memory_type="failure",
        text="Maybe useful but too weak.",
        confidence=0.2,
    )

    verified = MemoryVerifier(FakeLLMClient(), store).verify(candidate)

    assert verified.verified is False
    assert verified.verification_status == "rejected"
    assert verified.rejected_reason == "confidence_below_threshold"
