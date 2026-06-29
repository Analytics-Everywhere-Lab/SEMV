from __future__ import annotations

from src.memory.memory_consolidator import MemoryConsolidator
from src.memory.memory_store import MemoryStore
from src.memory.memory_verifier import MemoryVerifier
from src.schemas.memory_schema import MemoryUpdateCandidate

from tests.conftest import FakeLLMClient


def test_memory_update_verifies_and_appends_to_correct_store(tmp_path):
    store = MemoryStore(tmp_path)
    candidate = MemoryUpdateCandidate(
        candidate_id="cand1",
        memory_type="semantic_rule",
        text="Keep weak provenance uncertain.",
        source_case_id="case1",
        confidence=0.8,
    )

    verified = MemoryVerifier(FakeLLMClient(), store).verify(candidate)
    applied = MemoryConsolidator(store).apply([verified])

    assert verified.verified is True
    assert len(applied) == 1
    assert (tmp_path / "semantic_rules.jsonl").read_text(encoding="utf-8").strip()


def test_memory_verifier_rejects_low_confidence(tmp_path):
    store = MemoryStore(tmp_path)
    candidate = MemoryUpdateCandidate(
        candidate_id="cand1",
        memory_type="failure",
        text="Maybe useful but too weak.",
        source_case_id="case1",
        confidence=0.2,
    )

    verified = MemoryVerifier(FakeLLMClient(), store).verify(candidate)

    assert verified.verified is False
    assert verified.rejected_reason == "confidence_below_threshold"
