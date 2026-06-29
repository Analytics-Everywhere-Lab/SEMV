from __future__ import annotations

from src.memory.memory_store import MemoryStore
from src.schemas.memory_schema import MemoryRecord, MemoryUpdateCandidate
from src.utils.hashing import stable_hash_text


class MemoryConsolidator:
    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store or MemoryStore()

    def apply(self, candidates: list[MemoryUpdateCandidate]) -> list[MemoryRecord]:
        applied: list[MemoryRecord] = []
        for candidate in candidates:
            if not candidate.verified:
                continue
            record = MemoryRecord(
                memory_id=f"mem_{stable_hash_text(candidate.source_case_id + candidate.text)}",
                memory_type=candidate.memory_type,
                case_id=candidate.source_case_id,
                claim_type=candidate.claim_type,
                text=candidate.text,
                tags=[candidate.memory_type, candidate.claim_type or "general"],
                confidence=candidate.confidence,
                metadata={
                    "candidate_id": candidate.candidate_id,
                    "rationale": candidate.rationale,
                    **candidate.metadata,
                },
            )
            self.store.append(record)
            applied.append(record)
        return applied
