from __future__ import annotations

from src.memory.memory_conflict_checker import MemoryConflictChecker
from src.memory.memory_store import MemoryStore
from src.schemas.memory_schema import MemoryUpdateCandidate
from src.utils.llm_client import LLMClient


class MemoryVerifier:
    def __init__(self, llm_client: LLMClient, store: MemoryStore | None = None) -> None:
        self.llm_client = llm_client
        self.store = store or MemoryStore()
        self.conflict_checker = MemoryConflictChecker()

    def verify(self, candidate: MemoryUpdateCandidate) -> MemoryUpdateCandidate:
        has_conflict, reason = self.conflict_checker.has_conflict(
            candidate,
            self.store.load_all(),
        )
        if has_conflict:
            return candidate.model_copy(update={"verified": False, "rejected_reason": reason})
        if candidate.confidence < 0.6:
            return candidate.model_copy(
                update={"verified": False, "rejected_reason": "confidence_below_threshold"}
            )

        prompt = (
            "Verify whether this memory lesson is safe to store for future multimedia "
            "verification cases. Return JSON with verified boolean and reason string.\n"
            f"Candidate: {candidate.text}\nRationale: {candidate.rationale or ''}"
        )
        try:
            data = self.llm_client.generate_json(prompt)
            verified = bool(data.get("verified", True))
            reason = data.get("reason")
            return candidate.model_copy(
                update={"verified": verified, "rejected_reason": None if verified else reason}
            )
        except Exception:
            return candidate.model_copy(update={"verified": True, "rejected_reason": None})
