from __future__ import annotations

from src.schemas.memory_schema import MemoryRecord, MemoryUpdateCandidate


class MemoryConflictChecker:
    def has_conflict(
        self,
        candidate: MemoryUpdateCandidate,
        existing: list[MemoryRecord],
    ) -> tuple[bool, str | None]:
        candidate_text = candidate.text.lower()
        for record in existing:
            record_text = record.text.lower()
            if candidate.memory_type == record.memory_type and candidate_text == record_text:
                return True, "duplicate_memory"
            if "always" in candidate_text and "never" in record_text:
                return True, "semantic_rule_conflict"
            if "never" in candidate_text and "always" in record_text:
                return True, "semantic_rule_conflict"
        return False, None
