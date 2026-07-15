from __future__ import annotations

from src.memory.memory_config import MemoryConfig
from src.memory.memory_similarity import SimilarityBackend, build_similarity_backend
from src.schemas.memory_schema import MemoryRecord, MemoryUpdateCandidate, SemanticRelation


class MemoryConflictChecker:
    """Semantic relation checking between a candidate and existing memory.

    Delegates to the pluggable similarity backend (canonical/lexical/structured
    matching with optional LLM classification of ambiguous pairs) instead of the
    old exact-duplicate and always/never keyword heuristics.
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        similarity: SimilarityBackend | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.similarity = similarity or build_similarity_backend(self.config)

    def relation_to(
        self,
        candidate: MemoryUpdateCandidate,
        record: MemoryRecord,
    ) -> SemanticRelation:
        return self.similarity.relation(candidate.model_dump(), record.model_dump())

    def has_conflict(
        self,
        candidate: MemoryUpdateCandidate,
        existing: list[MemoryRecord],
    ) -> tuple[bool, str | None]:
        for record in existing:
            relation = self.relation_to(candidate, record)
            if relation == "equivalent" and candidate.source_case_id in record.source_case_ids:
                return True, "duplicate_memory"
            if relation == "contradicts":
                return True, f"contradicts:{record.memory_id}"
        return False, None
