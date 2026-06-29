from __future__ import annotations

from src.memory.memory_store import MemoryStore
from src.schemas.case_schema import MultimediaCase
from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceItem
from src.schemas.memory_schema import MemoryRecord


class MemoryRetriever:
    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store or MemoryStore()

    def retrieve(
        self,
        case: MultimediaCase,
        claim: SubClaim,
        evidence: list[EvidenceItem],
        top_k: int = 5,
    ) -> list[MemoryRecord]:
        del evidence
        query = f"{case.claim} {case.context or ''} {claim.claim_type} {claim.statement}".lower()

        def score(record: MemoryRecord) -> float:
            text = f"{record.text} {' '.join(record.tags)} {record.claim_type or ''}".lower()
            overlap = sum(1 for token in query.split() if token in text)
            type_bonus = 3 if record.claim_type == claim.claim_type else 0
            return overlap + type_bonus + record.confidence

        return sorted(self.store.load_all(), key=score, reverse=True)[:top_k]

    def retrieve_for_claims(
        self,
        bundle,
        claims,
        evidence,
        source_clusters=None,
        top_k: int = 5,
    ) -> dict[str, list[MemoryRecord]]:
        del evidence, source_clusters
        records = self.store.load_all()
        results: dict[str, list[MemoryRecord]] = {}
        for claim in claims:
            query_parts = [
                bundle.input.title or "",
                bundle.input.caption or "",
                bundle.input.description or "",
                claim.claim_type,
                claim.statement,
                bundle.task.task_type,
            ]
            query = " ".join(query_parts).lower()

            def score(record: MemoryRecord) -> float:
                text = " ".join(
                    str(part)
                    for part in [
                        record.text,
                        record.lesson,
                        record.trigger_pattern,
                        record.evidence_pattern,
                        record.argument_pattern,
                        " ".join(record.tags),
                        record.claim_type or "",
                        record.task_type or "",
                    ]
                    if part
                ).lower()
                overlap = sum(1 for token in query.split() if token and token in text)
                type_bonus = 3 if record.claim_type in {claim.claim_type, "general"} else 0
                task_bonus = 1 if record.task_type in {bundle.task.task_type, bundle.task.subtask} else 0
                active_bonus = 1 if record.status == "active" else -2
                return overlap + type_bonus + task_bonus + active_bonus + record.confidence

            results[claim.claim_id] = sorted(records, key=score, reverse=True)[:top_k]
        return results
