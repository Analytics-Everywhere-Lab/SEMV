from __future__ import annotations

from src.evidence.provenance_tracker import ProvenanceTracker
from src.schemas.evidence_schema import EvidenceItem


class EvidenceNormalizer:
    def __init__(self) -> None:
        self.provenance_tracker = ProvenanceTracker()

    def normalize(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        deduped: dict[str, EvidenceItem] = {}
        for item in self.provenance_tracker.ensure_provenance(evidence):
            if item.evidence_id not in deduped:
                deduped[item.evidence_id] = item
            else:
                existing = deduped[item.evidence_id]
                deduped[item.evidence_id] = existing.model_copy(
                    update={
                        "reliability": max(existing.reliability, item.reliability),
                        "relevance": max(existing.relevance, item.relevance),
                        "uncertainty_flags": sorted(
                            set(existing.uncertainty_flags + item.uncertainty_flags)
                        ),
                    }
                )
        return list(deduped.values())
