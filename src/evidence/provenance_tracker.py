from __future__ import annotations

from src.schemas.evidence_schema import EvidenceItem, Provenance


class ProvenanceTracker:
    def ensure_provenance(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        normalized = []
        for item in evidence:
            if item.provenance is None:
                item = item.model_copy(
                    update={
                        "provenance": Provenance(
                            source_id=item.evidence_id,
                            source_type=item.source_type,
                            source=item.source,
                            url=item.url,
                            retrieval_method="implicit",
                        )
                    }
                )
            normalized.append(item)
        return normalized
