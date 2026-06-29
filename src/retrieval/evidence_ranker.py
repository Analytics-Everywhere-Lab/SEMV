from __future__ import annotations

from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem


class EvidenceRanker:
    def select_for_claim(
        self,
        claim: SubClaim,
        evidence: list[EvidenceItem],
        evidence_graph: EvidenceGraph,
        top_k: int = 10,
    ) -> list[EvidenceItem]:
        del evidence_graph

        def score(item: EvidenceItem) -> float:
            claim_type_bonus = 0.2 if claim.claim_type in item.supports_claim_types else 0.0
            text = f"{item.title or ''} {item.content}".lower()
            token_overlap = sum(1 for token in claim.statement.lower().split() if token in text)
            overlap_bonus = min(token_overlap / 20.0, 0.2)
            uncertainty_penalty = 0.15 if item.uncertainty_flags else 0.0
            return item.relevance + item.reliability + claim_type_bonus + overlap_bonus - uncertainty_penalty

        return sorted(evidence, key=score, reverse=True)[:top_k]
