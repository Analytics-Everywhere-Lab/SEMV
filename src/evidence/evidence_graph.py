from __future__ import annotations

from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem


class EvidenceGraphBuilder:
    def build(
        self,
        evidence: list[EvidenceItem],
        claims: list[SubClaim] | None = None,
    ) -> EvidenceGraph:
        graph = EvidenceGraph()
        for item in evidence:
            graph.nodes[item.evidence_id] = {
                "node_type": "evidence",
                "title": item.title,
                "source_type": item.source_type,
                "source": item.source,
                "url": item.url,
                "reliability": item.reliability,
                "uncertainty_flags": item.uncertainty_flags,
            }
            if item.provenance:
                source_node_id = f"source:{item.provenance.source_id}"
                graph.nodes[source_node_id] = {
                    "node_type": "source",
                    "source_type": item.provenance.source_type,
                    "source": item.provenance.source,
                    "url": item.provenance.url,
                    "retrieval_method": item.provenance.retrieval_method,
                }
                graph.edges.append(
                    {
                        "from": item.evidence_id,
                        "to": source_node_id,
                        "relation": "provenance",
                    }
                )

        for claim in claims or []:
            graph.nodes[claim.claim_id] = {
                "node_type": "claim",
                "claim_type": claim.claim_type,
                "statement": claim.statement,
            }
            for item in evidence:
                if claim.claim_type in item.supports_claim_types or self._mentions_claim(
                    claim, item
                ):
                    graph.edges.append(
                        {
                            "from": claim.claim_id,
                            "to": item.evidence_id,
                            "relation": "uses_evidence",
                        }
                    )
        return graph

    @staticmethod
    def _mentions_claim(claim: SubClaim, item: EvidenceItem) -> bool:
        text = f"{item.title or ''} {item.content}".lower()
        return any(token in text for token in claim.statement.lower().split()[:8])
