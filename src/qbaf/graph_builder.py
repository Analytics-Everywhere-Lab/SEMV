from __future__ import annotations

from src.schemas.argument_schema import Argument
from src.schemas.qbaf_schema import QBAFEdge, QBAFGraph, QBAFNode


class QBAFGraphBuilder:
    def build(self, claim: object, arguments: list[Argument]) -> QBAFGraph:
        claim_id = getattr(claim, "claim_id")
        claim_type = getattr(claim, "claim_type")
        statement = getattr(claim, "statement")
        graph = QBAFGraph(claim_id=claim_id)
        graph.nodes[claim_id] = QBAFNode(
            node_id=claim_id,
            node_type="claim",
            base_score=0.5,
            final_score=0.5,
            metadata={"claim_type": claim_type, "statement": statement},
        )
        for argument in arguments:
            argument_score = argument.score or argument.intrinsic_score
            graph.nodes[argument.argument_id] = QBAFNode(
                node_id=argument.argument_id,
                node_type="argument",
                base_score=argument_score,
                final_score=argument_score,
                metadata={
                    "stance": argument.stance,
                    "evidence_ids": argument.evidence_ids,
                    "uncertainty_flags": argument.uncertainty_flags,
                },
            )
            if argument.stance in {"support", "attack"}:
                graph.edges.append(
                    QBAFEdge(
                        from_node=argument.argument_id,
                        to_node=claim_id,
                        relation=argument.stance,
                        weight=argument_score,
                    )
                )
        return graph
