from __future__ import annotations

from src.schemas.qbaf_schema import QBAFGraph


class QBAFPropagator:
    def propagate(self, graph: QBAFGraph) -> QBAFGraph:
        support = sum(edge.weight for edge in graph.edges if edge.relation == "support")
        attack = sum(edge.weight for edge in graph.edges if edge.relation == "attack")
        base = graph.nodes.get(graph.claim_id).base_score if graph.claim_id in graph.nodes else 0.5
        energy = support - attack
        claim_score = base + (1.0 - base) * _h(energy) - base * _h(-energy)
        claim_score = max(0.0, min(1.0, claim_score))
        updated_nodes = dict(graph.nodes)
        if graph.claim_id in updated_nodes:
            updated_nodes[graph.claim_id] = updated_nodes[graph.claim_id].model_copy(
                update={"final_score": claim_score}
            )
        flags = list(graph.uncertainty_flags)
        if support > 0.55 and attack > 0.55:
            flags.append("competing_high_weight_arguments")
        return graph.model_copy(
            update={
                "nodes": updated_nodes,
                "claim_score": claim_score,
                "uncertainty_flags": sorted(set(flags)),
            }
        )


def _h(value: float) -> float:
    positive = max(value, 0.0)
    return positive**2 / (1.0 + positive**2)
