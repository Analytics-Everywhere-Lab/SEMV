from __future__ import annotations

from src.schemas.argument_schema import Argument
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem
from src.utils.llm_client import LLMClient


class ArgumentScorer:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def score(
        self,
        claim: object,
        argument: Argument,
        evidence: list[EvidenceItem],
        evidence_graph: EvidenceGraph,
    ) -> Argument:
        del evidence_graph
        linked = [item for item in evidence if item.evidence_id in argument.evidence_ids]
        if linked:
            source_reliability = sum(item.reliability for item in linked) / len(linked)
            claim_relevance = sum(item.relevance for item in linked) / len(linked)
            groundedness = sum(1 for item in linked if item.provenance) / len(linked)
            cross_source_consistency = min(len({item.source for item in linked}) / 3.0, 1.0)
            cross_modal_consistency = min(
                len({item.metadata.get("modality", item.source_type) for item in linked}) / 2.0,
                1.0,
            )
        else:
            source_reliability = argument.source_reliability or argument.reliability
            claim_relevance = argument.claim_relevance or argument.relevance
            groundedness = argument.groundedness or argument.provenance_strength
            cross_source_consistency = argument.cross_source_consistency or argument.corroboration
            cross_modal_consistency = argument.cross_modal_consistency

        specificity = min(max(len(argument.text) / 280.0, 0.25), 1.0)
        intrinsic_score = (
            0.20 * source_reliability
            + 0.25 * claim_relevance
            + 0.15 * cross_source_consistency
            + 0.15 * cross_modal_consistency
            + 0.20 * groundedness
            + 0.05 * specificity
        )
        if not argument.verifier_valid:
            intrinsic_score *= 0.25
        if argument.uncertainty_flags:
            intrinsic_score *= 0.85
        intrinsic_score = max(0.0, min(1.0, intrinsic_score))
        return argument.model_copy(
            update={
                "case_id": getattr(claim, "case_id", argument.case_id),
                "claim_type": getattr(claim, "claim_type", argument.claim_type),
                "source_reliability": source_reliability,
                "claim_relevance": claim_relevance,
                "cross_source_consistency": cross_source_consistency,
                "cross_modal_consistency": cross_modal_consistency,
                "groundedness": groundedness,
                "specificity": specificity,
                "reliability": source_reliability,
                "relevance": claim_relevance,
                "corroboration": cross_source_consistency,
                "provenance_strength": groundedness,
                "intrinsic_strength": intrinsic_score,
                "intrinsic_score": intrinsic_score,
                "strength_components": {
                    "source_reliability": source_reliability,
                    "cross_source_corroboration": cross_source_consistency,
                    "cross_modal_consistency": cross_modal_consistency,
                    "claim_relevance": claim_relevance,
                },
                "score": intrinsic_score,
            }
        )

    def score_all(
        self,
        claim: object,
        arguments: list[Argument],
        evidence: list[EvidenceItem],
        evidence_graph: EvidenceGraph,
        bundle: object | None = None,
    ) -> list[Argument]:
        scored = [self.score(claim, argument, evidence, evidence_graph) for argument in arguments]
        case_id = getattr(bundle, "case_id", None) if bundle is not None else None
        if case_id is None:
            return scored
        return [argument.model_copy(update={"case_id": case_id}) for argument in scored]
