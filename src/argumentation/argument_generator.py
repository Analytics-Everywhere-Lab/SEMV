from __future__ import annotations

from src.schemas.argument_schema import Argument
from src.schemas.claim_schema import SubClaim
from src.schemas.contestation_schema import ArgumentProvenance
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem
from src.schemas.memory_schema import MemoryRecord
from src.utils.hashing import stable_hash_text
from src.utils.llm_client import LLMClient


def _argument_provenance(claim: SubClaim, evidence_ids: list[str]) -> ArgumentProvenance:
    return ArgumentProvenance(
        source_step="argument_construction",
        subclaim_id=claim.claim_id,
        evidence_ids=list(evidence_ids),
        retrieval_query_ids=list(claim.search_queries),
        upstream_steps=[
            "claim_decomposition",
            "evidence_retrieval",
            "evidence_validation",
            "argument_construction",
        ],
    )


class ArgumentGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        claim: SubClaim,
        evidence: list[EvidenceItem],
        evidence_graph: EvidenceGraph,
        memory_items: list[MemoryRecord],
    ) -> list[Argument]:
        del evidence_graph
        prompt = (
            "Generate concise support and attack arguments for this verification "
            "sub-claim. Return JSON as "
            '{"arguments":[{"stance":"support","title":"...","text":"...","evidence_ids":["..."]}]}.\n'
            f"Sub-claim: {claim.statement}\n"
            f"Evidence: {[{'id': item.evidence_id, 'text': item.content[:220]} for item in evidence]}\n"
            f"Memory: {[item.text for item in memory_items[:3]]}"
        )
        try:
            data = self.llm_client.generate_json(prompt)
            arguments = []
            for item in data.get("arguments", []):
                stance = item.get("stance", "neutral")
                if stance not in {"support", "attack", "mixed", "neutral"}:
                    stance = "neutral"
                evidence_ids = [
                    evidence_id
                    for evidence_id in item.get("evidence_ids", [])
                    if any(ev.evidence_id == evidence_id for ev in evidence)
                ]
                arguments.append(
                    Argument(
                        argument_id=self._argument_id(claim.claim_id, item.get("text", "")),
                        claim_id=claim.claim_id,
                        stance=stance,
                        title=item.get("title") or f"{stance.title()} argument",
                        text=item.get("text") or "No argument text returned.",
                        evidence_ids=evidence_ids,
                        provenance=_argument_provenance(claim, evidence_ids),
                    )
                )
            if arguments:
                return arguments
        except Exception:
            pass
        return self._fallback_arguments(claim, evidence)

    def _fallback_arguments(self, claim: SubClaim, evidence: list[EvidenceItem]) -> list[Argument]:
        arguments = []
        for item in evidence:
            stance = self._infer_stance(item)
            arguments.append(
                Argument(
                    argument_id=self._argument_id(claim.claim_id, item.evidence_id + stance),
                    claim_id=claim.claim_id,
                    stance=stance,
                    title=f"{stance.title()} from {item.title or item.source}",
                    text=f"{item.content}",
                    evidence_ids=[item.evidence_id],
                    provenance=_argument_provenance(claim, [item.evidence_id]),
                    relevance=item.relevance,
                    reliability=item.reliability,
                    corroboration=0.55,
                    provenance_strength=0.65 if item.provenance else 0.35,
                    specificity=0.65 if len(item.content) > 80 else 0.45,
                    uncertainty_flags=item.uncertainty_flags,
                )
            )
        return arguments

    @staticmethod
    def _infer_stance(item: EvidenceItem) -> str:
        metadata_stance = item.metadata.get("stance")
        if metadata_stance in {"support", "attack", "mixed", "neutral"}:
            return metadata_stance
        text = item.content.lower()
        attack_terms = ["false", "misleading", "not ", "unrelated", "different", "old", "out of context"]
        if any(term in text for term in attack_terms) or item.uncertainty_flags:
            return "attack"
        return "support"

    @staticmethod
    def _argument_id(claim_id: str, seed: str) -> str:
        return f"arg_{stable_hash_text(claim_id + seed)}"
