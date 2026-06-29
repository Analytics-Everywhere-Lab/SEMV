from __future__ import annotations

from src.schemas.case_schema import MultimediaCase
from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem
from src.schemas.memory_schema import MemoryRecord
from src.utils.llm_client import LLMClient


class ResearchPlanner:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def plan(
        self,
        case: MultimediaCase,
        subclaims: list[SubClaim],
        evidence: list[EvidenceItem],
        memory_by_claim: dict[str, list[MemoryRecord]],
    ) -> dict[str, ResearchPlan]:
        plans: dict[str, ResearchPlan] = {}
        for claim in subclaims:
            prompt = (
                "Create a concise research plan for one multimedia verification sub-claim. "
                "Return JSON with questions, search_queries, preferred_sources, uncertainty_checks.\n"
                f"Main claim: {case.claim}\nSub-claim: {claim.statement}\n"
                f"Claim type: {claim.claim_type}\n"
                f"Known evidence: {[item.title or item.content[:80] for item in evidence[:5]]}\n"
                f"Relevant memory: {[item.text for item in memory_by_claim.get(claim.claim_id, [])[:3]]}"
            )
            try:
                data = self.llm_client.generate_json(prompt)
                plans[claim.claim_id] = ResearchPlan(
                    claim_id=claim.claim_id,
                    questions=data.get("questions", []),
                    search_queries=data.get("search_queries", claim.search_queries),
                    preferred_sources=data.get("preferred_sources", []),
                    uncertainty_checks=data.get("uncertainty_checks", []),
                )
            except Exception:
                plans[claim.claim_id] = self._fallback_plan(case, claim)
        return plans

    @staticmethod
    def _fallback_plan(case: MultimediaCase, claim: SubClaim) -> ResearchPlan:
        return ResearchPlan(
            claim_id=claim.claim_id,
            questions=[
                f"What evidence supports the {claim.claim_type} sub-claim?",
                f"What evidence attacks the {claim.claim_type} sub-claim?",
            ],
            search_queries=claim.search_queries or [case.claim, claim.statement],
            preferred_sources=["case evidence", "cached web evidence", "cached fact checks"],
            uncertainty_checks=[
                "Check whether evidence directly addresses the sub-claim.",
                "Check whether media provenance conflicts with the stated context.",
            ],
        )
