from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.schemas.case_schema import MultimediaCase
from src.schemas.claim_schema import ClaimType, ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem
from src.schemas.memory_schema import MemoryRecord
from src.utils.env_loader import get_bool_env, get_int_env
from src.utils.llm_client import LLMClient


PLAIN_CLAIM_TYPES: set[ClaimType] = {
    "what",
    "where",
    "when",
    "who",
    "why",
    "authenticity",
}


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
        plain_claims = [claim for claim in subclaims if claim.claim_type in PLAIN_CLAIM_TYPES]
        other_claims = [claim for claim in subclaims if claim.claim_type not in PLAIN_CLAIM_TYPES]

        max_workers = self._max_parallel_workers(len(plain_claims))
        if max_workers <= 1:
            for claim in plain_claims:
                plans[claim.claim_id] = self._plan_claim(case, claim, evidence, memory_by_claim)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._plan_claim,
                        case,
                        claim,
                        evidence,
                        memory_by_claim,
                    ): claim
                    for claim in plain_claims
                }
                for future in as_completed(futures):
                    claim = futures[future]
                    try:
                        plans[claim.claim_id] = future.result()
                    except Exception:
                        plans[claim.claim_id] = self._fallback_plan(case, claim)

        for claim in other_claims:
            plans[claim.claim_id] = self._plan_claim(case, claim, evidence, memory_by_claim)
        return plans

    def _plan_claim(
        self,
        case: MultimediaCase,
        claim: SubClaim,
        evidence: list[EvidenceItem],
        memory_by_claim: dict[str, list[MemoryRecord]],
    ) -> ResearchPlan:
        memory_items = memory_by_claim.get(claim.claim_id, [])[:3]
        memory_lines = [f"[{item.memory_id}] {item.text}" for item in memory_items]
        valid_memory_ids = {item.memory_id for item in memory_items}
        prompt = (
            "Create a concise research plan for one multimedia verification sub-claim. "
            "Return JSON with questions, search_queries, preferred_sources, "
            "uncertainty_checks, and used_memory_ids. Memory is guidance only, not "
            "evidence; list a memory id in used_memory_ids only if that memory "
            "actually shaped the plan.\n"
            f"Main claim: {case.claim}\nSub-claim: {claim.statement}\n"
            f"Claim type: {claim.claim_type}\n"
            f"Known evidence: {[item.title or item.content[:80] for item in evidence[:5]]}\n"
            f"Relevant memory: {memory_lines}"
        )
        try:
            data = self.llm_client.generate_json(prompt)
            used_memory_ids = [
                memory_id
                for memory_id in data.get("used_memory_ids", [])
                if memory_id in valid_memory_ids
            ]
            return ResearchPlan(
                claim_id=claim.claim_id,
                questions=data.get("questions", []),
                search_queries=data.get("search_queries", claim.search_queries),
                preferred_sources=data.get("preferred_sources", []),
                uncertainty_checks=data.get("uncertainty_checks", []),
                used_memory_ids=used_memory_ids,
                metadata={"used_memory_ids": used_memory_ids},
            )
        except Exception:
            return self._fallback_plan(case, claim)

    @staticmethod
    def _max_parallel_workers(claim_count: int) -> int:
        if claim_count <= 1 or not get_bool_env("SEMV_PARALLEL_RESEARCH_PLANNING", True):
            return 1
        max_workers = get_int_env("SEMV_MAX_WORKERS", 2)
        return max(1, min(claim_count, max_workers))

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
