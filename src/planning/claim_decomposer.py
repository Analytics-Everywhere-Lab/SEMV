from __future__ import annotations

from src.schemas.case_schema import MultimediaCase
from src.schemas.claim_schema import ClaimType, SubClaim
from src.schemas.evidence_schema import EvidenceItem
from src.utils.diagnostics import record_fallback
from src.utils.hashing import stable_hash_text
from src.utils.llm_client import LLMClient


DEFAULT_CLAIM_TYPES: list[ClaimType] = [
    "what",
    "where",
    "when",
    "who",
    "why",
    "authenticity",
]


class ClaimDecomposer:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def decompose(self, case: MultimediaCase, evidence: list[EvidenceItem]) -> list[SubClaim]:
        prompt = (
            "Decompose this multimedia verification claim into What, Where, When, "
            "Who, Why, and Authenticity sub-claims. Return JSON as "
            '{"subclaims":[{"claim_type":"what","statement":"...","search_queries":["..."]}]}.\n'
            f"Claim: {case.claim}\nContext: {case.context or ''}\n"
            f"Evidence summaries: {[item.content[:160] for item in evidence[:5]]}"
        )
        try:
            data = self.llm_client.generate_json(prompt)
            subclaims = []
            for item in data.get("subclaims", []):
                claim_type = item.get("claim_type")
                if claim_type in DEFAULT_CLAIM_TYPES:
                    subclaims.append(
                        SubClaim(
                            claim_id=self._claim_id(case.case_id, claim_type),
                            claim_type=claim_type,
                            statement=item.get("statement") or self._fallback_statement(case, claim_type),
                            search_queries=item.get("search_queries", [case.claim]),
                        )
                    )
            if subclaims:
                return self._ensure_all_types(case, subclaims)
        except Exception as exc:
            record_fallback("claim_decomposition", exc, "deterministic_claim_templates", case_id=case.case_id)
        return [
            SubClaim(
                claim_id=self._claim_id(case.case_id, claim_type),
                claim_type=claim_type,
                statement=self._fallback_statement(case, claim_type),
                search_queries=[case.claim, f"{claim_type} verification {case.claim}"],
            )
            for claim_type in DEFAULT_CLAIM_TYPES
        ]

    def _ensure_all_types(self, case: MultimediaCase, subclaims: list[SubClaim]) -> list[SubClaim]:
        existing = {claim.claim_type for claim in subclaims}
        for claim_type in DEFAULT_CLAIM_TYPES:
            if claim_type not in existing:
                subclaims.append(
                    SubClaim(
                        claim_id=self._claim_id(case.case_id, claim_type),
                        claim_type=claim_type,
                        statement=self._fallback_statement(case, claim_type),
                        search_queries=[case.claim, f"{claim_type} verification {case.claim}"],
                    )
                )
        return subclaims

    @staticmethod
    def _claim_id(case_id: str, claim_type: str) -> str:
        return f"{case_id}_{claim_type}_{stable_hash_text(case_id + claim_type, 8)}"

    @staticmethod
    def _fallback_statement(case: MultimediaCase, claim_type: str) -> str:
        templates = {
            "what": "What event or action is depicted by the media?",
            "where": "Where was the media captured or where did the depicted event occur?",
            "when": "When was the media captured or when did the depicted event occur?",
            "who": "Who is involved or represented in the media?",
            "why": "Why is the media being shared, and is the implied context justified?",
            "authenticity": "Is the media authentic, unmanipulated, and presented in the correct context?",
        }
        return f"{templates[claim_type]} Main claim: {case.claim}"
