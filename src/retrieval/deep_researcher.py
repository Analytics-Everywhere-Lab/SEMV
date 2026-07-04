from __future__ import annotations

from src.retrieval.factcheck_search import FactCheckSearch
from src.retrieval.free_web_search import FreeWebSearch
from src.retrieval.geolocation_candidate_extractor import GeolocationCandidateExtractor
from src.retrieval.geolocation_search import GeolocationSearch
from src.retrieval.news_search import NewsSearch
from src.retrieval.reverse_search import ReverseSearch
from src.retrieval.web_search import CachedEvidenceSearch
from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.llm_client import LLMClient


class DeepResearcher:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        cached = CachedEvidenceSearch()
        self.adapters = [
            cached,
            ReverseSearch(cached),
            FactCheckSearch(cached),
            NewsSearch(cached),
            GeolocationSearch(cached),
            FreeWebSearch(),
        ]
        self.geolocation_candidate_extractor = GeolocationCandidateExtractor()

    def research(
        self,
        claim: SubClaim,
        plan: ResearchPlan,
        existing_evidence: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        found: dict[str, EvidenceItem] = {}
        for item in self._relevant_existing_evidence(claim, existing_evidence):
            found[item.evidence_id] = item

        if claim.claim_type == "where":
            for item in self.geolocation_candidate_extractor.extract(existing_evidence):
                found[item.evidence_id] = item

        for adapter in self.adapters:
            for item in adapter.search(claim, plan):
                found[item.evidence_id] = item
        if found:
            return list(found.values())

        evidence_id = f"research_gap_{stable_hash_text(claim.claim_id + claim.statement)}"
        return [
            EvidenceItem(
                evidence_id=evidence_id,
                source_type="synthetic_uncertainty",
                source="retrieval",
                title="No cached external evidence found",
                content=(
                    f"No cached/manual external evidence was available for "
                    f"{claim.claim_type} sub-claim: {claim.statement}"
                ),
                reliability=0.2,
                relevance=0.45,
                supports_claim_types=[claim.claim_type],
                uncertainty_flags=["external_research_cache_miss"],
                provenance=Provenance(
                    source_id=evidence_id,
                    source_type="synthetic_uncertainty",
                    source="retrieval",
                    retrieval_method="cached_retrieval_gap",
                ),
            )
        ]

    @staticmethod
    def _relevant_existing_evidence(
        claim: SubClaim,
        existing_evidence: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        relevant = []
        claim_terms = {token for token in claim.statement.lower().split() if len(token) > 3}
        for item in existing_evidence:
            if claim.claim_type in item.supports_claim_types:
                relevant.append(item)
                continue
            text = f"{item.title or ''} {item.content}".lower()
            if claim_terms and any(term in text for term in claim_terms):
                relevant.append(item)
        return relevant
