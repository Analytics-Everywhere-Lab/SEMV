from __future__ import annotations

from src.retrieval.web_search import CachedEvidenceSearch
from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem


class GeolocationSearch:
    def __init__(self, cached_search: CachedEvidenceSearch | None = None) -> None:
        self.cached_search = cached_search or CachedEvidenceSearch()

    def search(self, claim: SubClaim, plan: ResearchPlan) -> list[EvidenceItem]:
        if claim.claim_type != "where":
            return []
        return self.cached_search.search(claim, plan)
