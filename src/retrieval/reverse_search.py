from __future__ import annotations

from src.retrieval.web_search import CachedEvidenceSearch
from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem


class ReverseSearch:
    """Cached text-side reverse-search adapter.

    Real local image matching is implemented in LocalReverseImageSearch and is
    invoked by RawMediaProcessor over actual image/keyframe paths.
    """

    def __init__(self, cached_search: CachedEvidenceSearch | None = None) -> None:
        self.cached_search = cached_search or CachedEvidenceSearch()

    def search(self, claim: SubClaim, plan: ResearchPlan) -> list[EvidenceItem]:
        return [
            item
            for item in self.cached_search.search(claim, plan)
            if item.metadata.get("adapter") == "reverse_search"
            or item.source_type in {"cached_search", "manual_research", "reverse_image_web_candidate"}
        ]
