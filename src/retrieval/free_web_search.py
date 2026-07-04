from __future__ import annotations

from src.retrieval.web_article_extractor import WebArticleExtractor
from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.tool_config import retrieval_config


class FreeWebSearch:
    def __init__(self, config: dict | None = None) -> None:
        self.config = retrieval_config(config)
        self.extractor = WebArticleExtractor()

    def search(self, claim: SubClaim, plan: ResearchPlan) -> list[EvidenceItem]:
        if not self.config.get("free_web_search_enabled", False):
            return []
        try:
            from duckduckgo_search import DDGS
        except Exception:
            return [self._uncertainty_item(claim, "free_web_search_unavailable:duckduckgo_search_missing")]
        max_results = int(self.config.get("max_web_results_per_claim", 5))
        query = " ".join([claim.statement, *plan.search_queries]).strip()
        evidence: list[EvidenceItem] = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception as exc:
            return [self._uncertainty_item(claim, f"free_web_search_failed:{exc.__class__.__name__}")]
        for result in results[:max_results]:
            url = result.get("href") or result.get("url")
            if not url:
                continue
            article = None
            try:
                article = self.extractor.extract(url)
            except Exception:
                pass
            title = (article.title if article else result.get("title")) or "Web article candidate"
            body = (article.text if article and article.text else result.get("body", ""))[:1200]
            source_type = _classify_source_type(url, title)
            reliability = _source_reliability(url, title)
            evidence_id = f"web_{stable_hash_text(url + claim.claim_id)}"
            evidence.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_type=source_type,  # type: ignore[arg-type]
                    source=url,
                    title=title,
                    content=body or f"Search result candidate for claim: {claim.statement}",
                    url=url,
                    reliability=reliability,
                    relevance=0.65,
                    metadata={
                        "source_name": article.source_name if article else None,
                        "published_at": article.published_at if article else None,
                        "author": article.author if article else None,
                        "claim_relevance": 0.65,
                        "supports": [claim.claim_type],
                        "contradicts": [],
                        "summary": body[:360] if body else "",
                        "quoted_evidence": body[:220] if body else "",
                    },
                    supports_claim_types=[claim.claim_type],
                    provenance=Provenance(
                        source_id=evidence_id,
                        source_type=source_type,  # type: ignore[arg-type]
                        source=url,
                        url=url,
                        retrieval_method="duckduckgo_search+article_extraction",
                    ),
                )
            )
        return evidence

    @staticmethod
    def _uncertainty_item(claim: SubClaim, flag: str) -> EvidenceItem:
        evidence_id = f"uncertainty_{stable_hash_text(claim.claim_id + flag)}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="synthetic_uncertainty",
            source="free_web_search",
            title="Free web search unavailable",
            content=f"Free web/news search did not run for claim {claim.claim_id} ({flag}).",
            reliability=0.2,
            relevance=0.45,
            uncertainty_flags=[flag],
            supports_claim_types=[claim.claim_type],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="synthetic_uncertainty",
                source="free_web_search",
                retrieval_method="local_capability_check",
                metadata={"adapter": "free_web_search", "flag": flag},
            ),
        )


def _classify_source_type(url: str, title: str) -> str:
    text = f"{url} {title}".lower()
    if any(token in text for token in ("factcheck", "fact-check", "snopes", "politifact")):
        return "factcheck_article"
    if any(token in text for token in ("news", "reuters", "apnews", "bbc", "cnn", "aljazeera")):
        return "news_article"
    return "web_article"


def _source_reliability(url: str, title: str) -> float:
    text = f"{url} {title}".lower()
    if any(token in text for token in ("factcheck", "snopes", "politifact")):
        return 0.80
    if any(token in text for token in (".gov", ".int", ".edu", "who.int", "un.org")):
        return 0.70
    if any(token in text for token in ("reuters", "apnews", "bbc", "associated press")):
        return 0.80
    return 0.60
