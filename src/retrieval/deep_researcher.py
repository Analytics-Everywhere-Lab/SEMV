from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

from src.retrieval.factcheck_search import FactCheckSearch
from src.retrieval.free_web_search import FreeWebSearch
from src.retrieval.gdelt_search import GDELTSearch
from src.retrieval.geolocation_candidate_extractor import GeolocationCandidateExtractor
from src.retrieval.geolocation_search import GeolocationSearch
from src.retrieval.news_search import NewsSearch
from src.retrieval.reverse_search import ReverseSearch
from src.retrieval.web_search import CachedEvidenceSearch
from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.llm_client import LLMClient

logger = logging.getLogger("run_case")


MEDIA_QUERY_SOURCE_TYPES = {
    "ocr",
    "asr",
    "visual_caption",
    "visual_objects",
    "frame_analysis",
    "visual_vqa",
    "metadata_exiftool",
    "metadata_ffprobe",
    "reverse_image_local",
    "reverse_image_web_candidate",
    "geolocation_candidate",
}


class DeepResearcher:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        cached = CachedEvidenceSearch()
        self.free_web_search = FreeWebSearch()
        self.gdelt_search = GDELTSearch()
        logger.info("GDELT search enabled=%s", self.gdelt_search.config.get("gdelt_search_enabled", False))
        self.adapters = [
            cached,
            ReverseSearch(cached),
            FactCheckSearch(cached),
            NewsSearch(cached),
            GeolocationSearch(cached),
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

        derived_queries = build_queries_from_evidence(claim, plan, existing_evidence)
        enriched_plan = plan.model_copy(update={"search_queries": derived_queries})
        for adapter in self.adapters:
            for item in adapter.search(claim, enriched_plan):
                found[item.evidence_id] = item

        for item in self.gdelt_search.search(claim, enriched_plan, queries=derived_queries):
            found[item.evidence_id] = item

        query_images = _query_image_paths(existing_evidence)
        for item in self.free_web_search.search(
            claim,
            enriched_plan,
            queries=derived_queries,
            query_image_paths=query_images,
        ):
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


def build_queries_from_evidence(
    claim: SubClaim,
    plan: ResearchPlan,
    existing_evidence: list[EvidenceItem],
    max_queries: int = 12,
) -> list[str]:
    queries: list[str] = [*plan.search_queries, *claim.search_queries, claim.statement]
    for item in existing_evidence:
        if item.source_type not in MEDIA_QUERY_SOURCE_TYPES:
            continue
        if item.source_type == "ocr":
            queries.extend(_ocr_queries(claim.claim_type, item.content))
        elif item.source_type == "asr":
            queries.extend(_asr_queries(claim.claim_type, item.content))
        elif item.source_type in {"visual_caption", "visual_objects", "frame_analysis", "visual_vqa"}:
            queries.extend(_vlm_queries(claim.claim_type, item))
        elif item.source_type in {"metadata_exiftool", "metadata_ffprobe"}:
            queries.extend(_metadata_queries(claim.claim_type, item))
        elif item.source_type in {"reverse_image_local", "reverse_image_web_candidate"}:
            queries.extend(_reverse_queries(claim.claim_type, item))
        elif item.source_type == "geolocation_candidate":
            name = item.raw_output.get("candidate_name") or item.metadata.get("candidate_name") or item.content
            if name and name.strip():
                queries.append(f"{name.strip()} location")
    return _rank_and_dedupe_queries(claim.claim_type, queries, max_queries)


def _ocr_queries(claim_type: str, content: str) -> list[str]:
    text = _clean_signal_text(content.replace("Visible text:", ""))
    if not text:
        return []
    suffix = {
        "where": "location",
        "when": "date time",
        "who": "organization person",
        "why": "caption claim",
        "authenticity": "original source",
    }.get(claim_type, "event")
    return [f"{text} {suffix}"]


def _asr_queries(claim_type: str, content: str) -> list[str]:
    phrases = _interesting_phrases(content)
    suffix = "date" if claim_type == "when" else "event"
    return [f"{phrase} {suffix}" for phrase in phrases]


def _vlm_queries(claim_type: str, item: EvidenceItem) -> list[str]:
    raw = item.raw_output or {}
    queries = [str(query) for query in raw.get("search_queries", []) if str(query).strip()]
    for key in ("location_clues", "event_clues", "time_clues", "authenticity_clues"):
        values = raw.get(key) or []
        for value in values:
            if not str(value).strip():
                continue
            if claim_type == "where" and key == "location_clues":
                queries.append(f"{value} location")
            elif claim_type == "when" and key == "time_clues":
                queries.append(f"{value} date")
            elif claim_type in {"what", "authenticity"}:
                queries.append(str(value))
    if item.content:
        queries.append(_clean_signal_text(item.content))
    return queries


def _metadata_queries(claim_type: str, item: EvidenceItem) -> list[str]:
    text = str(item.metadata or {})
    queries = []
    gps = re.findall(r"GPS[^,}\]]+", text, flags=re.IGNORECASE)
    dates = re.findall(r"\b\d{4}[:/-]\d{1,2}[:/-]\d{1,2}\b", text)
    if claim_type == "where":
        queries.extend(gps)
    if claim_type == "when":
        queries.extend(dates)
    return queries


def _reverse_queries(claim_type: str, item: EvidenceItem) -> list[str]:
    raw = item.raw_output or {}
    title = raw.get("source_title") or item.title or ""
    matched = raw.get("matched_path") or raw.get("page_url") or item.source
    if claim_type == "authenticity":
        return [f"{title} {matched} old video original source"]
    if claim_type == "when":
        return [f"{title} {matched} first appeared date"]
    return [f"{title} {matched}"]


def _rank_and_dedupe_queries(claim_type: str, queries: Iterable[str], max_queries: int) -> list[str]:
    keywords = {
        "where": ["location", "gps", "city", "street", "bridge", "station"],
        "when": ["date", "time", "old", "first appeared"],
        "who": ["person", "organization", "username"],
        "authenticity": ["original", "source", "old", "fake"],
        "what": ["event", "scene"],
        "why": ["caption", "claim", "narrative"],
    }.get(claim_type, [])
    deduped = []
    seen = set()
    for query in queries:
        cleaned = _clean_signal_text(str(query))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        score = sum(1 for keyword in keywords if keyword in key)
        deduped.append((score, cleaned))
    deduped.sort(key=lambda item: (-item[0], len(item[1])))
    return [query for _, query in deduped[:max_queries]]


def _interesting_phrases(text: str) -> list[str]:
    cleaned = _clean_signal_text(text)
    if not cleaned:
        return []
    matches = re.findall(r"(?:[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,4}|\b\d{4}\b|\b\w+\s+\d{1,2}\b)", text)
    return [_clean_signal_text(match) for match in matches[:4]] or [cleaned[:120]]


def _clean_signal_text(text: str) -> str:
    cleaned = " ".join(str(text).split())
    cleaned = cleaned.strip(" .,:;|-_")
    return cleaned[:180]


def _query_image_paths(evidence: list[EvidenceItem]) -> list[Path]:
    paths = []
    seen = set()
    for item in evidence:
        raw = item.frame_path or item.media_path
        if not raw:
            continue
        path = Path(raw)
        if path.exists() and str(path) not in seen:
            paths.append(path)
            seen.add(str(path))
    return paths
