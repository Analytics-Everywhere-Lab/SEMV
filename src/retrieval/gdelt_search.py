from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.retrieval.web_article_extractor import WebArticleExtractor
from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.io import project_root
from src.utils.tool_config import retrieval_config

logger = logging.getLogger("run_case")


_TRUSTED_NEWS_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "aljazeera.com",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "cnn.com",
    "npr.org",
    "cbc.ca",
}

GENERIC_GDELT_QUERIES = {
    "location",
    "where",
    "when",
    "who",
    "what",
    "why",
    "how",
    "image",
    "video",
    "photo",
    "claim",
    "evidence",
    "news",
    "context",
    "source",
    "event",
}

def _is_good_gdelt_query(query: str) -> bool:
    cleaned = " ".join(str(query).lower().split())
    if not cleaned:
        return False
    if cleaned in GENERIC_GDELT_QUERIES:
        return False

    tokens = [token for token in cleaned.replace('"', "").split() if len(token) > 2]
    if len(tokens) < 2:
        return False

    has_digit = any(char.isdigit() for char in cleaned)
    has_named_entity_shape = any(token[:1].isupper() for token in str(query).split())
    has_long_token = any(len(token) >= 6 for token in tokens)

    return has_digit or has_named_entity_shape or has_long_token


class GDELTSearch:
    def __init__(self, config: dict | None = None) -> None:
        self.config = retrieval_config(config)
        self.extractor = WebArticleExtractor()
        self._cache_path = project_root() / self.config.get(
            "gdelt_cache_path", "data/cache/gdelt_search_cache.json"
        )
        self._rate_limited_until = 0.0
        self._last_request_ts = 0.0

    def _sleep_for_rate_limit(self, min_interval_sec: float) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if elapsed < min_interval_sec:
            time.sleep(min_interval_sec - elapsed)
        self._last_request_ts = time.monotonic()

    def search(
        self,
        claim: SubClaim,
        plan: ResearchPlan,
        queries: list[str] | None = None,
    ) -> list[EvidenceItem]:
        if not self.config.get("gdelt_search_enabled", False):
            logger.debug("GDELT search disabled, skipping claim %s", claim.claim_id)
            return []

        if time.monotonic() < self._rate_limited_until:
            logger.warning(
                "GDELT circuit breaker open, skipping claim=%s (cooldown for %.0fs more)",
                claim.claim_id,
                self._rate_limited_until - time.monotonic(),
            )
            return []

        import requests

        max_queries = int(self.config.get("gdelt_max_queries_per_claim", 4))
        source_lang = self.config.get("gdelt_source_lang") or None
        candidate_queries = _dedupe_queries(
            queries or [" ".join([claim.statement, *plan.search_queries]).strip()]
        )
        search_queries = [query for query in candidate_queries if _is_good_gdelt_query(query)][:max_queries]
        skipped = len(candidate_queries) - len(search_queries)
        if skipped > 0:
            logger.info(
                "GDELT filtered out %d low-quality/generic query candidate(s) for claim=%s",
                skipped,
                claim.claim_id,
            )

        if not search_queries:
            logger.info(
                "GDELT: no specific query could be built for claim=%s (type=%s), skipping",
                claim.claim_id,
                claim.claim_type,
            )
            return []

        logger.info(
            "GDELT search: claim=%s type=%s queries=%d",
            claim.claim_id,
            claim.claim_type,
            len(search_queries),
        )

        evidence: list[EvidenceItem] = []
        seen_urls: set[str] = set()
        max_records = int(self.config.get("gdelt_max_records_per_claim", 8))
        fetch_full = bool(self.config.get("gdelt_fetch_full_articles", True))
        min_relevance = float(self.config.get("gdelt_min_relevance", 0.0))

        for query in search_queries:
            gdelt_query = _gdelt_query(query, source_lang)
            params = {
                "query": gdelt_query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": max_records,
                "sort": self.config.get("gdelt_sort", "hybridrel"),
            }
            timespan = self.config.get("gdelt_timespan")
            if timespan:
                params["timespan"] = timespan

            try:
                response_json = self._fetch(requests, params)
            except Exception as exc:
                logger.warning(
                    "GDELT request failed for claim=%s query=%r: %s: %s",
                    claim.claim_id,
                    query,
                    exc.__class__.__name__,
                    exc,
                )
                return [self._uncertainty_item(claim, f"gdelt_search_failed:{exc.__class__.__name__}")]

            if response_json is None:
                logger.warning("GDELT returned invalid JSON for claim=%s query=%r", claim.claim_id, query)
                return [self._uncertainty_item(claim, "gdelt_search_failed:InvalidJSON")]

            articles = response_json.get("articles", []) or []
            logger.info(
                "GDELT query=%r returned %d article(s) for claim=%s",
                query,
                len(articles),
                claim.claim_id,
            )
            for article in articles:
                url = article.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                item = self._item_for_article(claim=claim, query=query, article=article, fetch_full=fetch_full)
                if item.relevance < min_relevance:
                    continue
                evidence.append(item)
                if len(evidence) >= max_records:
                    break

        logger.info(
            "GDELT search done: claim=%s -> %d evidence item(s)",
            claim.claim_id,
            len(evidence),
        )
        return evidence

    def _fetch(self, requests_module: Any, params: dict[str, Any]) -> dict[str, Any] | None:
        cache_enabled = bool(self.config.get("gdelt_cache_enabled", True))
        cache_key = stable_hash_text(json.dumps(params, sort_keys=True))
        cache = self._load_cache() if cache_enabled else {}

        if cache_enabled and cache_key in cache:
            logger.info("GDELT cache hit for query=%r (key=%s)", params.get("query"), cache_key)
            return cache[cache_key].get("response")

        try:
            response_json = self._request_with_retry(requests_module, params)
        except Exception as exc:
            if cache_enabled and cache_key in cache:
                logger.warning(
                    "GDELT live request failed (%s: %s), falling back to stale cache for query=%r",
                    exc.__class__.__name__,
                    exc,
                    params.get("query"),
                )
                stale = dict(cache[cache_key].get("response") or {})
                return stale
            raise

        logger.info(
            "GDELT live request succeeded: query=%r -> %d article(s)",
            params.get("query"),
            len(response_json.get("articles", []) or []),
        )

        if cache_enabled:
            cache[cache_key] = {
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "params": params,
                "response": response_json,
            }
            self._save_cache(cache)

        return response_json

    def _request_with_retry(self, requests_module: Any, params: dict[str, Any]) -> dict[str, Any]:
        base_url = self.config.get("gdelt_base_url", "https://api.gdeltproject.org/api/v2/doc/doc")
        timeout = float(self.config.get("gdelt_timeout_sec", 15))
        min_interval = float(self.config.get("gdelt_min_interval_sec", 12))
        max_retries = int(self.config.get("gdelt_max_retries", 3))
        base_sleep = float(self.config.get("gdelt_backoff_base_sec", 20))
        cooldown_sec = float(self.config.get("gdelt_circuit_breaker_cooldown_sec", 300))

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            self._sleep_for_rate_limit(min_interval)
            logger.info(
                "GDELT live request (attempt %d/%d): query=%r params=%s",
                attempt + 1,
                max_retries + 1,
                params.get("query"),
                params,
            )
            response = requests_module.get(
                base_url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "SEMV/1.0 academic multimedia verification research"},
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_sec = float(retry_after) if retry_after else base_sleep * (2**attempt) + random.uniform(0, 3)
                last_error = _http_error(response)
                if attempt < max_retries:
                    logger.warning(
                        "GDELT rate-limited (429) for query=%r, retrying in %.1fs (attempt %d/%d)",
                        params.get("query"),
                        sleep_sec,
                        attempt + 1,
                        max_retries + 1,
                    )
                    time.sleep(sleep_sec)
                    continue
                self._rate_limited_until = time.monotonic() + cooldown_sec
                logger.warning(
                    "GDELT still rate-limited after %d retries; opening circuit breaker for %.0fs",
                    max_retries,
                    cooldown_sec,
                )
                raise last_error

            try:
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    sleep_sec = base_sleep * (2**attempt) + random.uniform(0, 3)
                    logger.warning(
                        "GDELT request error (%s: %s) for query=%r, retrying in %.1fs (attempt %d/%d)",
                        exc.__class__.__name__,
                        exc,
                        params.get("query"),
                        sleep_sec,
                        attempt + 1,
                        max_retries + 1,
                    )
                    time.sleep(sleep_sec)
                    continue
                raise

        assert last_error is not None
        raise last_error

    def _load_cache(self) -> dict[str, Any]:
        if not self._cache_path.exists():
            return {}
        try:
            return json.loads(self._cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(cache, indent=2, default=str) + "\n", encoding="utf-8")

    def _item_for_article(
        self,
        claim: SubClaim,
        query: str,
        article: dict[str, Any],
        fetch_full: bool,
    ) -> EvidenceItem:
        url = article.get("url", "")
        title = article.get("title") or "GDELT news article"
        domain = article.get("domain", "")
        language = article.get("language", "")
        source_country = article.get("sourcecountry", "")
        seen_date = article.get("seendate", "")
        social_image = article.get("socialimage", "")

        body = ""
        if fetch_full and url:
            try:
                full_article = self.extractor.extract(url)
                body = full_article.text or ""
            except Exception:
                body = ""
        content = body[:1200] if body else title

        relevance = _estimate_relevance(query, title, body)
        reliability = _source_reliability(domain)
        evidence_id = f"gdelt_{stable_hash_text(url + claim.claim_id)}"

        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="news_article",
            source=url,
            title=title,
            content=content,
            url=url,
            reliability=reliability,
            relevance=relevance,
            language=language,
            metadata={
                "adapter": "gdelt",
                "query": query,
                "domain": domain,
                "language": language,
                "source_country": source_country,
                "seen_date": seen_date,
                "social_image": social_image,
                "claim_relevance": relevance,
                "supports": [claim.claim_type],
                "contradicts": [],
                "summary": body[:360] if body else "",
                "quoted_evidence": body[:220] if body else "",
            },
            supports_claim_types=[claim.claim_type],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="news_article",
                source=url,
                url=url,
                retrieval_method="gdelt_doc_api_artlist",
                metadata={
                    "adapter": "gdelt",
                    "query": query,
                    "seen_date": seen_date,
                    "domain": domain,
                },
            ),
        )

    @staticmethod
    def _uncertainty_item(claim: SubClaim, flag: str) -> EvidenceItem:
        evidence_id = f"uncertainty_{stable_hash_text(claim.claim_id + flag)}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="synthetic_uncertainty",
            source="gdelt_search",
            title="GDELT news search unavailable",
            content=f"GDELT news search did not run for claim {claim.claim_id} ({flag}).",
            reliability=0.2,
            relevance=0.45,
            uncertainty_flags=[flag],
            supports_claim_types=[claim.claim_type],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="synthetic_uncertainty",
                source="gdelt_search",
                retrieval_method="local_capability_check",
                metadata={"adapter": "gdelt", "flag": flag},
            ),
        )


def _http_error(response: Any) -> Exception:
    try:
        response.raise_for_status()
    except Exception as exc:
        return exc
    return RuntimeError(f"GDELT request failed with status {response.status_code}")


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for query in queries:
        cleaned = _clean_query(query)
        key = cleaned.lower()
        if cleaned and key not in seen:
            deduped.append(cleaned)
            seen.add(key)
    return deduped


def _clean_query(query: str) -> str:
    cleaned = " ".join(str(query).split())
    return cleaned[:220].strip()


def _gdelt_query(query: str, source_lang: str | None) -> str:
    cleaned = _clean_query(query)
    if source_lang:
        return f"({cleaned}) sourcelang:{source_lang}"
    return cleaned


def _estimate_relevance(query: str, title: str, body: str) -> float:
    text = f"{title} {body}".lower()
    terms = {token for token in query.lower().split() if len(token) > 3}
    if not terms:
        return 0.68
    overlap = sum(1 for term in terms if term in text)
    if overlap == 0:
        return 0.55
    return 0.68


def _source_reliability(domain: str) -> float:
    domain = (domain or "").lower()
    if domain in _TRUSTED_NEWS_DOMAINS or any(domain.endswith("." + trusted) for trusted in _TRUSTED_NEWS_DOMAINS):
        return 0.80
    return 0.65
