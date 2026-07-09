from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import sha256_file, stable_hash_text
from src.utils.io import project_root
from src.utils.tool_config import retrieval_config

logger = logging.getLogger("run_case")


class _YandexAuthFailed(Exception):
    pass


class _YandexInvalidJSON(Exception):
    pass


class YandexReverseImageSearch:
    def __init__(self, config: dict | None = None) -> None:
        self.config = retrieval_config(config)
        self.cache_path = project_root() / self.config.get(
            "yandex_reverse_cache_path", "data/cache/yandex_reverse_cache.json"
        )
        self._last_request_at = 0.0

    def search(
        self,
        claim: SubClaim,
        plan: ResearchPlan,
        query_image_paths: list[Path],
    ) -> list[EvidenceItem]:
        del plan
        if not self.config.get("yandex_reverse_enabled", False):
            logger.debug("Yandex reverse image search disabled, skipping claim %s", claim.claim_id)
            return []
        if not query_image_paths:
            return []

        api_key = os.getenv("SEMV_YANDEX_API_KEY")
        iam_token = os.getenv("SEMV_YANDEX_IAM_TOKEN")
        folder_id = os.getenv("SEMV_YANDEX_FOLDER_ID")
        if not api_key and not iam_token:
            return [self._uncertainty_item(claim, "yandex_reverse_missing_credentials")]
        if not folder_id:
            return [self._uncertainty_item(claim, "yandex_reverse_missing_folder_id")]

        import requests

        family_mode = self.config.get("yandex_reverse_family_mode", "FAMILY_MODE_MODERATE")
        max_images = int(self.config.get("yandex_reverse_max_images_per_claim", 3))
        max_results = int(self.config.get("yandex_reverse_max_results_per_image", 10))
        evidence: list[EvidenceItem] = []
        seen: set[tuple[str, str, str]] = set()

        for query_path in query_image_paths[:max_images]:
            try:
                image_data = _image_path_to_base64_jpeg(query_path)
                payload = {
                    "folderId": folder_id,
                    "data": image_data,
                    "page": "0",
                    "familyMode": family_mode,
                }
                response_json = self._fetch(requests, payload, query_path=query_path, page="0")
            except _YandexAuthFailed:
                return [self._uncertainty_item(claim, "yandex_reverse_auth_failed")]
            except _YandexInvalidJSON:
                return [self._uncertainty_item(claim, "yandex_reverse_failed:InvalidJSON")]
            except Exception as exc:
                logger.warning(
                    "Yandex reverse image search failed for claim=%s path=%s: %s: %s",
                    claim.claim_id,
                    query_path,
                    exc.__class__.__name__,
                    exc,
                )
                return [self._uncertainty_item(claim, f"yandex_reverse_failed:{exc.__class__.__name__}")]

            for rank, result in enumerate((response_json.get("images") or [])[:max_results], start=1):
                page_url = result.get("pageUrl") or ""
                image_url = result.get("url") or ""
                dedupe_key = (page_url, image_url, str(query_path))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                evidence.append(self._item_for_result(claim, query_path, result, rank))

        return evidence

    def _fetch(self, requests_module: Any, payload: dict[str, Any], query_path: Path, page: str) -> dict[str, Any]:
        cache_enabled = bool(self.config.get("yandex_reverse_cache_enabled", True))
        cache_key = self._cache_key(query_path, page)
        cache = self._load_cache() if cache_enabled else {}
        if cache_enabled and cache_key in cache:
            logger.info("Yandex reverse image search cache hit for path=%s key=%s", query_path, cache_key)
            return dict(cache[cache_key].get("response") or {})

        response_json = self._request_with_retry(requests_module, payload)
        if cache_enabled:
            cache[cache_key] = {
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "query_path": str(query_path),
                "family_mode": self.config.get("yandex_reverse_family_mode", "FAMILY_MODE_MODERATE"),
                "page": page,
                "response": response_json,
            }
            self._save_cache(cache)
        return response_json

    def _request_with_retry(self, requests_module: Any, payload: dict[str, Any]) -> dict[str, Any]:
        base_url = self.config.get(
            "yandex_reverse_base_url",
            "https://searchapi.api.cloud.yandex.net/v2/image/search_by_image",
        )
        timeout = float(self.config.get("yandex_reverse_timeout_sec", 20))
        min_interval = float(self.config.get("yandex_reverse_min_interval_sec", 3))
        max_retries = int(self.config.get("yandex_reverse_max_retries", 2))

        headers = self._auth_headers()
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            self._sleep_for_rate_limit(min_interval)
            response = requests_module.post(base_url, headers=headers, json=payload, timeout=timeout)
            if response.status_code in {401, 403}:
                raise _YandexAuthFailed()
            if response.status_code == 429 or 500 <= response.status_code < 600:
                last_error = _http_error(response)
                if attempt < max_retries:
                    retry_after = response.headers.get("Retry-After", "")
                    sleep_sec = _retry_after_seconds(retry_after) or min(2 ** attempt, 8)
                    logger.warning(
                        "Yandex reverse image search returned HTTP %s, retrying in %.1fs (attempt %d/%d)",
                        response.status_code,
                        sleep_sec,
                        attempt + 1,
                        max_retries + 1,
                    )
                    time.sleep(sleep_sec)
                    continue
                raise last_error
            try:
                response.raise_for_status()
            except Exception as exc:
                raise exc
            try:
                return response.json()
            except Exception as exc:
                raise _YandexInvalidJSON() from exc

        assert last_error is not None
        raise last_error

    def _auth_headers(self) -> dict[str, str]:
        api_key = os.getenv("SEMV_YANDEX_API_KEY")
        iam_token = os.getenv("SEMV_YANDEX_IAM_TOKEN")
        token = f"Api-Key {api_key}" if api_key else f"Bearer {iam_token}"
        return {"Authorization": token, "Content-Type": "application/json"}

    def _sleep_for_rate_limit(self, min_interval_sec: float) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_at
        if elapsed < min_interval_sec:
            time.sleep(min_interval_sec - elapsed)
        self._last_request_at = time.monotonic()

    def _cache_key(self, query_path: Path, page: str) -> str:
        try:
            image_key = sha256_file(query_path)
        except Exception:
            stat = query_path.stat()
            image_key = f"{query_path}:{stat.st_mtime_ns}:{stat.st_size}"
        payload = {
            "image": image_key,
            "family_mode": self.config.get("yandex_reverse_family_mode", "FAMILY_MODE_MODERATE"),
            "page": page,
        }
        return stable_hash_text(json.dumps(payload, sort_keys=True), length=32)

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, indent=2, default=str) + "\n", encoding="utf-8")

    def _item_for_result(self, claim: SubClaim, query_path: Path, result: dict[str, Any], rank: int) -> EvidenceItem:
        page_url = result.get("pageUrl") or ""
        image_url = result.get("url") or ""
        page_title = result.get("pageTitle") or "Yandex reverse image candidate"
        host = result.get("host") or ""
        passage = result.get("passage") or ""
        width = _optional_int(result.get("width"))
        height = _optional_int(result.get("height"))
        evidence_id = f"yandex_reverse_{stable_hash_text(page_url + image_url + str(query_path) + claim.claim_id)}"
        content_parts = [
            f"Yandex reverse image candidate for query image {query_path}.",
            f"Host: {host or 'unknown'}.",
            f"Page title: {page_title}.",
        ]
        if passage:
            content_parts.append(f"Passage: {passage}.")
        if image_url:
            content_parts.append(f"Image URL: {image_url}.")
        metadata = {
            "adapter": "yandex_reverse_image_search",
            "query_path": str(query_path),
            "image_url": image_url,
            "page_url": page_url,
            "host": host,
            "page_title": page_title,
            "passage": passage,
            "width": width,
            "height": height,
            "rank": rank,
        }
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="reverse_image_web_candidate",
            source="yandex_reverse_image_search",
            title=page_title,
            url=page_url or None,
            content=" ".join(content_parts),
            reliability=0.70,
            relevance=0.80,
            media_path=str(query_path),
            metadata=metadata,
            raw_output={"result": result},
            supports_claim_types=["what", "where", "when", "authenticity"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="reverse_image_web_candidate",
                source="yandex_reverse_image_search",
                url=page_url or None,
                retrieval_method="yandex_search_api_search_by_image",
                metadata=metadata,
            ),
        )

    @staticmethod
    def _uncertainty_item(claim: SubClaim, flag: str) -> EvidenceItem:
        evidence_id = f"uncertainty_{stable_hash_text(claim.claim_id + flag)}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="synthetic_uncertainty",
            source="yandex_reverse_image_search",
            title="Yandex reverse image search unavailable",
            content=f"Yandex reverse image search did not run for claim {claim.claim_id} ({flag}).",
            reliability=0.2,
            relevance=0.45,
            uncertainty_flags=[flag],
            supports_claim_types=[claim.claim_type],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="synthetic_uncertainty",
                source="yandex_reverse_image_search",
                retrieval_method="local_capability_check",
                metadata={"adapter": "yandex_reverse_image_search", "flag": flag},
            ),
        )


def _image_path_to_base64_jpeg(path: Path, max_bytes: int = 4_000_000) -> str:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        for max_dim in (1600, 1200, 900, 640):
            candidate = image.copy()
            candidate.thumbnail((max_dim, max_dim))
            for quality in (85, 75, 65, 55):
                buffer = BytesIO()
                candidate.save(buffer, format="JPEG", quality=quality, optimize=True)
                data = buffer.getvalue()
                if len(data) <= max_bytes:
                    return base64.b64encode(data).decode("ascii")
        buffer = BytesIO()
        candidate.save(buffer, format="JPEG", quality=45, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _http_error(response: Any) -> Exception:
    try:
        response.raise_for_status()
    except Exception as exc:
        return exc
    return RuntimeError(f"Yandex reverse image search failed with status {response.status_code}")


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(value: str) -> float | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return max(seconds, 0.0)
