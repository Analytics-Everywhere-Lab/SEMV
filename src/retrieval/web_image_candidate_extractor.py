from __future__ import annotations

from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

from src.retrieval.local_reverse_image_search import LocalReverseImageSearch
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.tool_config import retrieval_config


class VisualMatcher(Protocol):
    def compare_paths(self, query_path: str | Path, candidate_path: str | Path) -> dict | None:
        ...


class WebImageCandidateExtractor:
    def __init__(self, config: dict | None = None) -> None:
        self.config = retrieval_config(config)
        self.min_width = int(self.config.get("web_image_min_width", 160))
        self.min_height = int(self.config.get("web_image_min_height", 120))

    def extract_image_urls(self, html: str, base_url: str, limit: int | None = None) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        urls: list[str] = []
        for tag in soup.find_all("meta"):
            prop = str(tag.get("property") or tag.get("name") or "").lower()
            if prop in {"og:image", "twitter:image"}:
                content = tag.get("content")
                if content:
                    urls.append(urljoin(base_url, content))
        for image in soup.find_all("img"):
            src = image.get("src") or image.get("data-src") or image.get("data-original")
            if src:
                urls.append(urljoin(base_url, src))
        deduped = []
        seen = set()
        for url in urls:
            if url not in seen:
                deduped.append(url)
                seen.add(url)
            if limit and len(deduped) >= limit:
                break
        return deduped

    def download_candidate_images(
        self,
        image_urls: list[str],
        output_dir: Path,
        max_images: int,
        timeout: int = 10,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for url in image_urls[:max_images]:
            try:
                response = requests.get(url, timeout=timeout, headers={"User-Agent": "SEMV/1.0"})
                response.raise_for_status()
                content = response.content
                if len(content) < 10 * 1024:
                    continue
                suffix = Path(urlparse(url).path).suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    suffix = ".jpg"
                path = output_dir / f"candidate_{stable_hash_text(url)}{suffix}"
                path.write_bytes(content)
                if self._is_tiny_or_unreadable(path):
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    continue
                downloaded.append(path)
            except Exception:
                continue
        return downloaded

    def compare_candidates(
        self,
        query_image_paths: list[Path],
        candidate_image_paths: list[Path],
        visual_index_or_matcher: VisualMatcher,
        page_url: str = "",
        source_title: str | None = None,
        source_date: str | None = None,
        candidate_urls: dict[str, str] | None = None,
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for query_path in query_image_paths:
            for candidate_path in candidate_image_paths:
                try:
                    match = visual_index_or_matcher.compare_paths(query_path, candidate_path)
                except Exception:
                    match = None
                if not match:
                    continue
                evidence.append(
                    self._match_item(
                        query_path=query_path,
                        candidate_path=candidate_path,
                        match=match,
                        page_url=page_url,
                        source_title=source_title,
                        source_date=source_date,
                        candidate_image_url=(candidate_urls or {}).get(str(candidate_path)),
                    )
                )
        return evidence

    def _is_tiny_or_unreadable(self, path: Path) -> bool:
        try:
            with Image.open(path) as image:
                return image.width < self.min_width or image.height < self.min_height
        except Exception:
            return True

    def _match_item(
        self,
        query_path: Path,
        candidate_path: Path,
        match: dict,
        page_url: str,
        source_title: str | None,
        source_date: str | None,
        candidate_image_url: str | None,
    ) -> EvidenceItem:
        distance = match.get("phash_distance")
        similarity = match.get("clip_similarity")
        methods = sorted(set(match.get("methods") or []))
        reliability = _web_match_reliability(page_url, source_title or "", distance, similarity)
        evidence_id = f"reverse_web_{stable_hash_text(str(query_path) + str(candidate_path) + page_url)}"
        raw_output = {
            "page_url": page_url,
            "candidate_image_url": candidate_image_url,
            "candidate_image_path": str(candidate_path),
            "query_image_path": str(query_path),
            "phash_distance": distance,
            "clip_similarity": similarity,
            "source_title": source_title,
            "source_date": source_date,
            "methods": methods,
        }
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="reverse_image_web_candidate",
            source=page_url or str(candidate_path),
            title="Web candidate image visually matches query media",
            content="A candidate image from a retrieved web page visually matches the submitted media/keyframe.",
            url=page_url or None,
            reliability=reliability,
            relevance=0.82,
            media_path=str(query_path),
            raw_output=raw_output,
            metadata=raw_output,
            supports_claim_types=["what", "where", "when", "authenticity"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="reverse_image_web_candidate",
                source=page_url or str(candidate_path),
                url=page_url or None,
                retrieval_method="web_image_candidate_visual_compare",
                metadata={"methods": methods},
            ),
        )


class LocalVisualMatcher:
    def __init__(self, config: dict | None = None) -> None:
        self.index = LocalReverseImageSearch(config=config).index

    def compare_paths(self, query_path: str | Path, candidate_path: str | Path) -> dict | None:
        return self.index.compare_paths(query_path, candidate_path)


def _web_match_reliability(page_url: str, title: str, phash_distance: int | None, clip_similarity: float | None) -> float:
    text = f"{page_url} {title}".lower()
    strong = (phash_distance is not None and phash_distance <= 4) or (clip_similarity is not None and clip_similarity >= 0.90)
    if any(token in text for token in ("factcheck", "fact-check", "snopes", "politifact", "reuters", "apnews", "bbc")) and strong:
        return 0.85
    if any(token in text for token in ("facebook", "twitter", "x.com", "tiktok", "instagram", "youtube")):
        return 0.45
    if strong:
        return 0.70
    return 0.50
