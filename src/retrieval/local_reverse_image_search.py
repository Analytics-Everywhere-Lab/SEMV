from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.retrieval.visual_index import VisualIndex
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.tool_config import media_config


class LocalReverseImageSearch:
    def __init__(self, index: VisualIndex | None = None, config: dict | None = None) -> None:
        self.config = media_config(config)
        self.index = index or VisualIndex(config=config)

    def search(
        self,
        paths: Iterable[str | Path] | None = None,
        case_id: str = "",
        image_paths: Iterable[str | Path] | None = None,
    ) -> list[EvidenceItem]:
        query_paths = [Path(path) for path in (image_paths if image_paths is not None else paths or [])]
        if not self.config.get("enable_local_reverse_search", True):
            return [self._uncertainty_item(path, "local_reverse_search_disabled") for path in query_paths]
        evidence: list[EvidenceItem] = []
        for path in query_paths:
            if not path.exists():
                evidence.append(self._uncertainty_item(path, "local_reverse_query_missing"))
                continue
            try:
                matches = self.index.search(path, exclude_case_id=case_id)
            except Exception as exc:
                evidence.append(self._uncertainty_item(path, f"local_reverse_search_failed:{exc.__class__.__name__}"))
                continue
            for match in matches[:5]:
                evidence.append(self._match_item(path, match))
        return evidence

    def add_assets(
        self,
        paths: Iterable[str | Path] | None = None,
        case_id: str = "",
        image_paths: Iterable[str | Path] | None = None,
    ) -> None:
        if self.config.get("enable_local_reverse_search", True):
            assets = list(image_paths if image_paths is not None else paths or [])
            if assets:
                self.index.add_assets(assets, case_id=case_id)

    @staticmethod
    def reliability_for_match(phash_distance: int | None, clip_similarity: float | None, methods: list[str]) -> float:
        phash_strong = phash_distance is not None and phash_distance <= 4
        clip_strong = clip_similarity is not None and clip_similarity >= 0.90
        clip_weak = clip_similarity is not None and clip_similarity >= 0.0
        if "phash" in methods and "clip_faiss" in methods:
            return 0.90
        if phash_strong:
            return 0.80
        if clip_strong:
            return 0.80
        if clip_weak:
            return 0.65
        return 0.55

    def _match_item(self, query_path: Path, match: dict) -> EvidenceItem:
        methods = sorted(set(match.get("methods") or []))
        distance = match.get("phash_distance")
        similarity = match.get("clip_similarity")
        reliability = self.reliability_for_match(distance, similarity, methods)
        evidence_id = f"reverse_local_{stable_hash_text(str(query_path) + str(match.get('asset_id')))}"
        raw_output = {
            "query_path": str(query_path),
            "matched_path": match.get("path"),
            "matched_case_id": match.get("case_id"),
            "phash_distance": distance,
            "clip_similarity": similarity,
            "methods": methods,
        }
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="reverse_image_local",
            source=str(query_path),
            title="Local visual match found",
            content=(
                f"Query image/keyframe visually matches previous asset {match.get('path')} "
                f"from case {match.get('case_id')} using {', '.join(methods) or 'visual similarity'}."
            ),
            reliability=reliability,
            relevance=0.85,
            media_path=str(query_path),
            raw_output=raw_output,
            metadata={
                "matched_path": match.get("path"),
                "phash_distance": distance,
                "clip_similarity": similarity,
                "matched_case_id": match.get("case_id"),
                "asset_id": match.get("asset_id"),
                "methods": methods,
            },
            supports_claim_types=["what", "where", "when", "authenticity"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="reverse_image_local",
                source=str(query_path),
                retrieval_method="local_visual_index",
                metadata={"matched_asset_id": match.get("asset_id"), "methods": methods},
            ),
        )

    @staticmethod
    def _uncertainty_item(path: Path, flag: str) -> EvidenceItem:
        evidence_id = f"uncertainty_{stable_hash_text(str(path) + flag)}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="synthetic_uncertainty",
            source=str(path),
            title="Local reverse image search unavailable",
            content=f"Local reverse image search did not run for {path} ({flag}).",
            reliability=0.2,
            relevance=0.45,
            media_path=str(path),
            uncertainty_flags=[flag],
            supports_claim_types=["what", "where", "when", "authenticity"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="synthetic_uncertainty",
                source=str(path),
                retrieval_method="local_capability_check",
                metadata={"adapter": "local_reverse_image_search", "flag": flag},
            ),
        )
