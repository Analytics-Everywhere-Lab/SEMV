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

    def search(self, paths: Iterable[str | Path], case_id: str = "") -> list[EvidenceItem]:
        query_paths = [Path(path) for path in paths]
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
                distance = match.get("phash_distance")
                similarity = match.get("clip_similarity")
                strong = distance is not None and distance <= max(4, int(self.config.get("phash_threshold", 10)) // 2)
                evidence_id = f"reverse_local_{stable_hash_text(str(path) + str(match.get('asset_id')))}"
                evidence.append(
                    EvidenceItem(
                        evidence_id=evidence_id,
                        source_type="reverse_image_local",
                        source=str(path),
                        title="Local visual match found",
                        content=(
                            f"Query image/keyframe visually matches previous asset {match.get('path')} "
                            f"with pHash distance {distance} and CLIP similarity {similarity}."
                        ),
                        reliability=0.75 if strong else 0.55,
                        relevance=0.85,
                        media_path=str(path),
                        metadata={
                            "matched_path": match.get("path"),
                            "phash_distance": distance,
                            "clip_similarity": similarity,
                            "matched_case_id": match.get("case_id"),
                            "asset_id": match.get("asset_id"),
                        },
                        supports_claim_types=["what", "where", "when", "authenticity"],
                        provenance=Provenance(
                            source_id=evidence_id,
                            source_type="reverse_image_local",
                            source=str(path),
                            retrieval_method="imagehash_phash_local_index",
                            metadata={"matched_asset_id": match.get("asset_id")},
                        ),
                    )
                )
        return evidence

    def add_assets(self, paths: Iterable[str | Path], case_id: str) -> None:
        if self.config.get("enable_local_reverse_search", True):
            self.index.add_assets(list(paths), case_id=case_id)

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
