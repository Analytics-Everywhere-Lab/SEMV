from __future__ import annotations

from pathlib import Path

from src.processing.vlm_visual_analyzer import VLMVisualAnalyzer
from src.schemas.case_schema import MediaItem
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.llm_client import LLMClient


class VisualDescriber:
    """Backward-compatible wrapper around the real VLM analyzer.

    Existing callers can still ask for a single metadata-derived description; the
    media pipeline now uses VLMVisualAnalyzer directly for grounded frame evidence.
    """

    def __init__(self, config: dict | None = None, llm_client: LLMClient | None = None) -> None:
        self.vlm = VLMVisualAnalyzer(config, llm_client=llm_client)

    def describe(self, media: MediaItem, metadata: dict) -> EvidenceItem:
        path = Path(metadata.get("path", media.path))
        items = self.vlm.analyze([path]) if path.exists() else []
        if items and items[0].source_type != "synthetic_uncertainty":
            return items[0]

        parts = [f"Media path: {metadata.get('path', media.path)}."]
        if media.description:
            parts.append(f"Provided description: {media.description}.")
        if metadata.get("width") and metadata.get("height"):
            parts.append(f"Image dimensions: {metadata['width']}x{metadata['height']}.")
        if metadata.get("mime_type"):
            parts.append(f"MIME type: {metadata['mime_type']}.")
        if metadata.get("sha256"):
            parts.append(f"SHA256: {metadata['sha256']}.")

        evidence_id = f"visual_{stable_hash_text(media.path + ''.join(parts))}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="visual_description",
            source=media.path,
            title="Media visual description",
            content=" ".join(parts),
            media_path=media.path,
            reliability=0.55 if metadata.get("exists") else 0.25,
            relevance=0.65,
            metadata=metadata,
            uncertainty_flags=metadata.get("uncertainty_flags", []),
            provenance=Provenance(
                source_id=evidence_id,
                source_type="visual_description",
                source=media.path,
                retrieval_method="local_metadata_description",
            ),
        )
