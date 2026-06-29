from __future__ import annotations

from src.schemas.case_schema import MediaItem
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text


class OCRExtractor:
    def extract(self, media: MediaItem) -> list[EvidenceItem]:
        evidence_id = f"ocr_{stable_hash_text(media.path)}"
        return [
            EvidenceItem(
                evidence_id=evidence_id,
                source_type="synthetic_uncertainty",
                source=media.path,
                title="OCR adapter unavailable",
                content="OCR was not run because no production OCR adapter is configured.",
                reliability=0.2,
                relevance=0.35,
                uncertainty_flags=["ocr_adapter_unavailable"],
                provenance=Provenance(
                    source_id=evidence_id,
                    source_type="synthetic_uncertainty",
                    source=media.path,
                    retrieval_method="adapter_placeholder",
                ),
            )
        ]
