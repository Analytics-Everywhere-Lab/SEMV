from __future__ import annotations

from pathlib import Path

from src.processing.asr_extractor import ASRExtractor
from src.processing.keyframe_extractor import KeyframeExtractor
from src.processing.metadata_extractor import MetadataExtractor
from src.processing.ocr_extractor import OCRExtractor
from src.processing.visual_describer import VisualDescriber
from src.schemas.case_schema import MultimediaCase
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.io import project_root


class RawMediaProcessor:
    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or project_root() / "data" / "outputs" / "_media"
        self.metadata_extractor = MetadataExtractor()
        self.keyframe_extractor = KeyframeExtractor()
        self.ocr_extractor = OCRExtractor()
        self.asr_extractor = ASRExtractor()
        self.visual_describer = VisualDescriber()

    def process(self, case: MultimediaCase, case_path: Path | None = None) -> list[EvidenceItem]:
        base_dir = case_path.parent if case_path else project_root()
        evidence: list[EvidenceItem] = []
        for idx, media in enumerate(case.media):
            metadata = self.metadata_extractor.extract(media, base_dir=base_dir)
            metadata_id = f"media_meta_{stable_hash_text(case.case_id + media.path)}"
            evidence.append(
                EvidenceItem(
                    evidence_id=metadata_id,
                    source_type="media_metadata",
                    source=media.path,
                    title="Media metadata",
                    content=f"Local media metadata for {media.path}: {metadata}",
                    reliability=0.65 if metadata.get("exists") else 0.25,
                    relevance=0.55,
                    media_path=media.path,
                    metadata=metadata,
                    uncertainty_flags=metadata.get("uncertainty_flags", []),
                    provenance=Provenance(
                        source_id=metadata_id,
                        source_type="media_metadata",
                        source=media.path,
                        retrieval_method="local_file_inspection",
                    ),
                )
            )
            evidence.append(self.visual_describer.describe(media, metadata))
            evidence.extend(
                self.keyframe_extractor.extract(
                    media,
                    self.output_dir / case.case_id / f"media_{idx}",
                    base_dir=base_dir,
                )
            )
            evidence.extend(self.ocr_extractor.extract(media))
            evidence.extend(self.asr_extractor.extract(media))
        return evidence
