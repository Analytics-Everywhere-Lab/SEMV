from __future__ import annotations

from pathlib import Path

from src.processing.asr_extractor import ASRExtractor
from src.processing.forensic_analyzer import ForensicAnalyzer
from src.processing.metadata_extractor import MetadataExtractor
from src.processing.ocr_extractor import OCRExtractor
from src.processing.scene_keyframe_extractor import SceneKeyframeExtractor
from src.processing.vlm_visual_analyzer import VLMVisualAnalyzer
from src.retrieval.local_reverse_image_search import LocalReverseImageSearch
from src.schemas.case_schema import MediaItem, MultimediaCase
from src.schemas.evidence_schema import EvidenceItem
from src.utils.io import project_root
from src.utils.tool_config import load_tools_config, media_config


class RawMediaProcessor:
    def __init__(self, output_dir: Path | None = None, config: dict | None = None) -> None:
        self.output_dir = output_dir or project_root() / "data" / "outputs" / "_media"
        self.config = config or load_tools_config()
        self.media_config = media_config(self.config)
        self.metadata_extractor = MetadataExtractor()
        self.scene_keyframe_extractor = SceneKeyframeExtractor()
        self.ocr_extractor = OCRExtractor(self.config)
        self.vlm_visual_analyzer = VLMVisualAnalyzer(self.config)
        self.forensic_analyzer = ForensicAnalyzer(self.config)
        self.asr_extractor = ASRExtractor(self.config)
        self.local_reverse_search = LocalReverseImageSearch(config=self.config)

    def process(self, case: MultimediaCase, case_path: Path | None = None) -> list[EvidenceItem]:
        base_dir = case_path.parent if case_path else project_root()
        evidence: list[EvidenceItem] = []
        for idx, media in enumerate(case.media):
            media_output_dir = self.output_dir / case.case_id / f"media_{idx}"

            metadata_items = self.metadata_extractor.extract(media, base_dir=base_dir)
            evidence.extend(metadata_items)

            keyframe_items: list[EvidenceItem] = []
            if media.media_type == "video" and self.media_config.get("enable_ffmpeg_keyframes", True):
                keyframe_items = self.scene_keyframe_extractor.extract(
                    media=media,
                    output_dir=media_output_dir / "keyframes",
                    base_dir=base_dir,
                    max_frames=int(self.media_config.get("max_keyframes_per_video", 8)),
                    strategy=str(self.media_config.get("keyframe_strategy", "scene_detect")),
                    deduplicate=bool(self.media_config.get("deduplicate_keyframes", True)),
                )
                evidence.extend(keyframe_items)

            visual_targets: list[Path] = []
            if media.media_type == "image":
                visual_targets.append(resolve_media_path(media, base_dir))

            for item in keyframe_items:
                if item.frame_path:
                    visual_targets.append(Path(item.frame_path))

            visual_targets = [path for path in visual_targets if path.exists()]

            if self.media_config.get("enable_ocr_adapter", True):
                evidence.extend(
                    self.ocr_extractor.extract(
                        image_paths=visual_targets,
                        case_id=case.case_id,
                    )
                )

            if self.media_config.get("enable_vlm_adapter", True):
                evidence.extend(
                    self.vlm_visual_analyzer.analyze(
                        image_paths=visual_targets,
                        claim=case.claim,
                        context=case.context,
                        case_id=case.case_id,
                    )
                )

            if self.media_config.get("enable_forensic_adapter", True):
                evidence.extend(
                    self.forensic_analyzer.analyze(
                        media=media,
                        visual_targets=visual_targets,
                        output_dir=media_output_dir / "forensics",
                        base_dir=base_dir,
                        metadata_items=metadata_items,
                    )
                )

            if media.media_type == "video" and self.media_config.get("enable_asr_adapter", True):
                evidence.extend(
                    self.asr_extractor.extract(
                        media=media,
                        output_dir=media_output_dir / "asr",
                        base_dir=base_dir,
                    )
                )

            if self.media_config.get("enable_local_reverse_search", True):
                reverse_items = self.local_reverse_search.search(
                    image_paths=visual_targets,
                    case_id=case.case_id,
                )
                evidence.extend(reverse_items)
                self.local_reverse_search.add_assets(
                    image_paths=visual_targets,
                    case_id=case.case_id,
                )
        return evidence


def resolve_media_path(media: MediaItem, base_dir: Path | None = None) -> Path:
    return media.resolved_path(base_dir)
