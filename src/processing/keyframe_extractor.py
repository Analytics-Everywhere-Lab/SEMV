from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from src.schemas.case_schema import MediaItem
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text


class KeyframeExtractor:
    def extract(
        self,
        media: MediaItem,
        output_dir: Path,
        base_dir: Path | None = None,
        max_frames: int = 3,
    ) -> list[EvidenceItem]:
        if media.media_type != "video":
            return []

        media_path = media.resolved_path(base_dir)
        if not media_path.exists():
            return [
                self._uncertainty_item(
                    media,
                    "video_keyframes_unavailable:media_file_missing",
                )
            ]

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return [self._uncertainty_item(media, "video_keyframes_unavailable:ffmpeg_missing")]

        output_dir.mkdir(parents=True, exist_ok=True)
        pattern = output_dir / f"{media_path.stem}_frame_%03d.jpg"
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(media_path),
            "-vf",
            f"fps=1/{max(1, max_frames)}",
            "-frames:v",
            str(max_frames),
            str(pattern),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        except Exception as exc:  # pragma: no cover - ffmpeg availability varies
            return [
                self._uncertainty_item(
                    media,
                    f"video_keyframes_unavailable:{exc.__class__.__name__}",
                )
            ]

        evidence = []
        for frame in sorted(output_dir.glob(f"{media_path.stem}_frame_*.jpg")):
            evidence_id = f"keyframe_{stable_hash_text(str(frame))}"
            evidence.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_type="keyframe",
                    source=str(media_path),
                    title=f"Extracted keyframe {frame.name}",
                    content=f"Keyframe extracted from video media at {frame}.",
                    media_path=str(frame),
                    reliability=0.55,
                    relevance=0.6,
                    provenance=Provenance(
                        source_id=evidence_id,
                        source_type="keyframe",
                        source=str(media_path),
                        retrieval_method="ffmpeg",
                    ),
                )
            )
        return evidence

    @staticmethod
    def _uncertainty_item(media: MediaItem, flag: str) -> EvidenceItem:
        evidence_id = f"uncertainty_{stable_hash_text(media.path + flag)}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="synthetic_uncertainty",
            source=media.path,
            title="Video keyframe extraction unavailable",
            content=f"Keyframe extraction for {media.path} was not performed.",
            reliability=0.2,
            relevance=0.4,
            uncertainty_flags=[flag],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="synthetic_uncertainty",
                source=media.path,
                retrieval_method="local_capability_check",
            ),
        )
