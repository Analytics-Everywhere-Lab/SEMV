from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from src.schemas.case_schema import MediaItem
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text


class SceneKeyframeExtractor:
    def extract(
        self,
        media: MediaItem,
        output_dir: Path,
        base_dir: Path | None = None,
        max_frames: int = 8,
        deduplicate: bool = True,
        strategy: str = "scene_detect",
    ) -> list[EvidenceItem]:
        if media.media_type != "video":
            return []

        media_path = media.resolved_path(base_dir)
        if not media_path.exists():
            return [self._uncertainty_item(media, "video_keyframes_unavailable:media_file_missing")]

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return [self._uncertainty_item(media, "video_keyframes_unavailable:ffmpeg_missing")]

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamps = []
        method = "ffmpeg_uniform"
        if strategy in {"scene_detect", "hybrid"}:
            timestamps = self._scene_timestamps(media_path, max_frames)
            if timestamps:
                method = "pyscenedetect+ffmpeg"
        if not timestamps:
            timestamps = self._uniform_timestamps(media_path, max_frames)
            method = "ffmpeg_uniform_fallback"
        if not timestamps:
            return [self._uncertainty_item(media, "video_keyframes_unavailable:no_video_duration")]

        frame_paths: list[tuple[Path, float]] = []
        for index, timestamp in enumerate(timestamps[: max(1, max_frames)], start=1):
            frame_path = output_dir / f"{media_path.stem}_scene_{index:03d}.jpg"
            command = [
                ffmpeg,
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(media_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(frame_path),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
                if frame_path.exists():
                    frame_paths.append((frame_path, timestamp))
            except Exception:
                continue

        if deduplicate:
            frame_paths = self._deduplicate(frame_paths)
        if not frame_paths:
            return [self._uncertainty_item(media, "video_keyframes_unavailable:ffmpeg_extract_failed")]

        evidence: list[EvidenceItem] = []
        for frame_path, timestamp in frame_paths[:max_frames]:
            evidence_id = f"scene_keyframe_{stable_hash_text(str(frame_path) + str(timestamp))}"
            evidence.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_type="scene_keyframe",
                    source=str(media_path),
                    title=f"Scene keyframe at {timestamp:.2f}s",
                    content=f"Scene keyframe extracted from {media_path} at {timestamp:.2f}s.",
                    media_path=str(frame_path),
                    timestamp_sec=timestamp,
                    frame_path=str(frame_path),
                    reliability=0.70 if method == "pyscenedetect+ffmpeg" else 0.58,
                    relevance=0.70,
                    metadata={"video_path": str(media_path), "method": method},
                    supports_claim_types=["what", "where", "when", "authenticity"],
                    provenance=Provenance(
                        source_id=evidence_id,
                        source_type="scene_keyframe",
                        source=str(media_path),
                        retrieval_method=method,
                    ),
                )
            )
        return evidence

    @staticmethod
    def _scene_timestamps(video_path: Path, max_frames: int) -> list[float]:
        try:
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import ContentDetector

            video = open_video(str(video_path))
            manager = SceneManager()
            manager.add_detector(ContentDetector())
            manager.detect_scenes(video, show_progress=False)
            scenes = manager.get_scene_list()
            timestamps = []
            for start, end in scenes:
                midpoint = (start.get_seconds() + end.get_seconds()) / 2.0
                timestamps.append(midpoint)
            return timestamps[:max_frames]
        except Exception:
            return []

    @staticmethod
    def _uniform_timestamps(video_path: Path, max_frames: int) -> list[float]:
        ffprobe = shutil.which("ffprobe")
        duration = 0.0
        if ffprobe:
            try:
                result = subprocess.run(
                    [
                        ffprobe,
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(video_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                duration = float((result.stdout or "0").strip() or 0)
            except Exception:
                duration = 0.0
        if duration <= 0:
            return [0.0]
        count = max(1, max_frames)
        step = duration / (count + 1)
        return [step * (index + 1) for index in range(count)]

    @staticmethod
    def _deduplicate(frame_paths: list[tuple[Path, float]]) -> list[tuple[Path, float]]:
        try:
            import imagehash
            from PIL import Image
        except Exception:
            return frame_paths
        kept: list[tuple[Path, float]] = []
        hashes = []
        for frame_path, timestamp in frame_paths:
            try:
                with Image.open(frame_path) as image:
                    phash = imagehash.phash(image)
                if any(abs(phash - old) <= 4 for old in hashes):
                    continue
                hashes.append(phash)
                kept.append((frame_path, timestamp))
            except Exception:
                kept.append((frame_path, timestamp))
        return kept

    @staticmethod
    def _uncertainty_item(media: MediaItem, flag: str) -> EvidenceItem:
        evidence_id = f"uncertainty_{stable_hash_text(media.path + flag)}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="synthetic_uncertainty",
            source=media.path,
            title="Video keyframe extraction unavailable",
            content=f"Scene-aware keyframe extraction for {media.path} was not performed ({flag}).",
            reliability=0.2,
            relevance=0.4,
            uncertainty_flags=[flag],
            supports_claim_types=["what", "where", "when", "authenticity"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="synthetic_uncertainty",
                source=media.path,
                retrieval_method="local_capability_check",
                metadata={"adapter": "scene_keyframe", "flag": flag},
            ),
        )
