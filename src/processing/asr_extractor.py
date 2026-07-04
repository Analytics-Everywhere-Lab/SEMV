from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from src.schemas.case_schema import MediaItem
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.tool_config import media_config


class ASRExtractor:
    def __init__(self, config: dict | None = None) -> None:
        self.config = media_config(config)
        self._model = None

    def extract(
        self,
        media: MediaItem,
        output_dir: Path | None = None,
        base_dir: Path | None = None,
    ) -> list[EvidenceItem]:
        if media.media_type != "video":
            return []
        if not self.config.get("enable_asr_adapter", True):
            return [self._uncertainty_item(media, "asr_adapter_disabled")]
        if self.config.get("asr_engine", "faster_whisper") == "disabled":
            return [self._uncertainty_item(media, "asr_adapter_disabled")]

        media_path = media.resolved_path(base_dir)
        if not media_path.exists():
            return [self._uncertainty_item(media, "asr_media_missing")]
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return [self._uncertainty_item(media, "ffmpeg_audio_extract_failed:ffmpeg_missing")]

        output_dir = output_dir or media_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / f"{media_path.stem}_audio_16k.wav"
        command = [ffmpeg, "-y", "-i", str(media_path), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        except Exception as exc:
            return [self._uncertainty_item(media, f"ffmpeg_audio_extract_failed:{exc.__class__.__name__}")]
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return [self._uncertainty_item(media, "no_audio_stream")]

        try:
            model = self._get_model()
            language = self.config.get("asr_language")
            segments, info = model.transcribe(str(audio_path), word_timestamps=True, language=language)
            segment_list = list(segments)
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            return [self._uncertainty_item(media, f"asr_adapter_unavailable:{exc.__class__.__name__}")]

        texts = [getattr(segment, "text", "").strip() for segment in segment_list if getattr(segment, "text", "").strip()]
        if not texts:
            return [self._uncertainty_item(media, "asr_empty_transcript")]
        transcript = " ".join(texts)
        language = getattr(info, "language", None)
        confidence = float(getattr(info, "language_probability", 0.0) or 0.0)
        flags = ["asr_low_confidence"] if confidence and confidence < 0.5 else []

        evidence_id = f"asr_{stable_hash_text(str(media_path) + transcript)}"
        evidence = [
            EvidenceItem(
                evidence_id=evidence_id,
                source_type="asr",
                source=str(media_path),
                title="Video speech transcript",
                content=transcript,
                language=language,
                confidence=confidence or None,
                reliability=0.70 if not flags else 0.50,
                relevance=0.75,
                media_path=str(media_path),
                metadata={"audio_path": str(audio_path), "segment_count": len(segment_list)},
                raw_output={"language": language, "language_probability": confidence},
                uncertainty_flags=flags,
                supports_claim_types=["where", "when", "who", "what", "why", "authenticity"],
                provenance=Provenance(
                    source_id=evidence_id,
                    source_type="asr",
                    source=str(media_path),
                    retrieval_method="faster_whisper",
                    metadata={"audio_path": str(audio_path)},
                ),
            )
        ]
        for index, segment in enumerate(segment_list[:12], start=1):
            text = getattr(segment, "text", "").strip()
            if not text:
                continue
            start = float(getattr(segment, "start", 0.0) or 0.0)
            end = float(getattr(segment, "end", start) or start)
            segment_id = f"asr_segment_{stable_hash_text(str(media_path) + str(index) + text)}"
            evidence.append(
                EvidenceItem(
                    evidence_id=segment_id,
                    source_type="asr",
                    source=str(media_path),
                    title=f"Speech segment at {start:.2f}s",
                    content=text,
                    timestamp_sec=start,
                    language=language,
                    confidence=confidence or None,
                    reliability=0.68 if not flags else 0.48,
                    relevance=0.70,
                    media_path=str(media_path),
                    metadata={"start": start, "end": end, "text": text},
                    supports_claim_types=["where", "when", "who", "what", "why"],
                    provenance=Provenance(
                        source_id=segment_id,
                        source_type="asr",
                        source=str(media_path),
                        retrieval_method="faster_whisper_segment",
                    ),
                )
            )
        return evidence

    def _get_model(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        try:
            import torch

            use_cuda = bool(torch.cuda.is_available())
        except Exception:
            use_cuda = False
        self._model = WhisperModel(
            self.config.get("asr_model_size", "base"),
            device="cuda" if use_cuda else "cpu",
            compute_type="float16" if use_cuda else "int8",
        )
        return self._model

    @staticmethod
    def _uncertainty_item(media: MediaItem, flag: str) -> EvidenceItem:
        evidence_id = f"uncertainty_{stable_hash_text(media.path + flag)}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="synthetic_uncertainty",
            source=media.path,
            title="ASR unavailable",
            content=f"Audio transcription was not run for {media.path} ({flag}).",
            reliability=0.2,
            relevance=0.35,
            media_path=media.path,
            uncertainty_flags=[flag],
            supports_claim_types=["where", "when", "who", "what", "why", "authenticity"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="synthetic_uncertainty",
                source=media.path,
                retrieval_method="local_capability_check",
                metadata={"adapter": "asr", "flag": flag},
            ),
        )
