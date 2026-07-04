from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Iterable

from src.schemas.case_schema import MediaItem
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.tool_config import media_config


class OCRExtractor:
    def __init__(self, config: dict | None = None) -> None:
        self.config = media_config(config)
        self._reader = None

    def extract(
        self,
        media_or_frame_paths: MediaItem | Iterable[str | Path],
        case_id: str = "",
    ) -> list[EvidenceItem]:
        paths = self._coerce_paths(media_or_frame_paths)
        if not paths:
            return []
        if not self.config.get("enable_ocr_adapter", True):
            return [self._uncertainty_item(path, "ocr_adapter_disabled") for path in paths]
        if self.config.get("ocr_engine", "easyocr") == "disabled":
            return [self._uncertainty_item(path, "ocr_adapter_disabled") for path in paths]

        try:
            reader = self._get_reader()
        except Exception as exc:
            flag = f"ocr_adapter_unavailable:{exc.__class__.__name__}"
            return [self._uncertainty_item(path, flag) for path in paths]

        groups: dict[str, dict] = {}
        for path in paths:
            if not path.exists():
                groups[f"missing:{path}"] = {
                    "text": "",
                    "paths": [path],
                    "confidence": 0.0,
                    "bbox": None,
                    "flag": "ocr_media_missing",
                    "raw": [],
                }
                continue
            try:
                results = reader.readtext(str(path))
            except Exception as exc:  # pragma: no cover - engine/runtime dependent
                groups[f"failed:{path}"] = {
                    "text": "",
                    "paths": [path],
                    "confidence": 0.0,
                    "bbox": None,
                    "flag": f"ocr_failed:{exc.__class__.__name__}",
                    "raw": [],
                }
                continue
            for bbox, text, confidence in results:
                normalized = self._normalize(text)
                if not normalized:
                    continue
                box = self._bbox_to_xyxy(bbox)
                item = groups.setdefault(
                    normalized,
                    {
                        "text": text.strip(),
                        "paths": [],
                        "confidence": 0.0,
                        "bbox": box,
                        "raw": [],
                    },
                )
                item["paths"].append(path)
                item["confidence"] = max(float(confidence or 0.0), item["confidence"])
                item["raw"].append({"path": str(path), "bbox": bbox, "text": text, "confidence": confidence})

        evidence: list[EvidenceItem] = []
        for key, group in groups.items():
            if group.get("flag"):
                evidence.append(self._uncertainty_item(group["paths"][0], group["flag"]))
                continue
            paths_for_text = group.get("paths") or []
            if not group.get("text") or not paths_for_text:
                continue
            confidence = float(group.get("confidence") or 0.0)
            first_path = paths_for_text[0]
            evidence_id = f"ocr_{stable_hash_text(case_id + key + str(first_path))}"
            evidence.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_type="ocr",
                    source=str(first_path),
                    title="OCR text detected",
                    content=f"Visible text: {group['text']}",
                    confidence=confidence,
                    bbox=group.get("bbox"),
                    media_path=str(first_path),
                    frame_path=str(first_path) if "keyframe" in str(first_path) or "scene" in str(first_path) else None,
                    reliability=min(0.85, 0.45 + confidence * 0.4),
                    relevance=0.70,
                    raw_output={"occurrences": group.get("raw", [])},
                    metadata={"occurrence_count": len(paths_for_text), "paths": [str(path) for path in paths_for_text]},
                    supports_claim_types=["where", "when", "who", "what", "authenticity"],
                    provenance=Provenance(
                        source_id=evidence_id,
                        source_type="ocr",
                        source=str(first_path),
                        retrieval_method="easyocr",
                        metadata={"case_id": case_id},
                    ),
                )
            )
        return evidence

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        import easyocr

        languages = self.config.get("ocr_languages") or ["en"]
        try:
            import torch

            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
        self._reader = easyocr.Reader(languages, gpu=gpu)
        return self._reader

    @staticmethod
    def _coerce_paths(media_or_frame_paths: MediaItem | Iterable[str | Path]) -> list[Path]:
        if isinstance(media_or_frame_paths, MediaItem):
            return [Path(media_or_frame_paths.path)]
        return [Path(path) for path in media_or_frame_paths]

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = text.lower().strip()
        cleaned = lowered.translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _bbox_to_xyxy(bbox) -> list[float] | None:
        try:
            xs = [float(point[0]) for point in bbox]
            ys = [float(point[1]) for point in bbox]
            return [min(xs), min(ys), max(xs), max(ys)]
        except Exception:
            return None

    @staticmethod
    def _uncertainty_item(path: Path, flag: str) -> EvidenceItem:
        evidence_id = f"uncertainty_{stable_hash_text(str(path) + flag)}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="synthetic_uncertainty",
            source=str(path),
            title="OCR unavailable",
            content=f"OCR was not run for {path} ({flag}).",
            reliability=0.2,
            relevance=0.35,
            media_path=str(path),
            uncertainty_flags=[flag],
            supports_claim_types=["where", "when", "who", "what", "authenticity"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="synthetic_uncertainty",
                source=str(path),
                retrieval_method="local_capability_check",
                metadata={"adapter": "ocr", "flag": flag},
            ),
        )
