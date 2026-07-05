from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.llm_client import VLLMOpenAIClient
from src.utils.tool_config import media_config


VLM_PROMPT = """Return strict JSON about this image/frame with keys: scene_summary, main_objects, visible_people_or_groups, visible_text, location_clues, time_clues, event_clues, authenticity_clues, search_queries, uncertainty_flags. Treat this as observational evidence only; do not identify private people from faces."""


class VLMVisualAnalyzer:
    def __init__(self, config: dict | None = None) -> None:
        self.config = media_config(config)
        self.llm_client = VLLMOpenAIClient()

    def analyze(
        self,
        image_paths: Iterable[str | Path] | None = None,
        claim: str = "",
        context: str | None = None,
        case_id: str = "",
    ) -> list[EvidenceItem]:
        del case_id
        paths = [Path(path) for path in image_paths or []]
        if not paths:
            return []
        if not self.config.get("enable_vlm_adapter", True):
            return [self._uncertainty_item(path, "vlm_adapter_disabled") for path in paths]
        provider = self.config.get("vlm_provider", "vllm")
        if provider == "disabled":
            return [self._uncertainty_item(path, "vlm_adapter_disabled") for path in paths]
        if provider != "vllm":
            return [self._uncertainty_item(path, f"vlm_provider_unavailable:{provider}") for path in paths]

        evidence: list[EvidenceItem] = []
        for path in paths:
            if not path.exists():
                evidence.append(self._uncertainty_item(path, "vlm_media_missing"))
                continue
            try:
                data = self._vllm_generate(path, claim, context)
            except Exception as exc:  # pragma: no cover - local vLLM availability varies
                evidence.append(self._uncertainty_item(path, f"vlm_adapter_failed:{exc.__class__.__name__}"))
                continue
            evidence.extend(self._items_for_analysis(path, data))
        return evidence

    def _vllm_generate(self, path: Path, claim: str, context: str | None) -> dict:
        prompt = VLM_PROMPT
        if claim:
            prompt += f"\nVerification claim: {claim}"
        if context:
            prompt += f"\nCase context: {context}"

        kwargs: dict[str, object] = {
            "timeout": float(self.config.get("vlm_timeout_sec", 120)),
            "format": {"type": "json_object"},
        }

        vlm_model = self.config.get("vlm_model")
        if vlm_model:
            kwargs["model"] = str(vlm_model)

        response_text = self.llm_client.generate_with_images(
            prompt=prompt,
            image_paths=[path],
            system="Return strict JSON only. Do not include markdown.",
            **kwargs,
        )
        return _parse_json(response_text)

    def _items_for_analysis(self, path: Path, data: dict) -> list[EvidenceItem]:
        uncertainty_flags = [str(flag) for flag in data.get("uncertainty_flags", [])]
        reliability = 0.45 if uncertainty_flags else 0.65
        source = str(path)
        items: list[EvidenceItem] = []
        caption = str(data.get("scene_summary") or "").strip()
        if caption:
            items.append(
                self._item(
                    path,
                    "visual_caption",
                    "VLM scene caption",
                    caption,
                    reliability,
                    data,
                    uncertainty_flags,
                )
            )
        objects = data.get("main_objects") or []
        if objects:
            labels = []
            for obj in objects:
                if isinstance(obj, dict):
                    labels.append(str(obj.get("label", "object")))
                else:
                    labels.append(str(obj))
            items.append(
                self._item(
                    path,
                    "visual_objects",
                    "VLM object observations",
                    "Visible objects: " + ", ".join(labels),
                    reliability,
                    data,
                    uncertainty_flags,
                    metadata={"objects": objects},
                )
            )
        clue_bits = []
        for key in ("location_clues", "time_clues", "event_clues", "authenticity_clues"):
            values = data.get(key) or []
            if values:
                clue_bits.append(f"{key}: {', '.join(map(str, values))}")
        if clue_bits:
            items.append(
                self._item(
                    path,
                    "frame_analysis",
                    "VLM frame analysis",
                    "; ".join(clue_bits),
                    reliability,
                    data,
                    uncertainty_flags,
                )
            )
        qbits = []
        if data.get("visible_text"):
            qbits.append("visible_text: " + ", ".join(map(str, data.get("visible_text", []))))
        if data.get("search_queries"):
            qbits.append("search_queries: " + ", ".join(map(str, data.get("search_queries", []))))
        if qbits:
            items.append(
                self._item(
                    path,
                    "visual_vqa",
                    "VLM visual question-answer cues",
                    "; ".join(qbits),
                    reliability,
                    data,
                    uncertainty_flags,
                )
            )
        if not items:
            items.append(self._uncertainty_item(path, "vlm_empty_output"))
        for item in items:
            item.source = source
        return items

    @staticmethod
    def _item(
        path: Path,
        source_type: str,
        title: str,
        content: str,
        reliability: float,
        raw: dict,
        flags: list[str],
        metadata: dict | None = None,
    ) -> EvidenceItem:
        evidence_id = f"{source_type}_{stable_hash_text(str(path) + content)}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type=source_type,  # type: ignore[arg-type]
            source=str(path),
            title=title,
            content=content,
            media_path=str(path),
            frame_path=str(path) if "scene" in str(path) or "keyframe" in str(path) else None,
            reliability=reliability,
            relevance=0.72,
            raw_output=raw,
            metadata=metadata or {},
            uncertainty_flags=flags,
            supports_claim_types=["what", "where", "when", "who", "authenticity"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type=source_type,  # type: ignore[arg-type]
                source=str(path),
                retrieval_method="vllm_multimodal",
            ),
        )

    @staticmethod
    def _uncertainty_item(path: Path, flag: str) -> EvidenceItem:
        evidence_id = f"uncertainty_{stable_hash_text(str(path) + flag)}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="synthetic_uncertainty",
            source=str(path),
            title="VLM visual analysis unavailable",
            content=f"VLM visual analysis was not run for {path} ({flag}).",
            reliability=0.2,
            relevance=0.40,
            media_path=str(path),
            uncertainty_flags=[flag],
            supports_claim_types=["what", "where", "when", "who", "authenticity"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="synthetic_uncertainty",
                source=str(path),
                retrieval_method="local_capability_check",
                metadata={"adapter": "vlm", "flag": flag},
            ),
        )


def _parse_json(text: str) -> dict:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else {}
        raise
