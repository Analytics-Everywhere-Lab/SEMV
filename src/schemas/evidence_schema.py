from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EvidenceSourceType = Literal[
    "case_provided",
    "media_metadata",
    "visual_description",
    "keyframe",
    "visual_caption",
    "visual_objects",
    "visual_vqa",
    "frame_analysis",
    "scene_keyframe",
    "ocr",
    "asr",
    "metadata_exiftool",
    "metadata_ffprobe",
    "forensic_analysis",
    "reverse_image_local",
    "reverse_image_web_candidate",
    "visual_similarity",
    "web_article",
    "news_article",
    "factcheck_article",
    "geolocation_candidate",
    "source_analysis",
    "cached_search",
    "manual_research",
    "synthetic_uncertainty",
]


class Provenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_id: str
    source_type: EvidenceSourceType
    source: str
    url: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retrieval_method: str = "local"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    evidence_id: str
    source_type: EvidenceSourceType = "case_provided"
    source: str = "case"
    title: str | None = None
    content: str
    url: str | None = None
    reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    media_path: str | None = None
    confidence: float | None = None
    timestamp_sec: float | None = None
    frame_path: str | None = None
    bbox: list[float] | None = None
    language: str | None = None
    raw_output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    uncertainty_flags: list[str] = Field(default_factory=list)
    supports_claim_types: list[str] = Field(default_factory=list)
    provenance: Provenance | None = None
    human_status: Literal["unreviewed", "contested", "rejected"] = "unreviewed"
    excluded_by_human: bool = False


class EvidenceGraph(BaseModel):
    nodes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    edges: list[dict[str, str]] = Field(default_factory=list)

    def linked_evidence(self, claim_id: str) -> list[str]:
        return [
            edge["to"]
            for edge in self.edges
            if edge.get("from") == claim_id and edge.get("relation") == "uses_evidence"
        ]
