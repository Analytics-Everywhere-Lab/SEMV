from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.evidence_schema import EvidenceItem


MediaType = Literal["image", "video", "unknown"]


class MediaItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    media_type: MediaType = "unknown"
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolved_path(self, base_dir: Path | None = None) -> Path:
        candidate = Path(self.path)
        if candidate.is_absolute() or base_dir is None:
            return candidate
        return (base_dir / candidate).resolve()


class MultimediaCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    claim: str
    media: list[MediaItem] = Field(default_factory=list)
    context: str | None = None
    provided_evidence: list[EvidenceItem] = Field(default_factory=list)
    expected_label: str | None = None
    subclaim_labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
