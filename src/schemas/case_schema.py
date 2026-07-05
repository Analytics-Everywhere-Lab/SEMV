from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.evidence_schema import EvidenceItem
from src.utils.io import project_root


MediaType = Literal["image", "video", "unknown"]


class MediaItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    media_type: MediaType = "unknown"
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolved_path(self, base_dir: Path | None = None) -> Path:
        candidate = Path(self.path)
        if candidate.is_absolute():
            return candidate

        if base_dir is not None:
            by_base = (base_dir / candidate).resolve()
            if by_base.exists():
                return by_base

        by_project = (project_root() / candidate).resolve()
        if by_project.exists():
            return by_project

        return (base_dir / candidate).resolve() if base_dir is not None else by_project


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
