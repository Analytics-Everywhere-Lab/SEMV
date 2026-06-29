from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MemoryType = Literal["episodic", "failure", "semantic_rule"]


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    memory_id: str
    memory_type: MemoryType
    case_id: str | None = None
    claim_type: str | None = None
    task_type: str | None = None
    text: str = ""
    trigger_pattern: str | None = None
    lesson: str | None = None
    evidence_pattern: str | None = None
    argument_pattern: str | None = None
    recommended_action: str | None = None
    failure_type: str | None = None
    source_case_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    support_count: int = 1
    conflict_count: int = 0
    usage_count: int = 0
    last_used_at: str | None = None
    created_at: datetime | str | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: str | None = None
    verified_by: str | None = None
    status: Literal["active", "deprecated", "under_review"] = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def fill_text(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("text"):
            data = dict(data)
            parts = [
                data.get("lesson"),
                data.get("recommended_action"),
                data.get("trigger_pattern"),
            ]
            data["text"] = " ".join(str(part) for part in parts if part) or data.get("memory_id", "")
        return data


class MemoryUpdateCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_id: str
    memory_type: MemoryType
    text: str
    source_case_id: str
    claim_type: str | None = None
    trigger_pattern: str | None = None
    observed_failure_or_success: str | None = None
    lesson: str | None = None
    evidence_pattern: str | None = None
    argument_pattern: str | None = None
    recommended_action: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str | None = None
    grounding: dict[str, Any] = Field(default_factory=dict)
    verified: bool = False
    rejected_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
