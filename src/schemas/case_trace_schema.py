from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CaseTrace(BaseModel):
    case_id: str
    input_bundle: dict[str, Any] | None = None
    subclaims: list[Any] = Field(default_factory=list)
    retrieval_queries: list[Any] = Field(default_factory=list)
    evidence_items: list[Any] = Field(default_factory=list)
    validated_evidence_items: list[Any] = Field(default_factory=list)
    arguments: list[Any] = Field(default_factory=list)
    qbaf_state: dict[str, Any] = Field(default_factory=dict)
    final_decision: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
