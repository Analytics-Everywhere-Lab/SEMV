from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ClaimType = Literal[
    "main",
    "what",
    "where",
    "when",
    "who",
    "why",
    "authenticity",
    "caption_context",
]


class SubClaim(BaseModel):
    model_config = ConfigDict(extra="allow")

    claim_id: str
    claim_type: ClaimType
    statement: str
    priority: int = Field(default=1, ge=1)
    search_queries: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchPlan(BaseModel):
    claim_id: str
    questions: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    preferred_sources: list[str] = Field(default_factory=list)
    uncertainty_checks: list[str] = Field(default_factory=list)
    used_memory_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
