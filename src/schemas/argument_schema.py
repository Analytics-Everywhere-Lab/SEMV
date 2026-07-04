from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.contestation_schema import ArgumentProvenance


ArgumentStance = Literal["support", "attack", "mixed", "neutral"]
HumanArgumentStatus = Literal["unreviewed", "accepted", "rejected", "edited", "added"]


class Argument(BaseModel):
    model_config = ConfigDict(extra="allow")

    argument_id: str
    claim_id: str
    case_id: str | None = None
    claim_type: str | None = None
    stance: ArgumentStance
    title: str = "Argument"
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str | None = None
    intrinsic_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    claim_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    cross_source_consistency: float = Field(default=0.5, ge=0.0, le=1.0)
    cross_modal_consistency: float = Field(default=0.5, ge=0.0, le=1.0)
    groundedness: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    corroboration: float = Field(default=0.5, ge=0.0, le=1.0)
    provenance_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    specificity: float = Field(default=0.5, ge=0.0, le=1.0)
    verifier_valid: bool = True
    verification_notes: str | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_flags: list[str] = Field(default_factory=list)
    provenance: ArgumentProvenance | None = None
    human_status: HumanArgumentStatus = "unreviewed"
    human_original_argument_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
