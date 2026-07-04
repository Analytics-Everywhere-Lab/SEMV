from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


HumanAction = Literal["accept", "reject", "edit", "add"]

RevisionTarget = Literal[
    "claim_decomposition",
    "evidence_retrieval",
    "evidence_validation",
    "argument_construction",
    "qbaf_reasoning",
    "final_aggregation",
    "report_generation",
]


class ArgumentProvenance(BaseModel):
    source_step: RevisionTarget
    subclaim_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    retrieval_query_ids: list[str] = Field(default_factory=list)
    upstream_steps: list[RevisionTarget] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanArgumentContestation(BaseModel):
    contestation_id: str
    case_id: str
    action: HumanAction

    target_argument_id: str | None = None

    edited_text: str | None = None
    edited_stance: str | None = None
    edited_confidence: float | None = None

    added_subclaim_id: str | None = None
    added_text: str | None = None
    added_stance: str | None = None
    added_evidence_ids: list[str] = Field(default_factory=list)

    reason: str | None = None

    reviewer_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanReviewBatch(BaseModel):
    case_id: str
    reviewer_id: str | None = None
    contestations: list[HumanArgumentContestation] = Field(default_factory=list)
    global_comment: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RevisionPlan(BaseModel):
    case_id: str
    revision_target: RevisionTarget
    rerun_from_step: RevisionTarget
    affected_argument_ids: list[str] = Field(default_factory=list)
    affected_subclaim_ids: list[str] = Field(default_factory=list)
    affected_evidence_ids: list[str] = Field(default_factory=list)
    affected_retrieval_query_ids: list[str] = Field(default_factory=list)
    human_actions: list[HumanAction] = Field(default_factory=list)
    rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)
