from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.argument_schema import Argument
from src.schemas.contestation_schema import HumanReviewBatch, RevisionPlan
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem
from src.schemas.memory_schema import MemoryRecord, MemoryUpdateCandidate, ShortTermMemoryRecord


class SubClaimReport(BaseModel):
    claim_id: str
    claim_type: str
    statement: str
    score: float
    decision: str
    confidence: float
    top_support_arguments: list[Argument] = Field(default_factory=list)
    top_attack_arguments: list[Argument] = Field(default_factory=list)
    uncertainty_reason: str | None = None


class ReflectionLog(BaseModel):
    case_id: str
    predicted_label: str
    ground_truth_label: str | None = None
    human_feedback: dict[str, Any] = Field(default_factory=dict)
    failure_modes: list[str] = Field(default_factory=list)
    lessons: list[str] = Field(default_factory=list)
    media_analysis: dict[str, list[Any]] = Field(default_factory=dict)
    escalation: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    final_status: str
    # Confidence in `final_status`, including the "uncertain" label itself -
    # e.g. final_status="uncertain", final_confidence=0.8 means high confidence
    # that the case is genuinely uncertain, not 80% confidence in a verdict.
    final_confidence: float
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    subclaim_reports: list[SubClaimReport] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    evidence_graph: EvidenceGraph = Field(default_factory=EvidenceGraph)
    # memory_retrieved = everything retrieval returned; memory_used = only the
    # records actually cited by the planner or an argument via used_memory_ids.
    memory_retrieved: list[MemoryRecord] = Field(default_factory=list)
    memory_used: list[MemoryRecord] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    reflection_logs: list[ReflectionLog] = Field(default_factory=list)
    memory_update_candidates: list[MemoryUpdateCandidate] = Field(default_factory=list)
    # memory_updates_applied = long-term records actually changed by
    # consolidation; staged short-term records live in memory_updates_staged.
    memory_updates_applied: list[MemoryRecord] = Field(default_factory=list)
    memory_updates_staged: list[ShortTermMemoryRecord] = Field(default_factory=list)
    memory_consolidation_events: list[dict[str, Any]] = Field(default_factory=list)
    human_review_applied: bool = False
    human_review_batch: HumanReviewBatch | None = None
    revision_plan: RevisionPlan | None = None
    contestation_summary: dict[str, Any] = Field(default_factory=dict)
    media_analysis: dict[str, list[Any]] = Field(default_factory=dict)
    escalation: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
