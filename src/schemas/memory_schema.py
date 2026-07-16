from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MemoryType = Literal["episodic", "failure", "semantic_rule"]

MemoryLevel = Literal["short_term", "long_term"]

VerificationStatus = Literal["pending", "verified", "rejected", "under_review"]

LifecycleStatus = Literal[
    "staged",
    "active",
    "under_review",
    "deprecated",
    "promoted",
    "merged",
    "expired",
]

SemanticRelation = Literal[
    "equivalent", "entails", "a_entails_b", "b_entails_a", "contradicts", "unrelated"
]

SupervisionSource = Literal["gold_label", "human_feedback", "self_reflection"]

MemoryOrigin = Literal["seed", "consolidated", "manual", "legacy"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    memory_id: str
    memory_type: MemoryType
    memory_level: MemoryLevel = "long_term"
    version: int = 1
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
    canonical_key: str | None = None
    semantic_signature: str | None = None
    applicability_scope: str | None = None
    exceptions: list[str] = Field(default_factory=list)
    polarity: str | None = None
    source_case_ids: list[str] = Field(default_factory=list)
    source_fingerprints: list[str] = Field(default_factory=list)
    source_datasets: list[str] = Field(default_factory=list)
    source_splits: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    support_count: int = 1
    conflict_count: int = 0
    support_weight: float = 0.0
    conflict_weight: float = 0.0
    usage_count: int = 0
    successful_usage_count: int = 0
    contested_usage_count: int = 0
    last_used_at: str | None = None
    last_confirmed_at: str | None = None
    created_at: datetime | str | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: str | None = None
    verified_by: str | None = None
    status: LifecycleStatus = "active"
    superseded_by: str | None = None
    origin: MemoryOrigin = "legacy"
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

    def independent_support(self) -> int:
        """Independent support = unique source fingerprints (falling back to case ids)."""
        fingerprints = set(self.source_fingerprints)
        cases = set(self.source_case_ids)
        return max(self.support_count, len(fingerprints), len(cases))


class MemoryUpdateCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_id: str
    memory_type: MemoryType
    text: str
    source_case_id: str
    dataset_name: str | None = None
    dataset_split: str | None = None
    task_type: str | None = None
    claim_type: str | None = None
    failure_type: str | None = None
    trigger_pattern: str | None = None
    observed_failure_or_success: str | None = None
    lesson: str | None = None
    evidence_pattern: str | None = None
    argument_pattern: str | None = None
    recommended_action: str | None = None
    normalized_text: str | None = None
    canonical_key: str | None = None
    semantic_signature: str | None = None
    applicability_scope: str | None = None
    exceptions: list[str] = Field(default_factory=list)
    polarity: str | None = None
    source_fingerprint: str | None = None
    semantic_relation: SemanticRelation | None = None
    related_memory_id: str | None = None
    grounding_evidence_ids: list[str] = Field(default_factory=list)
    grounding_argument_ids: list[str] = Field(default_factory=list)
    supervision_source: SupervisionSource = "self_reflection"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str | None = None
    grounding: dict[str, Any] = Field(default_factory=dict)
    verified: bool = False
    verification_status: VerificationStatus = "pending"
    verified_by: str | None = None
    rejected_reason: str | None = None
    created_at: str | None = Field(default_factory=utc_now_iso)
    staged_at: str | None = None
    promoted_to_memory_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def merge_legacy_grounding(cls, data: Any) -> Any:
        """Old rows kept grounding ids inside the free-form `grounding` dict."""
        if isinstance(data, dict):
            grounding = data.get("grounding") or {}
            if isinstance(grounding, dict):
                if not data.get("grounding_evidence_ids") and grounding.get("evidence_ids"):
                    data = dict(data)
                    data["grounding_evidence_ids"] = list(grounding["evidence_ids"])
                if not data.get("grounding_argument_ids") and grounding.get("argument_ids"):
                    data = dict(data)
                    data["grounding_argument_ids"] = list(grounding["argument_ids"])
        return data


class ShortTermMemoryRecord(BaseModel):
    """A persistent staging record: verified case-grounded observation awaiting consolidation."""

    model_config = ConfigDict(extra="allow")

    stm_id: str
    candidate_id: str
    memory_type: MemoryType
    memory_level: MemoryLevel = "short_term"
    text: str
    source_case_id: str
    dataset_name: str | None = None
    dataset_split: str | None = None
    task_type: str | None = None
    claim_type: str | None = None
    failure_type: str | None = None
    trigger_pattern: str | None = None
    lesson: str | None = None
    evidence_pattern: str | None = None
    argument_pattern: str | None = None
    recommended_action: str | None = None
    normalized_text: str | None = None
    canonical_key: str | None = None
    semantic_signature: str | None = None
    applicability_scope: str | None = None
    exceptions: list[str] = Field(default_factory=list)
    polarity: str | None = None
    source_fingerprint: str | None = None
    semantic_relation: SemanticRelation | None = None
    related_memory_id: str | None = None
    grounding_evidence_ids: list[str] = Field(default_factory=list)
    grounding_argument_ids: list[str] = Field(default_factory=list)
    supervision_source: SupervisionSource = "self_reflection"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    verification_status: VerificationStatus = "pending"
    verified_by: str | None = None
    status: LifecycleStatus = "staged"
    promoted_to_memory_id: str | None = None
    created_at: str | None = Field(default_factory=utc_now_iso)
    staged_at: str | None = Field(default_factory=utc_now_iso)
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_candidate(cls, candidate: MemoryUpdateCandidate) -> "ShortTermMemoryRecord":
        return cls(
            stm_id=f"stm_{candidate.candidate_id}",
            candidate_id=candidate.candidate_id,
            memory_type=candidate.memory_type,
            text=candidate.text,
            source_case_id=candidate.source_case_id,
            dataset_name=candidate.dataset_name,
            dataset_split=candidate.dataset_split,
            task_type=candidate.task_type,
            claim_type=candidate.claim_type,
            failure_type=candidate.failure_type,
            trigger_pattern=candidate.trigger_pattern,
            lesson=candidate.lesson,
            evidence_pattern=candidate.evidence_pattern,
            argument_pattern=candidate.argument_pattern,
            recommended_action=candidate.recommended_action,
            normalized_text=candidate.normalized_text,
            canonical_key=candidate.canonical_key,
            semantic_signature=candidate.semantic_signature,
            applicability_scope=candidate.applicability_scope,
            exceptions=list(candidate.exceptions),
            polarity=candidate.polarity,
            source_fingerprint=candidate.source_fingerprint,
            semantic_relation=candidate.semantic_relation,
            related_memory_id=candidate.related_memory_id,
            grounding_evidence_ids=list(candidate.grounding_evidence_ids),
            grounding_argument_ids=list(candidate.grounding_argument_ids),
            supervision_source=candidate.supervision_source,
            confidence=candidate.confidence,
            verification_status=candidate.verification_status,
            verified_by=candidate.verified_by,
            status="under_review" if candidate.verification_status == "under_review" else "staged",
            metadata=dict(candidate.metadata),
        )


class MemoryUsageEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str
    case_id: str
    run_id: str | None = None
    protocol_phase: str = "unknown"
    memory_id: str
    stage: Literal["retrieved", "planner_cited", "argument_cited"]
    claim_id: str | None = None
    argument_id: str | None = None
    outcome: Literal["successful", "contested", "unknown"] = "unknown"
    dataset_name: str | None = None
    dataset_split: str | None = None
    frozen: bool = False
    created_at: str | None = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsolidationEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str
    event_type: Literal[
        "staged",
        "promoted",
        "merged",
        "support_increment",
        "conflict",
        "under_review",
        "deprecated",
        "expired",
        "generalized",
        "generalization_failed",
        "generalization_recovered",
        "snapshot",
        "archived",
        "candidate_verification",
        "usage_rollup",
    ]
    memory_id: str | None = None
    stm_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = Field(default_factory=utc_now_iso)


class ConsolidationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: str | None = None
    dry_run: bool = False
    stm_considered: int = 0
    staged: list[str] = Field(default_factory=list)
    promoted: list[str] = Field(default_factory=list)
    merged: list[str] = Field(default_factory=list)
    conflicted: list[str] = Field(default_factory=list)
    deprecated: list[str] = Field(default_factory=list)
    under_review: list[str] = Field(default_factory=list)
    expired: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    support_increments: dict[str, int] = Field(default_factory=dict)
    changed_long_term_ids: list[str] = Field(default_factory=list)
    counts_before: dict[str, int] = Field(default_factory=dict)
    counts_after: dict[str, int] = Field(default_factory=dict)
    events: list[ConsolidationEvent] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    snapshot_path: str | None = None
    state_hash: str | None = None
