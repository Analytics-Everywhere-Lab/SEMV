from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.utils.io import read_yaml, resolve_project_path


class MemoryPathsConfig(BaseModel):
    memory_dir: str = "data/memory"
    short_term_file: str = "short_term_memory.jsonl"
    episodic_file: str = "episodic_memory.jsonl"
    failure_file: str = "failure_memory.jsonl"
    semantic_file: str = "semantic_rules.jsonl"
    event_log_file: str = "consolidation_events.jsonl"
    usage_log_file: str = "memory_usage_events.jsonl"
    archive_dir: str = "data/memory/archive"
    snapshot_dir: str = "data/memory/snapshots"

    def resolved_memory_dir(self) -> Path:
        return resolve_project_path(self.memory_dir)

    def resolved_archive_dir(self) -> Path:
        return resolve_project_path(self.archive_dir)

    def resolved_snapshot_dir(self) -> Path:
        return resolve_project_path(self.snapshot_dir)


class ShortTermConfig(BaseModel):
    max_records: int = 5000
    ttl_days: int = 30
    retrieve_during_bootstrap: bool = False
    archive_expired: bool = True


class VerificationConfig(BaseModel):
    min_confidence: float = 0.60
    # Contradictions are retained as verified evidence against an existing memory.
    contradiction_policy: Literal["verified_evidence"] = "verified_evidence"
    fail_policy: str = "under_review"
    require_grounding: bool = True


class SimilarityConfig(BaseModel):
    backend: str = "hybrid"
    duplicate_similarity: float = 0.85
    contradiction_similarity: float = 0.75
    lexical_shortlist_k: int = 20
    use_llm_relation_check: bool = True
    optional_embedding_backend: str | None = None


class PromotionThresholds(BaseModel):
    min_confidence: float = 0.75
    min_distinct_cases: int = 1
    min_distinct_sources: int = 1


class ConsolidationConfig(BaseModel):
    every_n_cases: int = 25
    episodic: PromotionThresholds = Field(
        default_factory=lambda: PromotionThresholds(min_confidence=0.85, min_distinct_cases=1, min_distinct_sources=1)
    )
    failure: PromotionThresholds = Field(
        default_factory=lambda: PromotionThresholds(min_confidence=0.70, min_distinct_cases=2, min_distinct_sources=2)
    )
    semantic_rule: PromotionThresholds = Field(
        default_factory=lambda: PromotionThresholds(min_confidence=0.75, min_distinct_cases=3, min_distinct_sources=3)
    )
    max_conflict_ratio: float = 0.20
    under_review_conflict_ratio: float = 0.30
    deprecate_confidence_below: float = 0.45
    generalize_repeated_episodes: bool = True

    def thresholds_for(self, memory_type: str) -> PromotionThresholds:
        return {
            "episodic": self.episodic,
            "failure": self.failure,
            "semantic_rule": self.semantic_rule,
        }.get(memory_type, self.semantic_rule)


class RetrievalConfig(BaseModel):
    top_k: int = 5
    min_similarity: float | None = 0.05
    min_semantic_similarity: float = 0.05
    min_final_score: float = 0.05
    include_memory_types: list[str] = Field(
        default_factory=lambda: ["episodic", "failure", "semantic_rule"]
    )
    active_only: bool = True


class MemoryConfig(BaseModel):
    paths: MemoryPathsConfig = Field(default_factory=MemoryPathsConfig)
    short_term: ShortTermConfig = Field(default_factory=ShortTermConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    def with_memory_dir(self, memory_dir: str | Path) -> "MemoryConfig":
        """Return a copy rooted at a run-specific memory directory."""
        memory_dir = Path(memory_dir)
        paths = self.paths.model_copy(
            update={
                "memory_dir": str(memory_dir),
                "archive_dir": str(memory_dir / "archive"),
                "snapshot_dir": str(memory_dir / "snapshots"),
            }
        )
        return self.model_copy(update={"paths": paths})


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_memory_config(
    config_path: str | Path | None = "configs/memory.yaml",
    override_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> MemoryConfig:
    """Load configs/memory.yaml with optional run-specific override file/dict.

    Missing files or keys fall back to typed defaults, so the loader never fails
    on a partial legacy config.
    """
    data: dict[str, Any] = {}
    if config_path is not None:
        data = read_yaml(config_path) or {}
        # Allow the memory block to be nested under a top-level `memory:` key.
        if "memory" in data and isinstance(data["memory"], dict):
            data = data["memory"]
    if override_path is not None:
        override_data = read_yaml(override_path) or {}
        if "memory" in override_data and isinstance(override_data["memory"], dict):
            override_data = override_data["memory"]
        data = _deep_merge(data, override_data)
    if overrides:
        data = _deep_merge(data, overrides)
    retrieval = data.get("retrieval")
    if isinstance(retrieval, dict) and retrieval.get("min_similarity") is not None:
        retrieval = dict(retrieval)
        retrieval.setdefault("min_final_score", retrieval["min_similarity"])
        retrieval.setdefault("min_semantic_similarity", retrieval["min_similarity"])
        data["retrieval"] = retrieval
        # Legacy flat key from the old configs/memory.yaml.
    consolidation = data.get("consolidation")
    if isinstance(consolidation, dict):
        consolidation = dict(consolidation)
        if "duplicate_similarity" in consolidation:
            similarity = dict(data.get("similarity") or {})
            similarity.setdefault("duplicate_similarity", consolidation["duplicate_similarity"])
            data["similarity"] = similarity
            consolidation.pop("duplicate_similarity")
        # Legacy global thresholds fill only absent per-type thresholds.
        legacy_sources = consolidation.pop("min_distinct_sources", None)
        if legacy_sources is not None:
            for memory_type in ("episodic", "failure", "semantic_rule"):
                per_type = dict(consolidation.get(memory_type) or {})
                per_type.setdefault("min_distinct_sources", legacy_sources)
                consolidation[memory_type] = per_type
        data["consolidation"] = consolidation
    verification = data.get("verification")
    if isinstance(verification, dict):
        verification = dict(verification)
        # The legacy boolean never selected a different implemented policy.
        if "reject_on_conflict" in verification:
            verification.pop("reject_on_conflict")
            verification.setdefault("contradiction_policy", "verified_evidence")
        data["verification"] = verification
    return MemoryConfig.model_validate(data)
