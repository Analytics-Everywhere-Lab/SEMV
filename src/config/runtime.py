from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.utils.io import read_yaml


class RuntimeFeatures(BaseModel):
    """Immutable switches for one pipeline execution or ablation condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    use_memory: bool = True
    memory_types: tuple[Literal["episodic", "failure", "semantic_rule"], ...] = (
        "episodic", "failure", "semantic_rule"
    )
    use_qbaf: bool = True
    argument_verifier: bool = True
    clash_resolution: bool = True
    adaptive_revision: bool = True


class PipelineRuntimeConfig(BaseModel):
    """Resolved, injectable configuration for behavior that affects a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_top_k: int = Field(default=10, ge=1)
    memory_top_k: int = Field(default=5, ge=1)
    features: RuntimeFeatures = Field(default_factory=RuntimeFeatures)


def load_runtime_config(path: str | Path = "configs/default.yaml") -> PipelineRuntimeConfig:
    data = read_yaml(path)
    pipeline = data.get("pipeline") or {}
    allowed = {"evidence_top_k", "memory_top_k", "features"}
    important_unknown = set(pipeline) - allowed - {"default_mode"}
    if important_unknown:
        raise ValueError(
            "Unknown pipeline configuration field(s): "
            + ", ".join(sorted(important_unknown))
        )
    return PipelineRuntimeConfig.model_validate(
        {key: value for key, value in pipeline.items() if key in allowed}
    )
