from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


QBAFRelation = Literal["support", "attack"]


class QBAFNode(BaseModel):
    node_id: str
    node_type: Literal["claim", "argument"]
    base_score: float = Field(default=0.5, ge=0.0, le=1.0)
    final_score: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QBAFEdge(BaseModel):
    from_node: str
    to_node: str
    relation: QBAFRelation
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class QBAFGraph(BaseModel):
    claim_id: str
    nodes: dict[str, QBAFNode] = Field(default_factory=dict)
    edges: list[QBAFEdge] = Field(default_factory=list)
    claim_score: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty_flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
