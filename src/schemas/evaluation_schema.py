from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SubclaimPrediction(BaseModel):
    claim_type: Literal[
        "what",
        "where",
        "when",
        "who",
        "why",
        "authenticity",
        "main",
        "caption_context",
    ]
    statement: str
    decision: str
    score: float | None = None
    confidence: float | None = None
    top_support_argument_ids: list[str] = Field(default_factory=list)
    top_attack_argument_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    uncertainty_reason: str | None = None


class PredictionRecord(BaseModel):
    case_id: str
    dataset_name: str
    task_type: str
    final_label: str
    final_score: float | None = None
    final_confidence: float | None = None
    subclaims: list[SubclaimPrediction] = Field(default_factory=list)
    predicted_coordinates: list[dict[str, Any]] = Field(default_factory=list)
    predicted_time_bounds: dict[str, Any] = Field(default_factory=dict)
    predicted_entities: list[str] = Field(default_factory=list)
    predicted_source_urls: list[str] = Field(default_factory=list)
    report_path: str | None = None
    report_json_path: str | None = None
    memory_used_ids: list[str] = Field(default_factory=list)
    run_metadata: dict[str, Any] = Field(default_factory=dict)


class GoldRecord(BaseModel):
    case_id: str
    dataset_name: str
    task_type: str
    gold_final_label: str | None = None
    gold_status_text: str | None = None
    gold_subclaim_labels: dict[str, str] = Field(default_factory=dict)
    gold_subclaim_text: dict[str, str] = Field(default_factory=dict)
    gold_coordinates: list[dict[str, Any]] = Field(default_factory=list)
    gold_time_bounds: dict[str, Any] = Field(default_factory=dict)
    gold_entities: list[str] = Field(default_factory=list)
    gold_source_urls: list[str] = Field(default_factory=list)
    gold_report_path: str | None = None
    gold_raw_sections: dict[str, str] = Field(default_factory=dict)


class CaseMetricRecord(BaseModel):
    case_id: str
    dataset_name: str
    final_label_correct: bool | None = None
    final_label_score: float | None = None
    subclaim_accuracy: dict[str, float] = Field(default_factory=dict)
    subclaim_f1: dict[str, float] = Field(default_factory=dict)
    evidence_url_precision: float | None = None
    evidence_url_recall: float | None = None
    evidence_url_f1: float | None = None
    geolocation_error_meters: float | None = None
    geolocation_within_100m: bool | None = None
    geolocation_within_500m: bool | None = None
    geolocation_within_1km: bool | None = None
    temporal_bound_score: float | None = None
    temporal_interval_iou: float | None = None
    publication_event_confusion: bool | None = None
    entity_precision: float | None = None
    entity_recall: float | None = None
    entity_f1: float | None = None
    report_structure_score: float | None = None
    provenance_validity_score: float | None = None
    hallucinated_source_rate: float | None = None
    calibration_error: float | None = None
    memory_helpfulness_score: float | None = None
    notes: dict[str, Any] = Field(default_factory=dict)
