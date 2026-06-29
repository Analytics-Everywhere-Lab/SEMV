from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.case_schema import MediaItem, MultimediaCase
from src.schemas.evidence_schema import EvidenceItem


class DatasetInfo(BaseModel):
    dataset_name: str
    dataset_split: str | None = None
    native_case_path: str | None = None
    native_format: str | None = None
    adapter_version: str = "1.0"


class TaskInfo(BaseModel):
    task_type: Literal[
        "multimedia_verification",
        "cheapfake_detection",
        "out_of_context_detection",
        "multimodal_fact_verification",
    ]
    subtask: str | None = None
    media_type: Literal["image", "video", "multi_image", "multi_video", "mixed"]
    expected_output: Literal["label_only", "report_only", "report_and_label"] = (
        "report_and_label"
    )
    language: str = "en"


class InputMetadata(BaseModel):
    title: str | None = None
    caption: str | None = None
    description: str | None = None
    social_media_link: str | None = None
    location_hint: str | None = None
    violence_level: str | None = None
    category: str | None = None
    raw_input_json_path: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    claim_id: str
    claim_type: Literal[
        "main",
        "what",
        "where",
        "when",
        "who",
        "why",
        "authenticity",
        "caption_context",
    ]
    scope_type: Literal[
        "case",
        "media",
        "segment",
        "source_cluster",
        "event_cluster",
    ] = "case"
    statement: str
    media_ids: list[str] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    source_cluster_id: str | None = None
    priority: float = 1.0
    expected_evidence_types: list[str] = Field(default_factory=list)


class MediaAsset(BaseModel):
    media_id: str
    case_id: str
    media_type: Literal[
        "image",
        "video",
        "audio",
        "screenshot",
        "document_image",
        "keyframe",
        "unknown",
    ]
    role: Literal[
        "primary_claim_media",
        "related_claim_media",
        "context_media",
        "reference_media",
        "source_statement_media",
        "report_attachment",
        "derived_keyframe",
        "unknown",
    ] = "unknown"
    local_path: str | None = None
    source_url: str | None = None
    platform: str | None = None
    creator_or_uploader: str | None = None
    group_id: str | None = None
    sequence_index: int | None = None
    claimed_or_observed_publish_time: str | None = None
    language_hints: list[str] = Field(default_factory=list)
    description: str | None = None
    is_gold_only: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceCluster(BaseModel):
    cluster_id: str
    case_id: str
    source_name: str | None = None
    platform: str | None = None
    source_type: Literal[
        "original_uploader",
        "reposter",
        "official_statement",
        "news_media",
        "factchecker",
        "social_media_user",
        "benchmark_provider",
        "unknown",
    ] = "unknown"
    media_ids: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    observed_publish_times: list[str] = Field(default_factory=list)
    credibility_prior: float | None = None
    notes: str | None = None


class PublicationTime(BaseModel):
    source: str | None = None
    time_text: str | None = None
    normalized_time: str | None = None
    timezone: str | None = None
    evidence_id: str | None = None


class EventTimeBounds(BaseModel):
    earliest_possible: str | None = None
    latest_possible: str | None = None
    bound_type: Literal[
        "unknown",
        "exact",
        "interval",
        "latest_known_occurrence",
        "earliest_known_occurrence",
    ] = "unknown"
    exact_recording_time_known: bool = False


class TemporalContext(BaseModel):
    claimed_time_text: str | None = None
    publication_times: list[PublicationTime] = Field(default_factory=list)
    event_time_bounds: EventTimeBounds = Field(default_factory=EventTimeBounds)
    time_reasoning_signals: list[str] = Field(default_factory=list)
    timezone_assumptions: list[str] = Field(default_factory=list)


class GeoPoint(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    confidence: float | None = None
    description: str | None = None


class LocationContext(BaseModel):
    claimed_location_text: str | None = None
    camera_location: GeoPoint = Field(default_factory=GeoPoint)
    target_location: GeoPoint = Field(default_factory=GeoPoint)
    view_direction: str | None = None
    geolocation_cues: list[str] = Field(default_factory=list)


class ProvidedEvidence(BaseModel):
    evidence_id: str
    source_type: str
    modality: Literal["image", "video", "audio", "text", "metadata", "mixed"]
    content: str
    source_url: str | None = None
    timestamp: str | None = None
    reliability_score: float | None = None
    linked_claim_types: list[str] = Field(default_factory=list)
    media_ids: list[str] = Field(default_factory=list)
    is_gold_only: bool = False
    provenance: dict[str, Any] = Field(default_factory=dict)


class GoldAnnotation(BaseModel):
    gold_report_path: str | None = None
    gold_assets_dir: str | None = None
    gold_report_available: bool = False
    gold_visibility: Literal[
        "none",
        "train",
        "validation",
        "train_or_validation",
        "test_hidden",
    ] = "none"
    gold_final_label: str | None = None
    gold_subclaim_labels: dict[str, str] = Field(default_factory=dict)
    read_gold_before_prediction: bool = False


class RunConfig(BaseModel):
    allow_web_search: bool = True
    allow_reverse_search: bool = True
    allow_memory_retrieval: bool = True
    allow_memory_update: bool = False
    allow_human_contestation: bool = True
    expected_output_formats: list[str] = Field(
        default_factory=lambda: ["json", "markdown"]
    )


class CaseBundle(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    dataset: DatasetInfo
    task: TaskInfo
    input: InputMetadata
    claims: list[Claim] = Field(default_factory=list)
    media_assets: list[MediaAsset] = Field(default_factory=list)
    source_clusters: list[SourceCluster] = Field(default_factory=list)
    temporal_context: TemporalContext = Field(default_factory=TemporalContext)
    location_context: LocationContext = Field(default_factory=LocationContext)
    provided_evidence: list[ProvidedEvidence] = Field(default_factory=list)
    gold: GoldAnnotation = Field(default_factory=GoldAnnotation)
    run_config: RunConfig = Field(default_factory=RunConfig)

    def primary_claim_text(self) -> str:
        for value in (self.input.title, self.input.caption, self.input.description):
            if value:
                return value
        main = next((claim for claim in self.claims if claim.claim_type == "main"), None)
        if main:
            return main.statement
        return "The media depicts the event described in the source context."

    def context_text(self) -> str | None:
        parts = [
            self.input.description,
            self.input.caption,
            self.input.location_hint,
            self.input.social_media_link,
        ]
        text = "\n".join(part for part in parts if part)
        return text or None


def case_bundle_to_multimedia_case(bundle: CaseBundle) -> MultimediaCase:
    media = [
        MediaItem(
            path=asset.local_path or asset.source_url or asset.media_id,
            media_type=_legacy_media_type(asset.media_type),
            description=asset.description,
            metadata={
                "media_id": asset.media_id,
                "role": asset.role,
                "source_url": asset.source_url,
                "is_gold_only": asset.is_gold_only,
                **asset.metadata,
            },
        )
        for asset in bundle.media_assets
        if not asset.is_gold_only and asset.role != "report_attachment"
    ]
    return MultimediaCase(
        case_id=bundle.case_id,
        claim=bundle.primary_claim_text(),
        media=media,
        context=bundle.context_text(),
        provided_evidence=[
            _provided_evidence_to_evidence_item(item)
            for item in bundle.provided_evidence
            if not item.is_gold_only
        ],
        expected_label=bundle.gold.gold_final_label,
        subclaim_labels=bundle.gold.gold_subclaim_labels,
        metadata={
            "dataset": bundle.dataset.model_dump(mode="json"),
            "task": bundle.task.model_dump(mode="json"),
            "case_bundle": bundle.model_dump(mode="json"),
        },
    )


def multimedia_case_to_case_bundle(case: MultimediaCase) -> CaseBundle:
    media_assets = [
        MediaAsset(
            media_id=f"{case.case_id}_media_{idx:03d}",
            case_id=case.case_id,
            media_type=_asset_media_type(item.media_type),
            role="primary_claim_media" if idx == 1 else "related_claim_media",
            local_path=item.path,
            sequence_index=idx,
            description=item.description,
            metadata=item.metadata,
        )
        for idx, item in enumerate(case.media, start=1)
    ]
    return CaseBundle(
        case_id=case.case_id,
        dataset=DatasetInfo(dataset_name="legacy", native_format="multimedia_case"),
        task=TaskInfo(
            task_type="multimedia_verification",
            media_type=_infer_case_media_type(media_assets),
            expected_output="report_and_label",
        ),
        input=InputMetadata(
            title=case.claim,
            description=case.context,
            extra=case.metadata,
        ),
        media_assets=media_assets,
        provided_evidence=[
            ProvidedEvidence(
                evidence_id=item.evidence_id,
                source_type=item.source_type,
                modality="text",
                content=item.content,
                source_url=item.url,
                reliability_score=item.reliability,
                linked_claim_types=item.supports_claim_types,
                provenance=item.provenance.model_dump(mode="json")
                if item.provenance
                else {},
            )
            for item in case.provided_evidence
        ],
        gold=GoldAnnotation(
            gold_final_label=case.expected_label,
            gold_subclaim_labels=case.subclaim_labels,
            read_gold_before_prediction=False,
        ),
    )


def _legacy_media_type(media_type: str) -> str:
    return media_type if media_type in {"image", "video"} else "unknown"


def _asset_media_type(media_type: str) -> str:
    return media_type if media_type in {"image", "video"} else "unknown"


def _infer_case_media_type(media_assets: list[MediaAsset]) -> str:
    types = {asset.media_type for asset in media_assets if asset.media_type != "unknown"}
    if not types:
        return "mixed"
    if types == {"image"}:
        return "image" if len(media_assets) <= 1 else "multi_image"
    if types == {"video"}:
        return "video" if len(media_assets) <= 1 else "multi_video"
    return "mixed"


def _provided_evidence_to_evidence_item(item: ProvidedEvidence) -> EvidenceItem:
    from src.schemas.evidence_schema import Provenance

    provenance = None
    if item.provenance:
        provenance = Provenance(
            source_id=str(item.provenance.get("source_id", item.evidence_id)),
            source_type="case_provided",
            source=str(item.provenance.get("source", item.source_url or "case")),
            url=item.source_url,
            retrieval_method=str(item.provenance.get("retrieval_method", "provided")),
            metadata=item.provenance,
        )
    return EvidenceItem(
        evidence_id=item.evidence_id,
        source_type="case_provided",
        source=item.source_type,
        title=item.source_type,
        content=item.content,
        url=item.source_url,
        reliability=item.reliability_score or 0.5,
        relevance=0.65,
        supports_claim_types=item.linked_claim_types,
        provenance=provenance,
        metadata={"media_ids": item.media_ids, "timestamp": item.timestamp},
    )


def load_case_bundle(path: str | Path) -> CaseBundle:
    import json

    return CaseBundle.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
