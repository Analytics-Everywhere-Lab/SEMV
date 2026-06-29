from __future__ import annotations

from pathlib import Path

from src.ingestion.base_adapter import BaseDatasetAdapter
from src.ingestion.default_claim_builder import build_default_claims
from src.ingestion.media_manifest_builder import (
    build_media_assets,
    infer_case_media_type,
)
from src.ingestion.source_cluster_builder import build_source_clusters
from src.schemas.case_bundle_schema import (
    CaseBundle,
    DatasetInfo,
    GoldAnnotation,
    InputMetadata,
    LocationContext,
    RunConfig,
    TaskInfo,
    TemporalContext,
)
from src.utils.io import read_json


class MV2026Adapter(BaseDatasetAdapter):
    adapter_name = "mv2026_folder"
    adapter_version = "1.0"

    def can_load(self, case_path: Path) -> bool:
        input_dir = case_path / "input"
        return input_dir.exists() and any(input_dir.glob("*.json"))

    def load(self, case_path: Path, split: str | None = None) -> CaseBundle:
        case_path = Path(case_path)
        case_id = case_path.name
        input_json_path = next((case_path / "input").glob("*.json"))
        raw_input = read_json(input_json_path)
        media_assets = build_media_assets(case_id, case_path / "input" / "media")
        source_clusters = build_source_clusters(case_id, raw_input, media_assets)
        claims = build_default_claims(
            case_id=case_id,
            raw_input=raw_input,
            media_assets=media_assets,
            source_clusters=source_clusters,
        )
        gold_report_path = case_path / "output" / "report.md"
        gold_assets_dir = case_path / "output" / "report"
        return CaseBundle(
            case_id=case_id,
            dataset=DatasetInfo(
                dataset_name="mv2026",
                dataset_split=split,
                native_case_path=str(case_path),
                native_format="mv2026_folder",
                adapter_version=self.adapter_version,
            ),
            task=TaskInfo(
                task_type="multimedia_verification",
                subtask="cheapfake_or_out_of_context_verification",
                media_type=infer_case_media_type(media_assets),  # type: ignore[arg-type]
                expected_output="report_and_label",
                language=str(raw_input.get("language", "en")),
            ),
            input=InputMetadata(
                title=raw_input.get("title"),
                caption=raw_input.get("caption"),
                description=raw_input.get("description"),
                social_media_link=raw_input.get("social media link")
                or raw_input.get("social_media_link")
                or raw_input.get("source_url"),
                location_hint=raw_input.get("location") or raw_input.get("location_hint"),
                violence_level=raw_input.get("violence level")
                or raw_input.get("violence_level"),
                category=raw_input.get("category"),
                raw_input_json_path=str(input_json_path),
                extra=raw_input,
            ),
            claims=claims,
            media_assets=media_assets,
            source_clusters=source_clusters,
            temporal_context=TemporalContext(
                claimed_time_text=_extract_time_text(raw_input),
                time_reasoning_signals=[
                    "publication time",
                    "reverse search",
                    "metadata",
                    "news context",
                    "shadow direction",
                    "same-day source cluster",
                ],
            ),
            location_context=LocationContext(
                claimed_location_text=raw_input.get("location")
                or raw_input.get("location_hint")
            ),
            provided_evidence=[],
            gold=GoldAnnotation(
                gold_report_path=str(gold_report_path) if gold_report_path.exists() else None,
                gold_assets_dir=str(gold_assets_dir) if gold_assets_dir.exists() else None,
                gold_report_available=gold_report_path.exists(),
                gold_visibility="test_hidden"
                if split == "test"
                else "train_or_validation",
                read_gold_before_prediction=False,
            ),
            run_config=RunConfig(
                allow_web_search=True,
                allow_reverse_search=True,
                allow_memory_retrieval=True,
                allow_memory_update=False,
                allow_human_contestation=True,
                expected_output_formats=["json", "markdown"],
            ),
        )


def _extract_time_text(raw_input: dict) -> str | None:
    for key in ["time", "date", "published_at", "publication_time", "created_at"]:
        value = raw_input.get(key)
        if value:
            return str(value)
    return None
