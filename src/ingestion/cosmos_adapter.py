from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.evaluation.label_normalizer import normalize_cosmos_label
from src.ingestion.base_adapter import BaseDatasetAdapter
from src.ingestion.media_manifest_builder import infer_media_type
from src.schemas.case_bundle_schema import (
    CaseBundle,
    Claim,
    DatasetInfo,
    GoldAnnotation,
    InputMetadata,
    MediaAsset,
    RunConfig,
    TaskInfo,
)
from src.utils.io import read_jsonl


class COSMOSAdapter(BaseDatasetAdapter):
    adapter_name = "cosmos"
    adapter_version = "1.0"

    def can_load(self, case_path: Path) -> bool:
        return case_path.suffix.lower() in {".jsonl", ".csv"} or (
            case_path.is_dir()
            and any((case_path / name).exists() for name in ["train.jsonl", "test.jsonl"])
        )

    def load(self, case_path: Path, split: str | None = None) -> CaseBundle:
        rows = load_cosmos_rows(case_path)
        if not rows:
            raise ValueError(f"No COSMOS rows found at {case_path}")
        return build_cosmos_case(rows[0], split=split, image_root=_image_root(case_path))


def load_cosmos_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if target.is_dir():
        for name in ["test.jsonl", "validation.jsonl", "val.jsonl", "train.jsonl"]:
            candidate = target / name
            if candidate.exists():
                return load_cosmos_rows(candidate)
        return []
    if target.suffix.lower() == ".jsonl":
        return read_jsonl(target)
    if target.suffix.lower() == ".csv":
        with target.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    return []


def build_cosmos_case(
    row: dict[str, Any],
    split: str | None = None,
    image_root: str | Path | None = None,
) -> CaseBundle:
    caption_2 = _first(row, "caption_2", "caption2", "text_2", "alt_caption")
    if caption_2:
        return build_cosmos_triplet_case(row, split=split, image_root=image_root)
    return build_cosmos_pair_case(row, split=split, image_root=image_root)


def build_cosmos_pair_case(
    row: dict[str, Any],
    split: str | None = None,
    image_root: str | Path | None = None,
) -> CaseBundle:
    case_id = str(_first(row, "case_id", "id", "uid", "image_id") or "cosmos_case")
    caption = _first(row, "caption", "caption_1", "caption1", "text", "text_1")
    image_path = _resolve_image_path(
        _first(row, "image_path", "image", "img", "file_name", "filename"),
        image_root,
    )
    media_id = f"{case_id}_image"
    return CaseBundle(
        case_id=case_id,
        dataset=DatasetInfo(
            dataset_name="cosmos",
            dataset_split=split or _first(row, "split"),
            native_case_path=image_path,
            native_format="cosmos_pair",
        ),
        task=TaskInfo(
            task_type="out_of_context_detection",
            subtask="image_caption_pair",
            media_type="image",
            expected_output="label_only",
        ),
        input=InputMetadata(
            caption=caption,
            social_media_link=_first(row, "source_url", "url", "image_url"),
            location_hint=_first(row, "location", "place"),
            extra={k: v for k, v in row.items() if k != "label"},
        ),
        claims=_cosmos_claims(case_id, caption or "the caption", media_id),
        media_assets=[
            MediaAsset(
                media_id=media_id,
                case_id=case_id,
                media_type=infer_media_type(Path(image_path or "")),  # type: ignore[arg-type]
                role="primary_claim_media",
                local_path=image_path,
                source_url=_first(row, "image_url", "source_url"),
            )
        ],
        gold=GoldAnnotation(
            gold_final_label=normalize_cosmos_label(_first(row, "label", "gold", "target")),
            gold_visibility="test_hidden" if split == "test" else "train",
            read_gold_before_prediction=False,
        ),
        run_config=RunConfig(
            allow_web_search=False,
            allow_reverse_search=False,
            allow_memory_retrieval=True,
            allow_memory_update=False,
            allow_human_contestation=False,
            expected_output_formats=["json"],
        ),
    )


def build_cosmos_triplet_case(
    row: dict[str, Any],
    split: str | None = None,
    image_root: str | Path | None = None,
) -> CaseBundle:
    bundle = build_cosmos_pair_case(row, split=split, image_root=image_root)
    caption_2 = _first(row, "caption_2", "caption2", "text_2", "alt_caption")
    extra = dict(bundle.input.extra)
    extra["caption_2"] = caption_2
    return bundle.model_copy(
        update={
            "task": bundle.task.model_copy(update={"subtask": "image_caption_triplet"}),
            "input": bundle.input.model_copy(update={"extra": extra}),
            "dataset": bundle.dataset.model_copy(update={"native_format": "cosmos_triplet"}),
            "claims": bundle.claims
            + [
                Claim(
                    claim_id=f"{bundle.case_id}_caption_contradiction",
                    claim_type="caption_context",
                    statement=(
                        "The two captions are mutually consistent with the image "
                        "and with each other."
                    ),
                    media_ids=[bundle.media_assets[0].media_id],
                    expected_evidence_types=["visual", "semantic_contradiction"],
                )
            ],
        }
    )


def _cosmos_claims(case_id: str, caption: str, media_id: str) -> list[Claim]:
    return [
        Claim(
            claim_id=f"{case_id}_main",
            claim_type="main",
            statement="The image correctly depicts the event described in the caption.",
            media_ids=[media_id],
        ),
        Claim(
            claim_id=f"{case_id}_caption_context",
            claim_type="caption_context",
            statement="The caption is contextually consistent with the image.",
            media_ids=[media_id],
        ),
        Claim(
            claim_id=f"{case_id}_what",
            claim_type="what",
            statement=f"The image content matches the event described in: {caption}",
            media_ids=[media_id],
        ),
        Claim(
            claim_id=f"{case_id}_who",
            claim_type="who",
            statement="The people or entities in the image match the caption.",
            media_ids=[media_id],
        ),
        Claim(
            claim_id=f"{case_id}_where",
            claim_type="where",
            statement="The image location matches the caption context.",
            media_ids=[media_id],
        ),
        Claim(
            claim_id=f"{case_id}_when",
            claim_type="when",
            statement="The image time context matches the caption context.",
            media_ids=[media_id],
        ),
    ]


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in {None, ""}:
            return value
    return None


def _resolve_image_path(value: Any, image_root: str | Path | None) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute() or image_root is None:
        return str(path)
    return str(Path(image_root) / path)


def _image_root(path: Path) -> Path | None:
    if path.is_dir():
        images = path / "images"
        return images if images.exists() else path
    images = path.parent / "images"
    return images if images.exists() else path.parent
