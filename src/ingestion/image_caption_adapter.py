from __future__ import annotations

from pathlib import Path

from src.ingestion.base_adapter import BaseDatasetAdapter
from src.ingestion.cosmos_adapter import build_cosmos_pair_case
from src.schemas.case_bundle_schema import CaseBundle
from src.utils.io import read_json


class ImageCaptionAdapter(BaseDatasetAdapter):
    adapter_name = "image_caption"

    def can_load(self, case_path: Path) -> bool:
        return case_path.suffix.lower() == ".json" and case_path.exists()

    def load(self, case_path: Path, split: str | None = None) -> CaseBundle:
        row = read_json(case_path)
        bundle = build_cosmos_pair_case(row, split=split, image_root=case_path.parent)
        return bundle.model_copy(
            update={
                "dataset": bundle.dataset.model_copy(
                    update={
                        "dataset_name": row.get("dataset_name", "image_caption"),
                        "native_case_path": str(case_path),
                        "native_format": "image_caption_json",
                    }
                )
            }
        )
