from __future__ import annotations

from pathlib import Path

from src.schemas.case_bundle_schema import CaseBundle
from src.utils.io import write_json


def write_canonical_bundle(
    bundle: CaseBundle,
    canonical_root: str | Path = "data/canonical",
) -> Path:
    root = Path(canonical_root)
    if not root.is_absolute():
        from src.utils.io import project_root

        root = project_root() / root
    case_dir = (
        root / bundle.case_id
        if root.name == bundle.dataset.dataset_name
        else root / bundle.dataset.dataset_name / bundle.case_id
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = case_dir / "case_bundle.json"
    write_json(bundle_path, bundle)
    write_json(
        case_dir / "adapter_log.json",
        {
            "adapter_version": bundle.dataset.adapter_version,
            "native_case_path": bundle.dataset.native_case_path,
            "gold_read_before_prediction": bundle.gold.read_gold_before_prediction,
        },
    )
    return bundle_path
