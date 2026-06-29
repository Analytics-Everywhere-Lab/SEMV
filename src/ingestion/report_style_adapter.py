from __future__ import annotations

from pathlib import Path

from src.ingestion.base_adapter import BaseDatasetAdapter
from src.ingestion.mv2026_adapter import MV2026Adapter
from src.schemas.case_bundle_schema import CaseBundle


class ReportStyleAdapter(BaseDatasetAdapter):
    adapter_name = "report_style"

    def can_load(self, case_path: Path) -> bool:
        return (case_path / "input").exists() and (case_path / "output").exists()

    def load(self, case_path: Path, split: str | None = None) -> CaseBundle:
        if MV2026Adapter().can_load(case_path):
            return MV2026Adapter().load(case_path, split=split)
        raise ValueError(f"Unsupported report-style case: {case_path}")
