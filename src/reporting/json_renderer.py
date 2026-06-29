from __future__ import annotations

from pathlib import Path

from src.schemas.report_schema import VerificationReport
from src.utils.io import write_json


class JSONRenderer:
    def render_to_file(self, report: VerificationReport, path: str | Path) -> None:
        write_json(path, report)
