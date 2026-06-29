from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.schemas.case_bundle_schema import CaseBundle


class BaseDatasetAdapter(ABC):
    adapter_name: str
    adapter_version: str = "1.0"

    @abstractmethod
    def can_load(self, case_path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load(self, case_path: Path, split: str | None = None) -> CaseBundle:
        raise NotImplementedError
