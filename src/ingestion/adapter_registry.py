from __future__ import annotations

from pathlib import Path

from src.ingestion.base_adapter import BaseDatasetAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self.adapters: list[BaseDatasetAdapter] = []

    def register(self, adapter: BaseDatasetAdapter) -> None:
        self.adapters.append(adapter)

    def get_adapter(
        self, case_path: str | Path, adapter_name: str | None = None
    ) -> BaseDatasetAdapter:
        path = Path(case_path)
        if adapter_name and adapter_name != "auto":
            for adapter in self.adapters:
                if adapter.adapter_name == adapter_name:
                    return adapter
            raise ValueError(f"Adapter not found: {adapter_name}")
        for adapter in self.adapters:
            if adapter.can_load(path):
                return adapter
        raise ValueError(f"No adapter can load case path: {path}")


def default_registry() -> AdapterRegistry:
    from src.ingestion.cosmos_adapter import COSMOSAdapter
    from src.ingestion.image_caption_adapter import ImageCaptionAdapter
    from src.ingestion.mv2026_adapter import MV2026Adapter
    from src.ingestion.report_style_adapter import ReportStyleAdapter

    registry = AdapterRegistry()
    registry.register(MV2026Adapter())
    registry.register(COSMOSAdapter())
    registry.register(ImageCaptionAdapter())
    registry.register(ReportStyleAdapter())
    return registry
