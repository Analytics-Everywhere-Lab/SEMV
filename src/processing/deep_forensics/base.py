from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DeepForensicResult:
    target_path: str
    model_name: str
    manipulation_score: float | None = None
    anomaly_map_path: str | None = None
    confidence_map_path: str | None = None
    heatmap_overlay_path: str | None = None
    manipulated_area_ratio: float | None = None
    max_anomaly: float | None = None
    mean_anomaly: float | None = None
    mean_confidence: float | None = None
    flags: list[str] = field(default_factory=list)
    raw_output: dict[str, Any] = field(default_factory=dict)


class DeepForensicBackend:
    model_name: str = "unknown"

    def analyze_images(
        self,
        image_paths: list[Path],
        output_dir: Path,
    ) -> list[DeepForensicResult]:
        raise NotImplementedError
