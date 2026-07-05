from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from src.processing.forensic_analyzer import ForensicAnalyzer
from src.schemas.case_schema import MediaItem
from src.utils.env_loader import project_root

_REPO_DIR = os.getenv("SEMV_TRUFOR_REPO_DIR")
_WEIGHTS = os.getenv("SEMV_TRUFOR_WEIGHTS")


def _real_trufor_available() -> bool:
    if not _REPO_DIR or not _WEIGHTS:
        return False
    root = project_root()
    repo_dir = Path(_REPO_DIR)
    repo_dir = repo_dir if repo_dir.is_absolute() else root / repo_dir
    weights = Path(_WEIGHTS)
    weights = weights if weights.is_absolute() else root / weights
    return repo_dir.exists() and (repo_dir / "test.py").exists() and weights.exists()


@pytest.mark.skipif(
    not _real_trufor_available(),
    reason="Real TruFor checkout/weights not configured (SEMV_TRUFOR_REPO_DIR / SEMV_TRUFOR_WEIGHTS)",
)
def test_trufor_real_backend_smoke(tmp_path):
    image_path = tmp_path / "smoke.jpg"
    Image.new("RGB", (128, 128), "white").save(image_path)

    analyzer = ForensicAnalyzer(
        {
            "media": {
                "enable_forensic_adapter": True,
                "forensic_engine": "trufor",
                "forensic_deep_backend": "trufor",
                "forensic_device": os.getenv("SEMV_FORENSIC_DEVICE", "cpu"),
                "forensic_external_repo_dir": _REPO_DIR,
                "forensic_trufor_weights": _WEIGHTS,
                "forensic_trufor_python": os.getenv("SEMV_TRUFOR_PYTHON", "python"),
                "forensic_trufor_experiment": os.getenv("SEMV_TRUFOR_EXPERIMENT", "trufor_ph3"),
                "forensic_save_maps": True,
                "forensic_fallback_to_basic": False,
            }
        }
    )

    items = analyzer.analyze(
        media=MediaItem(path=str(image_path), media_type="image"),
        visual_targets=[image_path],
        output_dir=tmp_path / "forensics",
    )

    assert len(items) == 1
    item = items[0]
    assert item.source_type == "forensic_analysis"
    assert item.metadata["engine"] == "trufor"
    assert item.metadata["max_manipulation_score"] is not None

    for path in item.metadata["anomaly_map_paths"]:
        assert Path(path).exists()
    for path in item.metadata["confidence_map_paths"]:
        assert Path(path).exists()
