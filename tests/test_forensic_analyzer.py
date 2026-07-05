from __future__ import annotations

from PIL import Image

from src.processing.forensic_analyzer import ForensicAnalyzer
from src.schemas.case_schema import MediaItem


def test_forensic_analyzer_basic_image(tmp_path):
    image = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), "white").save(image)
    items = ForensicAnalyzer({"media": {"enable_forensic_adapter": True, "forensic_engine": "basic"}}).analyze(
        media=MediaItem(path=str(image), media_type="image"),
        visual_targets=[image],
        output_dir=tmp_path / "forensics",
    )
    assert items[0].source_type == "forensic_analysis"
    assert (tmp_path / "forensics").exists()
