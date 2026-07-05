from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.processing.deep_forensics.base import DeepForensicResult
from src.processing.forensic_analyzer import ForensicAnalyzer
from src.schemas.case_schema import MediaItem


def _fake_backend_factory(results):
    class FakeBackend:
        def __init__(self, config):
            pass

        def analyze_images(self, image_paths, output_dir):
            return results(image_paths)

    return FakeBackend


def test_forensic_analyzer_deep_high_score(monkeypatch, tmp_path):
    image = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), "white").save(image)

    def make_results(image_paths):
        return [
            DeepForensicResult(
                target_path=str(image_paths[0]),
                model_name="trufor",
                manipulation_score=0.91,
                anomaly_map_path=str(tmp_path / "map.png"),
                confidence_map_path=str(tmp_path / "conf.png"),
                flags=["deep_forensic_high_manipulation_score"],
            )
        ]

    monkeypatch.setattr(
        "src.processing.deep_forensics.trufor_backend.TruForBackend",
        _fake_backend_factory(make_results),
    )

    analyzer = ForensicAnalyzer(
        {
            "media": {
                "enable_forensic_adapter": True,
                "forensic_engine": "trufor",
                "forensic_manipulation_threshold": 0.50,
            }
        }
    )

    items = analyzer.analyze(
        media=MediaItem(path=str(image), media_type="image"),
        visual_targets=[image],
        output_dir=tmp_path / "forensics",
    )

    assert items[0].source_type == "forensic_analysis"
    assert items[0].metadata["engine"] == "trufor"
    assert items[0].confidence == pytest.approx(0.91)
    assert "deep_forensic_high_manipulation_score" in items[0].uncertainty_flags


def test_forensic_analyzer_deep_low_score_no_high_flag(monkeypatch, tmp_path):
    image = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), "white").save(image)

    def make_results(image_paths):
        return [
            DeepForensicResult(
                target_path=str(image_paths[0]),
                model_name="trufor",
                manipulation_score=0.10,
            )
        ]

    monkeypatch.setattr(
        "src.processing.deep_forensics.trufor_backend.TruForBackend",
        _fake_backend_factory(make_results),
    )

    analyzer = ForensicAnalyzer(
        {"media": {"enable_forensic_adapter": True, "forensic_engine": "trufor"}}
    )

    items = analyzer.analyze(
        media=MediaItem(path=str(image), media_type="image"),
        visual_targets=[image],
        output_dir=tmp_path / "forensics",
    )

    assert items[0].source_type == "forensic_analysis"
    assert "deep_forensic_high_manipulation_score" not in items[0].uncertainty_flags


def test_forensic_analyzer_deep_backend_unavailable_falls_back_to_basic(monkeypatch, tmp_path):
    image = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), "white").save(image)

    class BrokenBackend:
        def __init__(self, config):
            raise FileNotFoundError("no repo")

    monkeypatch.setattr(
        "src.processing.deep_forensics.trufor_backend.TruForBackend",
        BrokenBackend,
    )

    analyzer = ForensicAnalyzer(
        {
            "media": {
                "enable_forensic_adapter": True,
                "forensic_engine": "trufor",
                "forensic_fallback_to_basic": True,
            }
        }
    )

    items = analyzer.analyze(
        media=MediaItem(path=str(image), media_type="image"),
        visual_targets=[image],
        output_dir=tmp_path / "forensics",
    )

    assert items[0].source_type == "forensic_analysis"
    assert "deep_forensic_backend_unavailable" in items[0].uncertainty_flags
    assert items[0].metadata.get("deep_forensic_fallback") is True


def test_forensic_analyzer_deep_backend_unavailable_no_fallback(monkeypatch, tmp_path):
    image = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), "white").save(image)

    class BrokenBackend:
        def __init__(self, config):
            raise FileNotFoundError("no repo")

    monkeypatch.setattr(
        "src.processing.deep_forensics.trufor_backend.TruForBackend",
        BrokenBackend,
    )

    analyzer = ForensicAnalyzer(
        {
            "media": {
                "enable_forensic_adapter": True,
                "forensic_engine": "trufor",
                "forensic_fallback_to_basic": False,
            }
        }
    )

    items = analyzer.analyze(
        media=MediaItem(path=str(image), media_type="image"),
        visual_targets=[image],
        output_dir=tmp_path / "forensics",
    )

    assert items[0].source_type == "synthetic_uncertainty"
    assert "deep_forensic_backend_unavailable" in items[0].uncertainty_flags


def test_forensic_analyzer_deep_no_valid_targets(tmp_path):
    analyzer = ForensicAnalyzer(
        {"media": {"enable_forensic_adapter": True, "forensic_engine": "trufor"}}
    )

    missing_image = tmp_path / "missing.jpg"
    items = analyzer.analyze(
        media=MediaItem(path=str(missing_image), media_type="image"),
        visual_targets=[missing_image],
        output_dir=tmp_path / "forensics",
    )

    assert items[0].source_type == "synthetic_uncertainty"
    assert "deep_forensic_no_valid_targets" in items[0].uncertainty_flags


def test_forensic_analyzer_basic_engine_still_works(tmp_path):
    image = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), "white").save(image)

    analyzer = ForensicAnalyzer(
        {"media": {"enable_forensic_adapter": True, "forensic_engine": "basic"}}
    )

    items = analyzer.analyze(
        media=MediaItem(path=str(image), media_type="image"),
        visual_targets=[image],
        output_dir=tmp_path / "forensics",
    )

    assert items[0].source_type == "forensic_analysis"
    assert items[0].title == "Basic forensic analysis"
