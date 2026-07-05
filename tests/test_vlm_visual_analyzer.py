from __future__ import annotations

from PIL import Image

from src.processing.vlm_visual_analyzer import VLMVisualAnalyzer


def test_vlm_visual_analyzer_with_mock_vllm(tmp_path, monkeypatch):
    image = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(image)
    analyzer = VLMVisualAnalyzer({"media": {"enable_vlm_adapter": True, "vlm_provider": "vllm", "vlm_model": "Qwen/Qwen3.5-9B"}})
    monkeypatch.setattr(analyzer, "_vllm_generate", lambda path, claim, context: {"scene_summary": "A city street", "search_queries": ["city street"]})
    items = analyzer.analyze(image_paths=[image], case_id="c")
    assert any(item.source_type == "visual_caption" for item in items)
