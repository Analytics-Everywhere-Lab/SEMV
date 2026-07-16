from __future__ import annotations

import json

from src.evaluation.cosmos_evaluator import evaluate_cosmos

from tests.conftest import FakeLLMClient


def test_cosmos_evaluator_writes_metrics(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "img.jpg").write_bytes(b"not really an image")
    metadata = tmp_path / "test.jsonl"
    metadata.write_text(
        json.dumps({"case_id": "c1", "image_path": "img.jpg", "caption": "A caption", "label": "nooc"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "eval"

    result = evaluate_cosmos(metadata, image_root, output, llm_client=FakeLLMClient())

    assert result["dataset"] == "cosmos"
    assert result["aggregate_metrics"]["dataset"] == "cosmos"
    assert "memory_metrics" in result
    assert result["memory_metrics"]["negative_transfer_rate"] is None
    assert (output / "predictions.jsonl").exists()
    assert (output / "aggregate_metrics.json").exists()
