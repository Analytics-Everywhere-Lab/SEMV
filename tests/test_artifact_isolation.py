from __future__ import annotations

import json

from src.evaluation.cosmos_evaluator import evaluate_cosmos
from src.schemas.report_schema import VerificationReport


def test_same_case_conditions_and_output_roots_do_not_overwrite(tmp_path, monkeypatch):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    metadata = tmp_path / "cases.jsonl"
    metadata.write_text(json.dumps({
        "case_id": "same_case", "image_path": "image.jpg", "caption": "caption", "label": 0,
    }) + "\n", encoding="utf-8")
    calls = []

    def fake_run(bundle, **kwargs):
        root = kwargs["artifact_root"]
        case_dir = root / bundle.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        marker = "memory_on" if bundle.run_config.allow_memory_retrieval else "memory_off"
        (case_dir / "complete.txt").write_text(marker, encoding="utf-8")
        calls.append(case_dir)
        return VerificationReport(case_id=bundle.case_id, final_status="verified", final_confidence=0.9)

    monkeypatch.setattr("src.evaluation.cosmos_evaluator.run_case_bundle", fake_run)
    on_root = tmp_path / "evaluation_one"
    off_root = tmp_path / "evaluation_two"
    evaluate_cosmos(metadata, tmp_path, on_root, split="test", allow_memory_retrieval=True)
    evaluate_cosmos(metadata, tmp_path, off_root, split="test", allow_memory_retrieval=False)
    on_artifact = on_root / "cases" / "same_case" / "complete.txt"
    off_artifact = off_root / "cases" / "same_case" / "complete.txt"
    assert on_artifact.read_text(encoding="utf-8") == "memory_on"
    assert off_artifact.read_text(encoding="utf-8") == "memory_off"
    assert calls[0] != calls[1]
    assert (on_root / "per_case_metrics.jsonl").exists()
    assert (off_root / "per_case_metrics.jsonl").exists()
