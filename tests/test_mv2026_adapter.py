from __future__ import annotations

import json

from src.ingestion.mv2026_adapter import MV2026Adapter
from src.evaluation.mv2026_evaluator import _discover_case_dirs


def test_multiple_media_assets_are_loaded(tmp_path):
    case_dir = tmp_path / "ID407"
    media_dir = case_dir / "input" / "media"
    media_dir.mkdir(parents=True)
    (case_dir / "output").mkdir()
    (case_dir / "input" / "ID407.json").write_text(
        json.dumps({"title": "Claim title", "description": "Desc", "location": "Halifax"}),
        encoding="utf-8",
    )
    (media_dir / "a.jpg").write_bytes(b"not really an image")
    (media_dir / "b.mp4").write_bytes(b"not really a video")
    (case_dir / "output" / "report.md").write_text("# Final Verification Status\nStatus: verified", encoding="utf-8")

    bundle = MV2026Adapter().load(case_dir, split="validation")

    assert len(bundle.media_assets) == 2
    assert any(media.role == "primary_claim_media" for media in bundle.media_assets)
    assert bundle.gold.gold_report_available is True
    assert bundle.gold.read_gold_before_prediction is False
    assert not bundle.provided_evidence


def test_nested_mv2026_training_cases_are_discovered(tmp_path):
    nested_case = tmp_path / "training" / "ID333" / "ID333"
    input_dir = nested_case / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "ID333.json").write_text(json.dumps({"title": "Claim title"}), encoding="utf-8")

    direct_case = tmp_path / "ID407"
    direct_input = direct_case / "input"
    direct_input.mkdir(parents=True)
    (direct_input / "ID407.json").write_text(json.dumps({"title": "Other title"}), encoding="utf-8")

    assert _discover_case_dirs(tmp_path) == [direct_case, nested_case]
    assert _discover_case_dirs(nested_case) == [nested_case]


def test_mv2026_gold_is_loaded_only_by_post_prediction_provider(tmp_path, monkeypatch):
    import src.evaluation.mv2026_evaluator as evaluator
    from src.schemas.report_schema import VerificationReport

    case_dir = tmp_path / "ID900"
    (case_dir / "input" / "media").mkdir(parents=True)
    (case_dir / "output").mkdir()
    (case_dir / "input" / "ID900.json").write_text(
        json.dumps({"title": "A claim with no gold text in the input."}), encoding="utf-8"
    )
    sentinel = "out of context"
    (case_dir / "output" / "report.md").write_text(
        f"# Final Verification Status\nStatus: {sentinel}", encoding="utf-8"
    )
    observed = {}

    def fake_run(bundle, **kwargs):
        assert bundle.gold.gold_final_label is None
        observed["provider_present"] = kwargs.get("post_prediction_supervision_provider") is not None
        observed["gold_after_prediction"] = kwargs["post_prediction_supervision_provider"]()
        return VerificationReport(case_id=bundle.case_id, final_status="false_context", final_confidence=0.9)

    monkeypatch.setattr(evaluator, "run_case_bundle", fake_run)
    result = evaluator.evaluate_mv2026(
        raw_root=case_dir,
        output_dir=tmp_path / "results",
        split="train",
        update_memory=True,
    )
    assert observed == {"provider_present": True, "gold_after_prediction": "false_context"}
    assert result["accuracy"] == 1.0
    assert result["aggregate_metrics"]["accuracy"] == 1.0
    assert "memory_metrics" in result
    assert result["memory_metrics"]["negative_transfer_rate"] is None


def test_normalized_label_aliases_do_not_create_false_failures():
    from src.reflection.failure_classifier import FailureClassifier
    from src.schemas.report_schema import VerificationReport

    report = VerificationReport(case_id="case", final_status="false_context", final_confidence=0.9)
    modes = FailureClassifier().classify(report, "Out of context", None)
    assert "final_label_mismatch" not in modes
    assert "successful_strategy" in modes
