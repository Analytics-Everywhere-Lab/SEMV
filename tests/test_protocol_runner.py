from __future__ import annotations

from src.evaluation.protocol_runner import ABLATION_VARIANTS, run_protocol


def test_protocol_runner_lists_ablations_without_datasets(tmp_path):
    config = tmp_path / "evaluation.yaml"
    config.write_text("evaluation:\n  datasets: {}\n  protocol:\n    name: ablations\n", encoding="utf-8")
    result = run_protocol(config, protocol="ablations", output_dir=tmp_path, ablation_variant="A0")

    assert "A0" in ABLATION_VARIANTS
    assert result["variant_order"] == ["A0"]
    assert result["variants"]["A0"]["feature_configuration"]["use_memory"] is False
    assert (tmp_path / "ablations" / "A0" / "variant_results.json").exists()
