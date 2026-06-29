from __future__ import annotations

from src.evaluation.protocol_runner import ABLATION_VARIANTS, run_protocol


def test_protocol_runner_lists_ablations_without_datasets(tmp_path):
    result = run_protocol(protocol="ablations", output_dir=tmp_path)

    assert "A0" in ABLATION_VARIANTS
    assert "ablations" in result
