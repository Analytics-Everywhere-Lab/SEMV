from __future__ import annotations

import pytest

from src.evaluation.protocol_runner import ABLATION_VARIANTS, run_protocol
from src.schemas.claim_schema import SubClaim
from src.schemas.case_schema import MultimediaCase
from src.schemas.memory_schema import MemoryRecord
from tests.memory_test_utils import make_service


def _snapshot(tmp_path, records=None):
    service = make_service(tmp_path / "source", retrieval={"min_similarity": 0.0})
    for record in records or []:
        service.store.append(record)
    return service.snapshot("known")


def _config(tmp_path, snapshot, protocol="static", extra=""):
    path = tmp_path / f"{protocol}.yaml"
    path.write_text(f"""
evaluation:
  memory_config: configs/memory.yaml
  datasets:
    mv2026:
      enabled: true
      raw_root: unused
      split: validation
  protocol:
    name: {protocol}
    frozen_memory_snapshot: {snapshot}
    allow_memory_retrieval: true
{extra}
""", encoding="utf-8")
    return path


def test_static_uses_known_snapshot_and_preserves_files(tmp_path, monkeypatch):
    record = MemoryRecord(memory_id="mem_known", memory_type="episodic",
                          text="known temporal verification lesson", claim_type="what", status="active", origin="manual")
    snapshot = _snapshot(tmp_path, [record])
    before = {path.name: path.read_bytes() for path in snapshot.iterdir() if path.is_file()}

    def fake_evaluate(**kwargs):
        active = kwargs["memory_service"].store.load_long_term(statuses=["active"])
        assert [row.memory_id for row in active] == ["mem_known"]
        retrieved = kwargs["memory_service"].retrieve(
            MultimediaCase(case_id="static", claim="known temporal verification lesson"),
            SubClaim(claim_id="c1", claim_type="what", statement="known temporal verification lesson"),
            [], top_k=5,
        )
        assert [row.memory_id for row in retrieved] == ["mem_known"]
        active = retrieved
        assert kwargs["artifact_root"].as_posix().endswith("memory_on/cases")
        return {"retrieved_memory_ids": [row.memory_id for row in active]}

    monkeypatch.setattr("src.evaluation.protocol_runner.evaluate_mv2026", fake_evaluate)
    result = run_protocol(_config(tmp_path, snapshot), output_dir=tmp_path / "out")
    frozen = result["frozen_memory"]
    assert result["runs"]["mv2026_validation"]["retrieved_memory_ids"] == ["mem_known"]
    assert frozen["pre_run_hash"] == frozen["post_run_hash"]
    assert frozen["manifest_hash"] == frozen["post_manifest_hash"]
    assert frozen["active_memory_count"] == 1
    assert before == {path.name: path.read_bytes() for path in snapshot.iterdir() if path.is_file()}


@pytest.mark.parametrize("source_kind", ["missing", "invalid"])
def test_static_rejects_missing_or_invalid_snapshot(tmp_path, source_kind):
    source = tmp_path / "missing"
    if source_kind == "invalid":
        source.mkdir()
        (source / "manifest.json").write_text('{"state_hash":"wrong"}', encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError), match="snapshot|hash|manifest"):
        run_protocol(_config(tmp_path, source), output_dir=tmp_path / "out")


def test_static_empty_snapshot_requires_explicit_opt_in(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="no active"):
        run_protocol(_config(tmp_path, snapshot), output_dir=tmp_path / "denied")
    monkeypatch.setattr("src.evaluation.protocol_runner.evaluate_mv2026", lambda **kwargs: {"ok": True})
    result = run_protocol(_config(tmp_path, snapshot, extra="    allow_empty_memory: true"),
                          output_dir=tmp_path / "allowed")
    assert result["frozen_memory"]["active_memory_count"] == 0


def test_all_ablations_execute_with_isolated_features_and_artifacts(tmp_path, monkeypatch):
    records = [MemoryRecord(memory_id=f"m_{kind}", memory_type=kind,
                            text=f"{kind} lesson", status="active")
               for kind in ("episodic", "failure", "semantic_rule")]
    snapshot = _snapshot(tmp_path, records)
    calls = []

    def fake_evaluate(**kwargs):
        runtime = kwargs["runtime_config"]
        calls.append((runtime.features, kwargs["allow_memory_retrieval"], kwargs["artifact_root"]))
        kwargs["artifact_root"].mkdir(parents=True, exist_ok=True)
        (kwargs["artifact_root"] / "executed.txt").write_text("executed", encoding="utf-8")
        return {"executed": True, "features": runtime.features.model_dump(mode="json")}

    monkeypatch.setattr("src.evaluation.protocol_runner.evaluate_mv2026", fake_evaluate)
    result = run_protocol(_config(tmp_path, snapshot, protocol="ablations"), output_dir=tmp_path / "out")
    assert result["variant_order"] == list(ABLATION_VARIANTS)
    assert len(calls) == 11
    for variant, variant_result in result["variants"].items():
        assert variant_result["runs"]["mv2026_validation"]["executed"] is True
        assert (tmp_path / "out" / "ablations" / variant / "variant_results.json").exists()
        assert (tmp_path / "out" / "ablations" / variant / "mv2026" / variant / "cases" / "executed.txt").exists()
    assert calls[0][0].use_memory is False and calls[0][1] is False
    assert calls[2][0].memory_types == ("episodic",)
    assert calls[3][0].memory_types == ("semantic_rule",)
    assert calls[4][0].memory_types == ("failure",)
    assert calls[5][0].argument_verifier is False
    assert calls[7][0].clash_resolution is False
    assert calls[0][0].use_qbaf is False
    assert (tmp_path / "out" / "ablation_comparison.json").exists()
