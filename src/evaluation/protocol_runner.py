from __future__ import annotations

from pathlib import Path

from src.evaluation.cosmos_evaluator import evaluate_cosmos
from src.evaluation.mv2026_evaluator import evaluate_mv2026
from src.memory.seed_memory import seed_semantic_rules
from src.utils.io import project_root, read_yaml, write_json

ABLATION_VARIANTS = {
    "A0": {"name": "No memory, no QBAF", "use_memory": False, "use_qbaf": False},
    "A1": {"name": "QBAF only", "use_memory": False, "use_qbaf": True},
    "A2": {"name": "QBAF + episodic memory", "memory_types": ["episodic"]},
    "A3": {"name": "QBAF + semantic memory", "memory_types": ["semantic_rule"]},
    "A4": {"name": "QBAF + failure memory", "memory_types": ["failure"]},
    "A5": {"name": "QBAF + all memory, no Verify Agent", "argument_verifier": False},
    "A6": {"name": "QBAF + all memory + Verify Agent", "argument_verifier": True},
    "A7": {"name": "Full system without clash resolution", "clash_resolution": False},
    "A8": {"name": "Full system with clash resolution", "clash_resolution": True},
    "A9": {"name": "Full system without argument verifier", "argument_verifier": False},
    "A10": {"name": "Full system with argument verifier", "argument_verifier": True},
}


def run_protocol(
    config_path: str | Path = "configs/evaluation.yaml",
    protocol: str | None = None,
    output_dir: str | Path | None = None,
    llm_client=None,
) -> dict:
    config = read_yaml(config_path)
    evaluation = config.get("evaluation", {})
    datasets = evaluation.get("datasets", {})
    selected = protocol or evaluation.get("protocol", {}).get("name", "static")
    out = _resolve(output_dir or "data/outputs/evaluation/joint_mv_cosmos")
    out.mkdir(parents=True, exist_ok=True)
    seed_semantic_rules()

    results = {"protocol": selected, "runs": {}}
    if selected in {"static", "prequential", "train_memory_freeze_test", "mv2026_to_cosmos_transfer"}:
        mv_cfg = datasets.get("mv2026", {})
        cosmos_cfg = datasets.get("cosmos", {})
        if mv_cfg.get("enabled", True):
            results["runs"]["mv2026"] = evaluate_mv2026(
                raw_root=mv_cfg.get("raw_root", "data/raw/mv2026"),
                output_dir=out / "mv2026",
                protocol=selected,
                split=mv_cfg.get("split", "validation"),
                llm_client=llm_client,
            )
        if cosmos_cfg.get("enabled", True):
            results["runs"]["cosmos"] = evaluate_cosmos(
                cosmos_metadata=cosmos_cfg.get("metadata", "data/raw/cosmos/test.jsonl"),
                image_root=cosmos_cfg.get("image_root", "data/raw/cosmos/images"),
                output_dir=out / "cosmos",
                mode=cosmos_cfg.get("mode", "closed_world"),
                split=cosmos_cfg.get("split", "test"),
                llm_client=llm_client,
            )
    elif selected == "ablations":
        results["ablations"] = ABLATION_VARIANTS
    else:
        raise ValueError(f"Unsupported protocol: {selected}")
    write_json(out / "protocol_results.json", results)
    return results


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else project_root() / target
