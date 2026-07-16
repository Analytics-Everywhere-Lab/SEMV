from __future__ import annotations

import logging
from pathlib import Path

from src.evaluation.cosmos_evaluator import evaluate_cosmos
from src.evaluation.mv2026_evaluator import evaluate_mv2026
from src.memory.memory_service import MemoryService
from src.memory.seed_memory import seed_semantic_rules
from src.utils.io import project_root, read_yaml, write_json


logger = logging.getLogger("run_case")

ABLATION_VARIANTS = {
    "A0": {"name": "No memory, no A-QBAF", "use_memory": False, "use_qbaf": False},
    "A1": {"name": "A-QBAF only", "use_memory": False, "use_qbaf": True},
    "A2": {"name": "A-QBAF + episodic memory", "memory_types": ["episodic"]},
    "A3": {"name": "A-QBAF + semantic memory", "memory_types": ["semantic_rule"]},
    "A4": {"name": "A-QBAF + failure memory", "memory_types": ["failure"]},
    "A5": {"name": "A-QBAF + all memory, no Verify Agent", "argument_verifier": False},
    "A6": {"name": "A-QBAF + all memory + Verify Agent", "argument_verifier": True},
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
    resume: bool | None = None,
) -> dict:
    config = read_yaml(config_path)
    evaluation = config.get("evaluation", {})
    datasets = evaluation.get("datasets", {})
    protocol_cfg = evaluation.get("protocol", {})
    selected = protocol or protocol_cfg.get("name", "static")
    out = _resolve(output_dir or "data/outputs/evaluation/joint_mv_cosmos")
    out.mkdir(parents=True, exist_ok=True)
    resume_run = protocol_cfg.get("resume", False) if resume is None else resume
    _ensure_clean_memory_dir(out / "memory", resume=bool(resume_run))
    gold_stage = protocol_cfg.get("gold_reading_stage")
    if gold_stage not in {None, "post_prediction", "after_prediction_only"}:
        raise ValueError("gold_reading_stage must be post_prediction when configured.")
    if protocol_cfg.get("allow_memory_update_test", False):
        raise ValueError("allow_memory_update_test=true is forbidden for validation/test phases.")

    memory_config_path = evaluation.get("memory_config", "configs/memory.yaml")

    if selected == "static":
        results = _run_static(datasets, protocol_cfg, out, llm_client, memory_config_path)
    elif selected == "prequential":
        results = _run_prequential(datasets, protocol_cfg, out, llm_client, memory_config_path)
    elif selected == "train_memory_freeze_test":
        results = _run_train_memory_freeze_test(
            datasets, protocol_cfg, out, llm_client, memory_config_path
        )
    elif selected == "mv2026_to_cosmos_transfer":
        results = _run_transfer(datasets, protocol_cfg, out, llm_client, memory_config_path)
    elif selected == "ablations":
        results = {"protocol": selected, "ablations": ABLATION_VARIANTS}
    else:
        raise ValueError(f"Unsupported protocol: {selected}")

    write_json(out / "protocol_results.json", results)
    return results


# ------------------------------------------------------------------ protocols


def _run_static(datasets, protocol_cfg, out: Path, llm_client, memory_config_path) -> dict:
    """Frozen memory, retrieval optional, no updates."""
    service = MemoryService.from_config_path(
        memory_config_path,
        memory_dir=out / "memory",
        llm_client=llm_client,
        frozen=True,
        usage_log_path=out / "memory_usage_events.jsonl",
    )
    results = {"protocol": "static", "runs": {}}
    _evaluate_datasets(
        datasets,
        out,
        llm_client,
        memory_service=service,
        allow_memory_retrieval=protocol_cfg.get("allow_memory_retrieval", True),
        update_memory=False,
        results=results,
    )
    return results


def _run_prequential(datasets, protocol_cfg, out: Path, llm_client, memory_config_path) -> dict:
    """For each training case: predict with memory from previous cases, evaluate,
    reveal gold, stage reflection, and consolidate on schedule. The current case
    never learns from its own gold before prediction (bootstrap_memory mode
    reflects only after the report exists)."""
    memory_dir = out / "memory"
    service = MemoryService.from_config_path(
        memory_config_path,
        memory_dir=memory_dir,
        llm_client=llm_client,
    )
    _apply_schedule_override(service, protocol_cfg)
    seed_semantic_rules(store=service.store)
    results = {"protocol": "prequential", "runs": {}, "memory_dir": str(memory_dir)}
    train_cfg = _train_phase_config(datasets, protocol_cfg)
    results["runs"]["train_{}_{}".format(train_cfg["dataset"], train_cfg.get("split") or "default")] = _evaluate_phase(
        train_cfg,
        out / "prequential",
        llm_client,
        memory_service=service,
        update_memory=protocol_cfg.get("allow_memory_update_train", True),
    )
    final = service.consolidate()
    results["final_consolidation"] = {
        "promoted": final.promoted,
        "merged": final.merged,
        "conflicted": final.conflicted,
    }
    results["memory_counts"] = service.store.counts()
    return results


def _run_train_memory_freeze_test(
    datasets, protocol_cfg, out: Path, llm_client, memory_config_path
) -> dict:
    """1) isolated run-specific memory dir, 2) seed rules, 3) deterministic
    training with retrieval+updates, 4) consolidate every N cases, 5) force final
    consolidation, 6) frozen snapshot with manifest and state hash, 7) frozen
    validation/test, 8) assert the state hash is unchanged."""
    memory_dir = out / "memory"
    train_service = MemoryService.from_config_path(
        memory_config_path,
        memory_dir=memory_dir,
        llm_client=llm_client,
    )
    _apply_schedule_override(train_service, protocol_cfg)
    seed_semantic_rules(store=train_service.store)

    results: dict = {"protocol": "train_memory_freeze_test", "runs": {}, "memory_dir": str(memory_dir)}

    train_cfg = _train_phase_config(datasets, protocol_cfg)
    results["runs"]["train_{}_{}".format(train_cfg["dataset"], train_cfg.get("split") or "default")] = _evaluate_phase(
        train_cfg,
        out / "train",
        llm_client,
        memory_service=train_service,
        update_memory=protocol_cfg.get("allow_memory_update_train", True),
    )

    final = train_service.consolidate()
    results["final_consolidation"] = {
        "promoted": final.promoted,
        "merged": final.merged,
        "conflicted": final.conflicted,
        "under_review": final.under_review,
    }

    snapshot_path = train_service.snapshot("frozen")
    frozen_hash = train_service.state_hash()
    results["snapshot_path"] = str(snapshot_path)
    results["state_hash"] = frozen_hash

    frozen_service = MemoryService.from_config_path(
        memory_config_path,
        memory_dir=snapshot_path,
        llm_client=llm_client,
        frozen=True,
        usage_log_path=out / "eval" / "memory_usage_events.jsonl",
    )

    paired_baselines = {}
    for phase in _eval_phase_configs(datasets, protocol_cfg):
        phase_key = f"{phase['dataset']}_{phase.get('split') or 'default'}"
        phase_out = out / "eval" / phase_key
        baseline_case_metrics = None
        if protocol_cfg.get("run_paired_memory_off_baseline", False):
            baseline_phase = {**phase, "allow_memory_retrieval": False}
            baseline_result = _evaluate_phase(
                baseline_phase,
                out / "eval_memory_off" / phase_key,
                llm_client,
                memory_service=frozen_service,
                update_memory=False,
                include_case_metrics=True,
            )
            baseline_case_metrics = baseline_result.pop("_case_metrics", None)
            paired_baselines[phase_key] = baseline_result
        results["runs"][f"eval_{phase_key}"] = _evaluate_phase(
            phase,
            phase_out,
            llm_client,
            memory_service=frozen_service,
            update_memory=False,
            paired_baseline_case_metrics=baseline_case_metrics,
        )
    if paired_baselines:
        results["paired_memory_off_baselines"] = paired_baselines


    post_hash = frozen_service.state_hash()
    results["state_hash_after_eval"] = post_hash
    if post_hash != frozen_hash:
        raise RuntimeError(
            "Frozen memory snapshot changed during validation/test: "
            f"{frozen_hash} -> {post_hash}"
        )
    results["memory_state_unchanged"] = True
    return results


def _run_transfer(datasets, protocol_cfg, out: Path, llm_client, memory_config_path) -> dict:
    """Build memory only from the configured MV2026 training split, freeze it,
    and evaluate COSMOS without updates."""
    transfer_cfg = dict(protocol_cfg)
    transfer_cfg["train"] = {**dict(transfer_cfg.get("train") or {}), "dataset": "mv2026", "split": "train"}
    transfer_cfg["eval"] = [
        phase
        for phase in _eval_phase_configs(datasets, protocol_cfg)
        if phase["dataset"] == "cosmos"
    ] or [{"dataset": "cosmos", "split": datasets.get("cosmos", {}).get("split", "test")}]
    result = _run_train_memory_freeze_test(datasets, transfer_cfg, out, llm_client,
                                           memory_config_path)
    result["protocol"] = "mv2026_to_cosmos_transfer"
    return result


# ------------------------------------------------------------------- helpers


def _apply_schedule_override(service: MemoryService, protocol_cfg: dict) -> None:
    every_n = protocol_cfg.get("consolidate_every_n_cases")
    if every_n:
        service.config = service.config.model_copy(
            update={
                "consolidation": service.config.consolidation.model_copy(
                    update={"every_n_cases": int(every_n)}
                )
            }
        )
        service.consolidator.config = service.config
        service.store.config = service.config


def _train_phase_config(datasets: dict, protocol_cfg: dict) -> dict:
    phase = dict(protocol_cfg.get("train") or {})
    phase.setdefault("dataset", "mv2026")
    dataset_cfg = datasets.get(phase["dataset"], {})
    phase.setdefault("split", dataset_cfg.get("train_split", "train"))
    phase.setdefault("raw_root", dataset_cfg.get("raw_root", f"data/raw/{phase['dataset']}"))
    phase.setdefault("metadata", dataset_cfg.get("train_metadata", dataset_cfg.get("metadata")))
    phase.setdefault("image_root", dataset_cfg.get("image_root"))
    phase.setdefault("allow_memory_retrieval", protocol_cfg.get("allow_memory_retrieval", True))
    return phase


def _eval_phase_configs(datasets: dict, protocol_cfg: dict) -> list[dict]:
    phases = protocol_cfg.get("eval")
    if phases:
        resolved = []
        for phase in phases:
            phase = dict(phase)
            dataset_cfg = datasets.get(phase.get("dataset", ""), {})
            phase.setdefault("raw_root", dataset_cfg.get("raw_root"))
            phase.setdefault("split", dataset_cfg.get("split"))
            phase.setdefault("metadata", dataset_cfg.get("metadata"))
            phase.setdefault("image_root", dataset_cfg.get("image_root"))
            phase.setdefault("allow_memory_retrieval", protocol_cfg.get("allow_memory_retrieval", True))
            resolved.append(phase)
        return resolved
    resolved = []
    for name, dataset_cfg in datasets.items():
        if not dataset_cfg.get("enabled", True):
            continue
        resolved.append(
            {
                "dataset": name,
                "split": dataset_cfg.get("split"),
                "raw_root": dataset_cfg.get("raw_root"),
                "metadata": dataset_cfg.get("metadata"),
                "image_root": dataset_cfg.get("image_root"),
                "allow_memory_retrieval": protocol_cfg.get("allow_memory_retrieval", True),
            }
        )
    return resolved


def _evaluate_phase(
    phase: dict,
    output_dir: Path,
    llm_client,
    memory_service: MemoryService | None,
    update_memory: bool,
    paired_baseline_case_metrics: list[dict] | None = None,
    include_case_metrics: bool = False,
) -> dict:
    allow_memory_retrieval = phase.get("allow_memory_retrieval", True)
    dataset = phase.get("dataset")
    if dataset == "mv2026":
        return evaluate_mv2026(
            raw_root=phase.get("raw_root") or "data/raw/mv2026",
            output_dir=output_dir,
            protocol="train" if update_memory else "frozen_eval",
            split=phase.get("split"),
            llm_client=llm_client,
            limit=phase.get("limit"),
            memory_service=memory_service,
            update_memory=update_memory,
            allow_memory_retrieval=allow_memory_retrieval,
            paired_baseline_case_metrics=paired_baseline_case_metrics,
            include_case_metrics=include_case_metrics,
        )
    if dataset == "cosmos":
        return evaluate_cosmos(
            cosmos_metadata=phase.get("metadata") or "data/raw/cosmos/test.jsonl",
            image_root=phase.get("image_root") or "data/raw/cosmos/images",
            output_dir=output_dir,
            mode=phase.get("mode", "closed_world"),
            split=phase.get("split"),
            llm_client=llm_client,
            memory_service=memory_service,
            update_memory=update_memory,
            allow_memory_retrieval=allow_memory_retrieval,
            limit=phase.get("limit"),
            paired_baseline_case_metrics=paired_baseline_case_metrics,
            include_case_metrics=include_case_metrics,
        )
    raise ValueError(f"Unknown dataset in protocol phase: {dataset!r}")


def _evaluate_datasets(
    datasets: dict,
    out: Path,
    llm_client,
    memory_service: MemoryService | None,
    update_memory: bool,
    results: dict,
    allow_memory_retrieval: bool = True,
) -> None:
    mv_cfg = datasets.get("mv2026", {})
    cosmos_cfg = datasets.get("cosmos", {})
    if mv_cfg.get("enabled", True):
        results["runs"]["mv2026_{}".format(mv_cfg.get("split", "validation"))] = evaluate_mv2026(
            raw_root=mv_cfg.get("raw_root", "data/raw/mv2026"),
            output_dir=out / "mv2026",
            protocol="static",
            split=mv_cfg.get("split", "validation"),
            llm_client=llm_client,
            memory_service=memory_service,
            update_memory=update_memory,
            allow_memory_retrieval=allow_memory_retrieval,
        )
    if cosmos_cfg.get("enabled", True):
        results["runs"]["cosmos_{}".format(cosmos_cfg.get("split", "test"))] = evaluate_cosmos(
            cosmos_metadata=cosmos_cfg.get("metadata", "data/raw/cosmos/test.jsonl"),
            image_root=cosmos_cfg.get("image_root", "data/raw/cosmos"),
            output_dir=out / "cosmos",
            mode=cosmos_cfg.get("mode", "closed_world"),
            split=cosmos_cfg.get("split", "test"),
            llm_client=llm_client,
            memory_service=memory_service,
            update_memory=update_memory,
            allow_memory_retrieval=allow_memory_retrieval,
        )


def _ensure_clean_memory_dir(memory_dir: Path, resume: bool) -> None:
    if resume or not memory_dir.exists():
        return
    state_names = {
        "short_term_memory.jsonl", "episodic_memory.jsonl", "failure_memory.jsonl",
        "semantic_rules.jsonl", "consolidation_events.jsonl", "manifest.json",
    }
    contaminated = [path for path in memory_dir.rglob("*") if path.is_file() and (path.name in state_names or path.suffix == ".jsonl") and path.stat().st_size > 0]
    if contaminated:
        raise FileExistsError(
            f"Protocol memory directory already contains state: {memory_dir}. "
            "Use a clean output directory or set resume=true explicitly."
        )


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else project_root() / target
