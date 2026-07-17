from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from src.config.runtime import PipelineRuntimeConfig, RuntimeFeatures, load_runtime_config
from src.evaluation.cosmos_evaluator import evaluate_cosmos
from src.evaluation.mv2026_evaluator import evaluate_mv2026
from src.memory.memory_service import MemoryService
from src.memory.seed_memory import seed_semantic_rules
from src.utils.io import project_root, read_yaml, write_json


logger = logging.getLogger("run_case")

ABLATION_VARIANTS = {
    "A0": {"name": "No memory, no A-QBAF", "use_memory": False, "memory_types": [], "use_qbaf": False, "argument_verifier": False, "clash_resolution": False, "adaptive_revision": False},
    "A1": {"name": "A-QBAF only", "use_memory": False, "memory_types": [], "use_qbaf": True, "argument_verifier": True, "clash_resolution": True, "adaptive_revision": True},
    "A2": {"name": "A-QBAF + episodic memory", "use_memory": True, "memory_types": ["episodic"], "use_qbaf": True, "argument_verifier": True, "clash_resolution": True, "adaptive_revision": True},
    "A3": {"name": "A-QBAF + semantic memory", "use_memory": True, "memory_types": ["semantic_rule"], "use_qbaf": True, "argument_verifier": True, "clash_resolution": True, "adaptive_revision": True},
    "A4": {"name": "A-QBAF + failure memory", "use_memory": True, "memory_types": ["failure"], "use_qbaf": True, "argument_verifier": True, "clash_resolution": True, "adaptive_revision": True},
    "A5": {"name": "All memory, no Verify Agent", "use_memory": True, "memory_types": ["episodic", "failure", "semantic_rule"], "use_qbaf": True, "argument_verifier": False, "clash_resolution": True, "adaptive_revision": True},
    "A6": {"name": "All memory + Verify Agent", "use_memory": True, "memory_types": ["episodic", "failure", "semantic_rule"], "use_qbaf": True, "argument_verifier": True, "clash_resolution": True, "adaptive_revision": True},
    "A7": {"name": "Full system without clash resolution", "use_memory": True, "memory_types": ["episodic", "failure", "semantic_rule"], "use_qbaf": True, "argument_verifier": True, "clash_resolution": False, "adaptive_revision": True},
    "A8": {"name": "Full system with clash resolution", "use_memory": True, "memory_types": ["episodic", "failure", "semantic_rule"], "use_qbaf": True, "argument_verifier": True, "clash_resolution": True, "adaptive_revision": True},
    "A9": {"name": "Full system without argument verifier", "use_memory": True, "memory_types": ["episodic", "failure", "semantic_rule"], "use_qbaf": True, "argument_verifier": False, "clash_resolution": True, "adaptive_revision": True},
    "A10": {"name": "Full system with argument verifier", "use_memory": True, "memory_types": ["episodic", "failure", "semantic_rule"], "use_qbaf": True, "argument_verifier": True, "clash_resolution": True, "adaptive_revision": True},
}


def run_protocol(
    config_path: str | Path = "configs/evaluation.yaml",
    protocol: str | None = None,
    output_dir: str | Path | None = None,
    llm_client=None,
    resume: bool | None = None,
    ablation_variant: str | None = None,
) -> dict:
    config = read_yaml(config_path)
    evaluation = config.get("evaluation", {})
    datasets = evaluation.get("datasets", {})
    protocol_cfg = dict(evaluation.get("protocol", {}))
    protocol_cfg["_evaluation_run_id"] = evaluation.get("run_id")
    if ablation_variant is not None:
        protocol_cfg["ablation_variant"] = ablation_variant
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
    runtime = load_runtime_config(evaluation.get("pipeline_config", "configs/default.yaml"))

    if selected == "static":
        results = _run_static(datasets, protocol_cfg, out, llm_client, memory_config_path, runtime)
    elif selected == "prequential":
        results = _run_prequential(datasets, protocol_cfg, out, llm_client, memory_config_path, runtime)
    elif selected == "train_memory_freeze_test":
        results = _run_train_memory_freeze_test(
            datasets, protocol_cfg, out, llm_client, memory_config_path, runtime
        )
    elif selected == "mv2026_to_cosmos_transfer":
        results = _run_transfer(datasets, protocol_cfg, out, llm_client, memory_config_path, runtime)
    elif selected == "ablations":
        results = _run_ablations(
            datasets, protocol_cfg, out, llm_client, memory_config_path, runtime
        )
    else:
        raise ValueError(f"Unsupported protocol: {selected}")

    write_json(out / "protocol_results.json", results)
    return results


# ------------------------------------------------------------------ protocols


def _run_static(
    datasets, protocol_cfg, out: Path, llm_client, memory_config_path,
    runtime: PipelineRuntimeConfig,
) -> dict:
    """Evaluate against an explicit, verified, read-only memory snapshot."""
    allow_retrieval = bool(protocol_cfg.get("allow_memory_retrieval", True))
    source = _open_frozen_source(
        protocol_cfg, memory_config_path, llm_client,
        usage_log_path=out / "memory_usage_events.jsonl",
        required=allow_retrieval,
    )
    service = source["service"] if source else None
    run_runtime = runtime.model_copy(update={
        "features": runtime.features.model_copy(update={"use_memory": allow_retrieval})
    })
    results = {
        "protocol": "static", "runs": {},
        "resolved_configuration": run_runtime.model_dump(mode="json"),
    }
    if source:
        results["frozen_memory"] = {key: value for key, value in source.items() if key != "service"}
    _evaluate_datasets(
        datasets, out, llm_client, memory_service=service,
        allow_memory_retrieval=allow_retrieval, update_memory=False, results=results,
        runtime_config=run_runtime, condition="memory_on" if allow_retrieval else "memory_off",
    )
    if source:
        post_hash = service.state_hash(include_short_term=True)
        post_manifest_hash = _sha256_file(source["manifest_path"])
        results["frozen_memory"]["post_run_hash"] = post_hash
        results["frozen_memory"]["post_manifest_hash"] = post_manifest_hash
        results["frozen_memory"]["unchanged"] = (
            post_hash == source["pre_run_hash"]
            and post_manifest_hash == source["manifest_hash"]
        )
        if not results["frozen_memory"]["unchanged"]:
            raise RuntimeError("Frozen memory snapshot changed during static evaluation")
    return results


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _open_frozen_source(
    protocol_cfg: dict, memory_config_path, llm_client, usage_log_path: Path,
    required: bool,
) -> dict | None:
    configured = (protocol_cfg.get("frozen_memory_snapshot")
                  or protocol_cfg.get("memory_source_dir"))
    if not configured:
        if required:
            raise ValueError(
                "Memory retrieval is enabled but evaluation.protocol.frozen_memory_snapshot "
                "(or memory_source_dir) is not configured."
            )
        return None
    source = _resolve(configured)
    if not source.is_dir():
        raise FileNotFoundError(f"Frozen memory snapshot directory not found: {source}")
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Frozen memory snapshot has no manifest.json: {source}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid frozen memory manifest {manifest_path}: {exc}") from exc
    expected = manifest.get("state_hash") or manifest.get("full_state_hash")
    if not isinstance(expected, str) or not expected:
        raise ValueError(f"Frozen memory manifest has no valid state_hash: {manifest_path}")
    service = MemoryService.from_config_path(
        memory_config_path, memory_dir=source, llm_client=llm_client, frozen=True,
        usage_log_path=usage_log_path,
    )
    actual = service.state_hash(include_short_term=True)
    if actual != expected:
        raise ValueError(
            f"Frozen memory snapshot state hash mismatch: manifest={expected}, actual={actual}"
        )
    active_count = len(service.store.load_long_term(statuses=["active"]))
    if required and active_count == 0 and not protocol_cfg.get("allow_empty_memory", False):
        raise ValueError(
            "Frozen memory snapshot contains no active long-term memory; set "
            "allow_empty_memory=true only for an intentional empty-memory evaluation."
        )
    return {
        "service": service, "source_path": str(source),
        "manifest_path": manifest_path, "manifest_hash": _sha256_file(manifest_path),
        "manifest_state_hash": expected, "pre_run_hash": actual,
        "active_memory_count": active_count,
    }


def _run_ablations(
    datasets, protocol_cfg, out: Path, llm_client, memory_config_path,
    base_runtime: PipelineRuntimeConfig,
) -> dict:
    selection = protocol_cfg.get("ablation_variant", "all")
    variant_ids = list(ABLATION_VARIANTS) if selection in {None, "all"} else [str(selection)]
    unknown = [variant for variant in variant_ids if variant not in ABLATION_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown ablation variant(s): {unknown}; expected A0-A10 or all")
    results = {"protocol": "ablations", "variants": {}, "variant_order": variant_ids}
    for variant_id in variant_ids:
        definition = dict(ABLATION_VARIANTS[variant_id])
        name = definition.pop("name")
        features = RuntimeFeatures.model_validate(definition)
        runtime = base_runtime.model_copy(update={"features": features})
        variant_out = out / "ablations" / variant_id
        source = _open_frozen_source(
            protocol_cfg, memory_config_path, llm_client,
            usage_log_path=variant_out / "memory_usage_events.jsonl",
            required=features.use_memory,
        )
        service = source["service"] if source else None
        variant_result = {
            "variant": variant_id, "name": name, "runs": {},
            "feature_configuration": features.model_dump(mode="json"),
            "resolved_configuration": runtime.model_dump(mode="json"),
            "output_dir": str(variant_out),
        }
        if source:
            variant_result["frozen_memory"] = {
                key: value for key, value in source.items() if key != "service"
            }
        _evaluate_datasets(
            datasets, variant_out, llm_client, memory_service=service,
            update_memory=False, results=variant_result,
            allow_memory_retrieval=features.use_memory, runtime_config=runtime,
            condition=variant_id,
        )
        if source:
            post_hash = service.state_hash(include_short_term=True)
            post_manifest_hash = _sha256_file(source["manifest_path"])
            unchanged = (post_hash == source["pre_run_hash"]
                         and post_manifest_hash == source["manifest_hash"] )
            variant_result["frozen_memory"].update({
                "post_run_hash": post_hash, "post_manifest_hash": post_manifest_hash,
                "unchanged": unchanged,
            })
            if not unchanged:
                raise RuntimeError(f"Frozen snapshot changed during ablation {variant_id}")
        results["variants"][variant_id] = variant_result
        write_json(variant_out / "variant_results.json", variant_result)
    write_json(out / "ablation_comparison.json", results)
    return results


def _run_prequential(datasets, protocol_cfg, out: Path, llm_client, memory_config_path, runtime) -> dict:
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
        runtime_config=runtime, condition="training",
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
    datasets, protocol_cfg, out: Path, llm_client, memory_config_path, runtime
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

    results: dict = {"protocol": "train_memory_freeze_test", "runs": {}, "memory_dir": str(memory_dir),
                     "resolved_configuration": runtime.model_dump(mode="json")}

    train_cfg = _train_phase_config(datasets, protocol_cfg)
    results["runs"]["train_{}_{}".format(train_cfg["dataset"], train_cfg.get("split") or "default")] = _evaluate_phase(
        train_cfg,
        out / "train",
        llm_client,
        memory_service=train_service,
        update_memory=protocol_cfg.get("allow_memory_update_train", True),
        runtime_config=runtime, condition="training",
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

    paired_enabled = bool(
        protocol_cfg.get("run_paired_memory_off_baseline", False)
    )
    evaluation_run_id = (
        protocol_cfg.get("_evaluation_run_id") or "semv_evaluation"
    )
    decoding = _llm_reproducibility(llm_client)
    if paired_enabled and decoding["deterministic_decoding"] is False:
        logger.warning(
            "Paired memory evaluation uses stochastic decoding; transfer "
            "estimates may include decoding variance and are not strictly causal."
        )
    results["evaluation_run_id"] = evaluation_run_id
    results["paired_evaluation"] = {
        "enabled": paired_enabled,
        "snapshot_hash_before": frozen_hash,
        "deterministic_decoding": decoding["deterministic_decoding"],
        "model_configuration": decoding["model_configuration"],
        "phases": {},
    }

    paired_baselines = {}
    memory_off_runtime = runtime.model_copy(update={
        "features": runtime.features.model_copy(update={"use_memory": False})
    })
    for phase in _eval_phase_configs(datasets, protocol_cfg):
        phase_key = f"{phase['dataset']}_{phase.get('split') or 'default'}"
        phase_out = out / "eval" / phase_key
        baseline_case_metrics = None
        baseline_run_id = f"{evaluation_run_id}:{phase_key}:memory_off"
        memory_on_run_id = f"{evaluation_run_id}:{phase_key}:memory_on"
        if paired_enabled:
            baseline_phase = {**phase, "allow_memory_retrieval": False}
            baseline_result = _evaluate_phase(
                baseline_phase,
                out / "eval_memory_off" / phase_key,
                llm_client,
                memory_service=frozen_service,
                update_memory=False,
                include_case_metrics=True,
                runtime_config=memory_off_runtime, condition="memory_off",
            )
            baseline_case_metrics = baseline_result.pop("_case_metrics", None)
            baseline_result["evaluation_run_id"] = baseline_run_id
            baseline_result["deterministic_decoding"] = decoding[
                "deterministic_decoding"
            ]
            paired_baselines[phase_key] = baseline_result

        memory_on_result = _evaluate_phase(
            phase,
            phase_out,
            llm_client,
            memory_service=frozen_service,
            update_memory=False,
            paired_baseline_case_metrics=baseline_case_metrics,
            runtime_config=runtime, condition="memory_on",
        )
        memory_on_result["evaluation_run_id"] = memory_on_run_id
        memory_on_result["deterministic_decoding"] = decoding[
            "deterministic_decoding"
        ]
        results["runs"][f"eval_{phase_key}"] = memory_on_result
        if paired_enabled:
            results["paired_evaluation"]["phases"][phase_key] = {
                "memory_on_run_id": memory_on_run_id,
                "memory_off_run_id": baseline_run_id,
                "case_ids": memory_on_result.get("paired_case_ids", []),
                "paired_case_count": memory_on_result.get(
                    "paired_case_count", 0
                ),
            }
    if paired_baselines:
        results["paired_memory_off_baselines"] = paired_baselines




    post_hash = frozen_service.state_hash()
    results["state_hash_after_eval"] = post_hash
    results["paired_evaluation"]["snapshot_hash_after"] = post_hash
    results["paired_evaluation"]["snapshot_unchanged"] = post_hash == frozen_hash
    if post_hash != frozen_hash:
        raise RuntimeError(
            "Frozen memory snapshot changed during validation/test: "
            f"{frozen_hash} -> {post_hash}"
        )
    results["memory_state_unchanged"] = True
    return results


def _run_transfer(datasets, protocol_cfg, out: Path, llm_client, memory_config_path, runtime) -> dict:
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
                                           memory_config_path, runtime)
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
    runtime_config: PipelineRuntimeConfig | None = None,
    condition: str | None = None,
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
            runtime_config=runtime_config,
            artifact_root=output_dir / "cases",
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
            runtime_config=runtime_config,
            artifact_root=output_dir / "cases",
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
    runtime_config: PipelineRuntimeConfig | None = None,
    condition: str = "memory_on",
) -> None:
    mv_cfg = datasets.get("mv2026", {})
    cosmos_cfg = datasets.get("cosmos", {})
    if mv_cfg and mv_cfg.get("enabled", True):
        results["runs"]["mv2026_{}".format(mv_cfg.get("split", "validation"))] = evaluate_mv2026(
            raw_root=mv_cfg.get("raw_root", "data/raw/mv2026"),
            output_dir=out / "mv2026",
            protocol="static",
            split=mv_cfg.get("split", "validation"),
            llm_client=llm_client,
            memory_service=memory_service,
            update_memory=update_memory,
            allow_memory_retrieval=allow_memory_retrieval,
            runtime_config=runtime_config,
            artifact_root=out / "mv2026" / condition / "cases",
        )
    if cosmos_cfg and cosmos_cfg.get("enabled", True):
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
            runtime_config=runtime_config,
            artifact_root=out / "cosmos" / condition / "cases",
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


def _llm_reproducibility(llm_client) -> dict:
    client = getattr(llm_client, "wrapped", llm_client)
    model_configuration = {}
    for name in ("model", "temperature", "top_p", "top_k", "max_tokens"):
        value = getattr(client, name, None)
        if value is not None:
            model_configuration[name] = value
    temperature = model_configuration.get("temperature")
    deterministic = None if temperature is None else float(temperature) == 0.0
    return {
        "deterministic_decoding": deterministic,
        "model_configuration": model_configuration,
    }



def _resolve(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else project_root() / target
