from __future__ import annotations

from pathlib import Path

from src.evaluation.calibration_metrics import brier_score, expected_calibration_error
from src.evaluation.common import (
    auroc,
    average_precision,
    binary_probability,
    classification_metrics,
    confusion_matrix,
    prediction_from_report,
    write_records,
)
from src.evaluation.label_normalizer import normalize_cosmos_label, to_binary_cheapfake
from src.evaluation.memory_metrics import memory_metrics
from src.ingestion.cosmos_adapter import build_cosmos_case, load_cosmos_rows
from src.main import run_case_bundle
from src.schemas.evaluation_schema import CaseMetricRecord, GoldRecord
from src.utils.io import project_root


def evaluate_cosmos(
    cosmos_metadata: str | Path = "data/raw/cosmos/test.jsonl",
    image_root: str | Path | None = "data/raw/cosmos/images",
    output_dir: str | Path = "data/outputs/evaluation/cosmos_static",
    mode: str = "closed_world",
    split: str | None = "test",
    llm_client=None,
    memory_service=None,
    update_memory: bool = False,
    allow_memory_retrieval: bool = True,
    limit: int | None = None,
    paired_baseline_case_metrics: list[dict] | None = None,
    include_case_metrics: bool = False,
) -> dict:
    metadata_path = _resolve(cosmos_metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"COSMOS metadata not found: {metadata_path}")
    rows = load_cosmos_rows(metadata_path)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No COSMOS rows found in {metadata_path}")
    image_base = _resolve(image_root) if image_root else metadata_path.parent

    predictions = []
    gold_records = []
    per_case = []
    gold_labels = []
    pred_labels = []
    y_true = []
    y_prob = []

    if update_memory and memory_service is not None and memory_service.frozen:
        raise ValueError("update_memory=True is incompatible with a frozen memory service.")

    for row in rows:
        bundle = build_cosmos_case(row, split=split, image_root=image_base)
        bundle = bundle.model_copy(
            update={
                "run_config": bundle.run_config.model_copy(
                    update={
                        "allow_web_search": False,
                        "allow_reverse_search": False,
                        "allow_memory_update": update_memory,
                        "allow_memory_retrieval": allow_memory_retrieval,
                    }
                )
            }
        )
        run_mode = "bootstrap_memory" if update_memory else "test"
        report = run_case_bundle(
            bundle,
            mode=run_mode,
            llm_client=llm_client,
            memory_service=memory_service,
        )
        prediction = prediction_from_report(report, "cosmos", "out_of_context_detection")
        predictions.append(prediction.model_dump(mode="json"))
        gold = GoldRecord(
            case_id=bundle.case_id,
            dataset_name="cosmos",
            task_type="out_of_context_detection",
            gold_final_label=bundle.gold.gold_final_label,
        )
        gold_records.append(gold.model_dump(mode="json"))
        gold_label = normalize_cosmos_label(bundle.gold.gold_final_label)
        pred_label = normalize_cosmos_label(report.final_status)
        gold_labels.append(gold_label)
        pred_labels.append(pred_label)
        y_true.append(1 if to_binary_cheapfake(gold_label) == "cheapfake" else 0)
        y_prob.append(binary_probability(report.final_status, report.final_confidence))
        per_case.append(
            CaseMetricRecord(
                case_id=bundle.case_id,
                dataset_name="cosmos",
                final_label_correct=to_binary_cheapfake(gold_label) == to_binary_cheapfake(pred_label),
                final_label_score=1.0 if to_binary_cheapfake(gold_label) == to_binary_cheapfake(pred_label) else 0.0,
                calibration_error=abs(y_true[-1] - y_prob[-1]),
                notes={"mode": mode, "closed_world": True},
            ).model_dump(mode="json")
        )

    class_metrics = classification_metrics(gold_labels, pred_labels)
    binary_gold = [to_binary_cheapfake(label) for label in gold_labels]
    binary_pred = [to_binary_cheapfake(label) for label in pred_labels]
    binary_metrics = classification_metrics(binary_gold, binary_pred)
    calibration = expected_calibration_error(y_true, y_prob)
    aggregate = {
        "dataset": "cosmos",
        "mode": mode,
        "closed_world": True,
        "accuracy": binary_metrics["accuracy"],
        "balanced_accuracy": _balanced_accuracy(y_true, [1 if label == "cheapfake" else 0 for label in binary_pred]),
        "macro_f1": binary_metrics["macro_f1"],
        "label_macro_f1": class_metrics["macro_f1"],
        "precision": binary_metrics["per_label"].get("cheapfake", {}).get("precision", 0.0),
        "recall": binary_metrics["per_label"].get("cheapfake", {}).get("recall", 0.0),
        "auroc": auroc(y_true, y_prob),
        "average_precision": average_precision(y_true, y_prob),
        "auprc": average_precision(y_true, y_prob),
        "brier_score": brier_score(y_true, y_prob),
        "ece": calibration["ece"] if calibration else None,
        "abstention_rate": sum(1 for label in binary_pred if label == "abstain") / len(binary_pred),
        "risk_coverage": risk_coverage(y_true, y_prob),
        "confusion_matrix": confusion_matrix(binary_gold, binary_pred),
    }
    mem = memory_metrics(
        predictions,
        per_case,
        store=memory_service.store if memory_service is not None else None,
        paired_baseline_case_metrics=paired_baseline_case_metrics,
    )
    out = _resolve(output_dir)
    write_records(out, predictions, gold_records, per_case, aggregate, calibration or {}, mem)
    if include_case_metrics:
        return {**aggregate, "_case_metrics": per_case}
    return aggregate


def risk_coverage(y_true: list[int], y_prob: list[float]) -> list[dict]:
    rows = []
    ranked = sorted(zip(y_true, y_prob), key=lambda item: abs(item[1] - 0.5), reverse=True)
    for coverage in [0.25, 0.5, 0.75, 1.0]:
        k = max(1, int(round(len(ranked) * coverage))) if ranked else 0
        subset = ranked[:k]
        if not subset:
            rows.append({"coverage": coverage, "risk": None})
            continue
        errors = sum(int(true != (prob >= 0.5)) for true, prob in subset)
        rows.append({"coverage": coverage, "risk": errors / len(subset)})
    return rows


def _balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float | None:
    recalls = []
    for label in [0, 1]:
        support = sum(1 for value in y_true if value == label)
        if not support:
            continue
        recalls.append(sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label) / support)
    return sum(recalls) / len(recalls) if recalls else None


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else project_root() / target
