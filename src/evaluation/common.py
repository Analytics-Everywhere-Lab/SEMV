from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from src.evaluation.label_normalizer import to_binary_cheapfake
from src.schemas.evaluation_schema import PredictionRecord
from src.schemas.report_schema import VerificationReport
from src.utils.io import append_jsonl, write_json


PAIRED_MEMORY_RESULT_KEYS = (
    "negative_transfer_rate",
    "positive_transfer_rate",
    "paired_case_count",
    "baseline_correct_memory_wrong_count",
    "memory_correct_baseline_wrong_count",
    "both_correct_count",
    "both_wrong_count",
    "missing_from_memory_on",
    "missing_from_memory_off",
    "paired_case_ids",
)


def evaluation_result(
    aggregate_metrics: dict,
    memory_metrics: dict,
    case_metrics: list[dict] | None = None,
) -> dict:
    """Return backward-compatible aggregate fields plus structured metrics."""
    result = {
        **aggregate_metrics,
        "aggregate_metrics": aggregate_metrics,
        "memory_metrics": memory_metrics,
    }
    for key in PAIRED_MEMORY_RESULT_KEYS:
        result[key] = memory_metrics.get(key)
    if case_metrics is not None:
        result["_case_metrics"] = case_metrics
    return result



def prediction_from_report(report: VerificationReport, dataset_name: str, task_type: str) -> PredictionRecord:
    return PredictionRecord(
        case_id=report.case_id,
        dataset_name=dataset_name,
        task_type=task_type,
        final_label=report.final_status,
        final_score=report.final_confidence,
        final_confidence=report.final_confidence,
        subclaims=[
            {
                "claim_type": row.claim_type,
                "statement": row.statement,
                "decision": row.decision,
                "score": row.score,
                "confidence": row.confidence,
                "top_support_argument_ids": [arg.argument_id for arg in row.top_support_arguments],
                "top_attack_argument_ids": [arg.argument_id for arg in row.top_attack_arguments],
                "evidence_ids": sorted({eid for arg in row.top_support_arguments + row.top_attack_arguments for eid in arg.evidence_ids}),
                "uncertainty_reason": row.uncertainty_reason,
            }
            for row in report.subclaim_reports
        ],
        predicted_source_urls=sorted({item.url for item in report.evidence if item.url}),
        memory_used_ids=[item.memory_id for item in report.memory_used],
        memory_retrieved_ids=[item.memory_id for item in report.memory_retrieved],
        run_metadata=report.metadata,
    )


def classification_metrics(gold_labels: list[str], pred_labels: list[str]) -> dict:
    labels = sorted(set(gold_labels) | set(pred_labels))
    total = len(gold_labels)
    accuracy = sum(int(g == p) for g, p in zip(gold_labels, pred_labels)) / total if total else 0.0
    f1s = []
    per_label = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(1 for g in gold_labels if g == label)}
        f1s.append(f1)
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0, "per_label": per_label}


def confusion_matrix(gold_labels: list[str], pred_labels: list[str]) -> dict:
    counts = Counter(f"{g}::{p}" for g, p in zip(gold_labels, pred_labels))
    labels = sorted(set(gold_labels) | set(pred_labels))
    return {gold: {pred: counts.get(f"{gold}::{pred}", 0) for pred in labels} for gold in labels}


def binary_probability(label: str, confidence: float | None) -> float:
    conf = 0.5 if confidence is None else max(0.0, min(1.0, confidence))
    binary = to_binary_cheapfake(label)
    if binary == "cheapfake":
        return conf
    if binary == "not_cheapfake":
        return 1.0 - conf
    if binary == "manipulated":
        return conf
    return 0.5


def auroc(y_true: list[int], y_score: list[float]) -> float | None:
    positives = [(s, y) for y, s in zip(y_true, y_score) if y == 1]
    negatives = [(s, y) for y, s in zip(y_true, y_score) if y == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for ps, _ in positives:
        for ns, _ in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def average_precision(y_true: list[int], y_score: list[float]) -> float | None:
    if not any(y_true):
        return None
    pairs = sorted(zip(y_score, y_true), reverse=True)
    tp = 0
    precision_sum = 0.0
    for idx, (_, true) in enumerate(pairs, start=1):
        if true:
            tp += 1
            precision_sum += tp / idx
    return precision_sum / sum(y_true)


def write_records(output_dir: Path, predictions, gold, per_case, aggregate, calibration_bins, memory):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in [
        ("predictions.jsonl", predictions),
        ("gold.jsonl", gold),
        ("per_case_metrics.jsonl", per_case),
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()
        for record in records:
            append_jsonl(path, record)
    write_json(output_dir / "aggregate_metrics.json", aggregate)
    write_json(output_dir / "confusion_matrix.json", aggregate.get("confusion_matrix", {}))
    write_json(output_dir / "calibration_bins.json", calibration_bins)
    write_json(output_dir / "memory_metrics.json", memory)
    failed_path = output_dir / "failed_cases.jsonl"
    if failed_path.exists():
        failed_path.unlink()
    for record in per_case:
        if record.get("final_label_correct") is False:
            append_jsonl(failed_path, record)
    (output_dir / "evaluation_report.md").write_text(_render_eval_report(aggregate), encoding="utf-8")


def _render_eval_report(aggregate: dict) -> str:
    lines = ["# Evaluation Report", ""]
    for key, value in aggregate.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"
