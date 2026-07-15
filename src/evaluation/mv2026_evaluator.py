from __future__ import annotations

from pathlib import Path

from src.evaluation.calibration_metrics import expected_calibration_error
from src.evaluation.common import (
    binary_probability,
    classification_metrics,
    confusion_matrix,
    prediction_from_report,
    write_records,
)
from src.evaluation.evidence_metrics import url_f1
from src.evaluation.label_normalizer import normalize_mv2026_label, to_binary_cheapfake
from src.evaluation.memory_metrics import memory_metrics
from src.evaluation.mv2026_gold_parser import parse_mv2026_gold_report
from src.evaluation.report_metrics import report_structure_score
from src.evaluation.temporal_metrics import publication_event_confusion, temporal_bound_score
from src.ingestion.mv2026_adapter import MV2026Adapter
from src.main import run_case_bundle
from src.schemas.evaluation_schema import CaseMetricRecord, GoldRecord
from src.utils.io import project_root, write_json


def evaluate_mv2026(
    raw_root: str | Path = "data/raw/mv2026",
    output_dir: str | Path = "data/outputs/evaluation/mv2026_static",
    protocol: str = "static",
    split: str | None = "validation",
    llm_client=None,
    case_id: str | None = None,
    limit: int | None = None,
    memory_service=None,
    update_memory: bool = False,
) -> dict:
    root = _resolve(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"MV2026 raw root not found: {root}")
    case_dirs = _discover_case_dirs(root)
    if case_id:
        case_dirs = [path for path in case_dirs if path.name == case_id]
    if limit is not None:
        case_dirs = case_dirs[:limit]
    if not case_dirs:
        suffix = f" matching case_id={case_id!r}" if case_id else ""
        raise ValueError(
            f"No MV2026 cases found under {root}{suffix}; expected case folders containing input/*.json"
        )

    adapter = MV2026Adapter()
    predictions = []
    gold_records = []
    per_case = []
    gold_labels = []
    pred_labels = []
    y_true = []
    y_prob = []

    if update_memory and memory_service is not None and memory_service.frozen:
        raise ValueError("update_memory=True is incompatible with a frozen memory service.")

    for case_dir in case_dirs:
        bundle = adapter.load(case_dir, split=split)
        bundle = bundle.model_copy(
            update={
                "run_config": bundle.run_config.model_copy(
                    update={"allow_memory_update": update_memory}
                )
            }
        )
        # Prequential/bootstrap: predict first with memory from previous cases,
        # then reveal gold and stage reflection. Evaluation-only runs never update.
        mode = "bootstrap_memory" if update_memory else "inference_only"
        report = run_case_bundle(
            bundle,
            mode=mode,
            llm_client=llm_client,
            case_path=case_dir,
            memory_service=memory_service,
        )
        prediction = prediction_from_report(report, "mv2026", "multimedia_verification")
        predictions.append(prediction.model_dump(mode="json"))

        gold = _gold_from_bundle(bundle)
        gold_records.append(gold.model_dump(mode="json"))
        gold_label = normalize_mv2026_label(gold.gold_final_label or gold.gold_status_text)
        pred_label = normalize_mv2026_label(report.final_status)
        gold_labels.append(gold_label)
        pred_labels.append(pred_label)
        y_true.append(1 if to_binary_cheapfake(gold_label) == "cheapfake" else 0)
        y_prob.append(binary_probability(report.final_status, report.final_confidence))

        evidence_scores = url_f1(set(prediction.predicted_source_urls), set(gold.gold_source_urls))
        markdown_path = project_root() / "data" / "outputs" / "cases" / bundle.case_id / "report.md"
        markdown = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
        metric = CaseMetricRecord(
            case_id=bundle.case_id,
            dataset_name="mv2026",
            final_label_correct=gold_label == pred_label,
            final_label_score=1.0 if gold_label == pred_label else 0.0,
            subclaim_accuracy=_subclaim_accuracy(prediction.model_dump(mode="json"), gold.model_dump(mode="json")),
            evidence_url_precision=evidence_scores["precision"],
            evidence_url_recall=evidence_scores["recall"],
            evidence_url_f1=evidence_scores["f1"],
            temporal_bound_score=temporal_bound_score(prediction.predicted_time_bounds, gold.gold_time_bounds),
            publication_event_confusion=publication_event_confusion(prediction.predicted_time_bounds, gold.gold_time_bounds),
            report_structure_score=report_structure_score(markdown),
            notes={"protocol": protocol, "gold_report_available": bundle.gold.gold_report_available},
        )
        per_case.append(metric.model_dump(mode="json"))

    class_metrics = classification_metrics(gold_labels, pred_labels)
    calibration = expected_calibration_error(y_true, y_prob)
    aggregate = {
        "dataset": "mv2026",
        "protocol": protocol,
        **class_metrics,
        "binary_accuracy": _binary_accuracy(gold_labels, pred_labels),
        "abstention_rate": sum(1 for label in pred_labels if label in {"uncertain", "insufficient_evidence"}) / len(pred_labels),
        "correct_abstention_rate": _correct_abstention_rate(gold_labels, pred_labels),
        "confusion_matrix": confusion_matrix(gold_labels, pred_labels),
        "ece": calibration["ece"] if calibration else None,
    }
    mem = memory_metrics(
        predictions,
        per_case,
        store=memory_service.store if memory_service is not None else None,
    )
    out = _resolve(output_dir)
    write_records(out, predictions, gold_records, per_case, aggregate, calibration or {}, mem)
    return aggregate


def _discover_case_dirs(root: Path) -> list[Path]:
    if (root / "input").is_dir() and any((root / "input").glob("*.json")):
        return [root]

    case_dirs = {
        input_dir.parent
        for input_dir in root.rglob("input")
        if input_dir.is_dir() and any(input_dir.glob("*.json"))
    }
    return sorted(case_dirs)


def _gold_from_bundle(bundle) -> GoldRecord:
    if bundle.gold.gold_report_path:
        return parse_mv2026_gold_report(bundle.gold.gold_report_path, case_id=bundle.case_id)
    return GoldRecord(
        case_id=bundle.case_id,
        dataset_name="mv2026",
        task_type="multimedia_verification",
        gold_final_label=bundle.gold.gold_final_label,
        gold_subclaim_labels=bundle.gold.gold_subclaim_labels,
        gold_report_path=bundle.gold.gold_report_path,
    )


def _subclaim_accuracy(prediction: dict, gold: dict) -> dict[str, float]:
    gold_labels = gold.get("gold_subclaim_labels", {})
    if not gold_labels:
        return {}
    pred_by_type = {row["claim_type"]: row["decision"] for row in prediction.get("subclaims", [])}
    return {
        claim_type: section_label_score(pred_by_type.get(claim_type, "uncertain"), label)
        for claim_type, label in gold_labels.items()
    }


def section_label_score(pred_label: str, gold_label: str) -> float:
    if pred_label == gold_label:
        return 1.0
    compatible = {
        ("supported", "partially_supported"),
        ("weakly_supported", "partially_supported"),
        ("partially_supported", "supported"),
        ("uncertain", "insufficient_evidence"),
        ("insufficient_evidence", "uncertain"),
    }
    return 0.5 if (pred_label, gold_label) in compatible else 0.0


def _binary_accuracy(gold_labels: list[str], pred_labels: list[str]) -> float:
    if not gold_labels:
        return 0.0
    return sum(to_binary_cheapfake(g) == to_binary_cheapfake(p) for g, p in zip(gold_labels, pred_labels)) / len(gold_labels)


def _correct_abstention_rate(gold_labels: list[str], pred_labels: list[str]) -> float:
    abstained = [(g, p) for g, p in zip(gold_labels, pred_labels) if p in {"uncertain", "insufficient_evidence"}]
    if not abstained:
        return 0.0
    return sum(1 for g, _ in abstained if g in {"uncertain", "insufficient_evidence"}) / len(abstained)


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else project_root() / target
