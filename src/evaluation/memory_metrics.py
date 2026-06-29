from __future__ import annotations


def memory_metrics(predictions: list[dict], case_metrics: list[dict]) -> dict:
    retrieved = [len(row.get("memory_used_ids", [])) for row in predictions]
    correct_by_case = {row["case_id"]: row.get("final_label_correct") for row in case_metrics}
    used_cases = [row for row in predictions if row.get("memory_used_ids")]
    helpful = [row for row in used_cases if correct_by_case.get(row["case_id"]) is True]
    harmful = [row for row in used_cases if correct_by_case.get(row["case_id"]) is False]
    return {
        "memory_retrieval_rate": sum(retrieved) / len(retrieved) if retrieved else 0.0,
        "memory_usage_rate": len(used_cases) / len(predictions) if predictions else 0.0,
        "memory_helpfulness_rate": len(helpful) / len(used_cases) if used_cases else 0.0,
        "negative_transfer_rate": len(harmful) / len(used_cases) if used_cases else 0.0,
        "lesson_acceptance_rate": None,
        "failure_recurrence_rate": None,
        "semantic_promotion_rate": None,
        "memory_conflict_rate": None,
    }
