from __future__ import annotations


def memory_metrics(
    predictions: list[dict],
    case_metrics: list[dict],
    store=None,
    paired_baseline_case_metrics: list[dict] | None = None,
) -> dict:
    """Real memory metrics.

    Per-run metrics come from predictions (memory_used_ids = memory actually
    cited by the planner or an argument). Lifecycle metrics come from the
    memory store when one is supplied. `memory_associated_error_rate` is an
    association only; actual negative transfer is computed exclusively from a
    paired memory-on vs memory-off run (`paired_baseline_case_metrics`)."""
    retrieved = [
        len(row.get("memory_retrieved_ids") or row.get("memory_used_ids", []))
        for row in predictions
    ]
    correct_by_case = {row["case_id"]: row.get("final_label_correct") for row in case_metrics}
    used_cases = [row for row in predictions if row.get("memory_used_ids")]
    helpful = [row for row in used_cases if correct_by_case.get(row["case_id"]) is True]
    harmful = [row for row in used_cases if correct_by_case.get(row["case_id"]) is False]

    comparison = paired_memory_comparison(case_metrics, paired_baseline_case_metrics)
    metrics = {
        "memory_retrieval_rate": sum(1 for count in retrieved if count > 0) / len(retrieved) if retrieved else 0.0,
        "average_memories_retrieved_per_case": sum(retrieved) / len(retrieved) if retrieved else 0.0,
        "memory_citation_rate": len(used_cases) / len(predictions) if predictions else 0.0,
        "memory_usage_rate": len(used_cases) / len(predictions) if predictions else 0.0,
        "memory_associated_correctness_rate": (
            len(helpful) / len(used_cases) if used_cases else None
        ),
        "memory_associated_error_rate": len(harmful) / len(used_cases) if used_cases else None,
    }
    metrics.update(comparison)
    metrics.update(_store_metrics(store))
    return metrics


def paired_memory_comparison(
    memory_on_case_metrics: list[dict],
    memory_off_case_metrics: list[dict] | None,
) -> dict:
    """Compare matched memory-on/off outcomes by case_id.

    Duplicate case IDs are rejected because silently overwriting them would make
    the paired estimate ambiguous. Rates use matched case IDs only.
    """
    empty = {
        "negative_transfer_rate": None,
        "positive_transfer_rate": None,
        "paired_case_count": 0,
        "baseline_correct_memory_wrong_count": 0,
        "memory_correct_baseline_wrong_count": 0,
        "both_correct_count": 0,
        "both_wrong_count": 0,
        "missing_from_memory_on": [],
        "missing_from_memory_off": [],
        "paired_case_ids": [],
    }
    if memory_off_case_metrics is None:
        return empty

    memory_on = _unique_case_metrics(memory_on_case_metrics, "memory-on")
    memory_off = _unique_case_metrics(memory_off_case_metrics, "memory-off")
    paired_case_ids = sorted(set(memory_on) & set(memory_off))
    missing_from_memory_on = sorted(set(memory_off) - set(memory_on))
    missing_from_memory_off = sorted(set(memory_on) - set(memory_off))

    counts = {
        "baseline_correct_memory_wrong_count": 0,
        "memory_correct_baseline_wrong_count": 0,
        "both_correct_count": 0,
        "both_wrong_count": 0,
    }
    for case_id in paired_case_ids:
        with_memory = memory_on[case_id].get("final_label_correct") is True
        without_memory = memory_off[case_id].get("final_label_correct") is True
        if without_memory and not with_memory:
            counts["baseline_correct_memory_wrong_count"] += 1
        elif with_memory and not without_memory:
            counts["memory_correct_baseline_wrong_count"] += 1
        elif with_memory and without_memory:
            counts["both_correct_count"] += 1
        else:
            counts["both_wrong_count"] += 1

    denominator = len(paired_case_ids)
    return {
        "negative_transfer_rate": (
            counts["baseline_correct_memory_wrong_count"] / denominator
            if denominator
            else None
        ),
        "positive_transfer_rate": (
            counts["memory_correct_baseline_wrong_count"] / denominator
            if denominator
            else None
        ),
        "paired_case_count": denominator,
        **counts,
        "missing_from_memory_on": missing_from_memory_on,
        "missing_from_memory_off": missing_from_memory_off,
        "paired_case_ids": paired_case_ids,
    }


def _unique_case_metrics(rows: list[dict], run_name: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    duplicates: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in indexed:
            duplicates.add(case_id)
        indexed[case_id] = row
    if duplicates:
        raise ValueError(
            f"Duplicate case_id values in {run_name} metrics: {sorted(duplicates)}"
        )
    return indexed



def _store_metrics(store) -> dict:
    if store is None:
        return {
            "lesson_acceptance_rate": None,
            "stm_to_ltm_promotion_rate": None,
            "merge_rate": None,
            "memory_conflict_rate": None,
            "semantic_promotion_rate": None,
            "semantic_generalization_rate": None,
            "semantic_generalization_proposal_count": None,
            "semantic_generalization_recovery_count": None,
            "active_semantic_rule_count": None,
            "under_review_semantic_proposal_count": None,
            "average_independent_support": None,
            "active_memory_count": None,
            "under_review_memory_count": None,
            "deprecated_memory_count": None,
            "failure_recurrence_rate": None,
        }

    short_term = store.load_short_term()
    long_term = store.load_long_term()

    verification_events = [event for event in store.load_consolidation_events() if event.event_type == "candidate_verification"]
    latest_by_candidate = {}
    for event in verification_events:
        latest_by_candidate[event.details.get("candidate_id")] = event
    acceptance_rate = (
        sum(1 for event in latest_by_candidate.values() if event.details.get("verified") is True) / len(latest_by_candidate)
        if latest_by_candidate else (
            sum(1 for row in short_term if row.verification_status == "verified") / len(short_term)
            if short_term else None
        )
    )

    promoted = [row for row in short_term if row.status == "promoted"]
    merged = [row for row in short_term if row.status == "merged"]
    conflicted_events = [
        event for event in store.load_consolidation_events() if event.event_type == "conflict"
    ]
    processed = [row for row in short_term if row.status in {"promoted", "merged", "staged", "under_review"}]

    semantic_consolidated = [
        record
        for record in long_term
        if record.memory_type == "semantic_rule" and record.origin == "consolidated"
    ]
    active_semantic = [record for record in semantic_consolidated if record.status == "active"]
    active_generalized = [
        record for record in active_semantic
        if record.metadata.get("generalized_from_stm_ids")
        and not record.metadata.get("proposal_only", False)
    ]
    generalization_proposals = [
        record for record in semantic_consolidated
        if record.status == "under_review"
        and record.metadata.get("generalized_from_stm_ids")
        and record.metadata.get("proposal_only", False)
    ]
    semantic_candidates = [row for row in short_term if row.memory_type == "semantic_rule"]

    consolidated = [record for record in long_term if record.origin == "consolidated"]
    average_support = (
        sum(record.independent_support() for record in consolidated) / len(consolidated)
        if consolidated
        else None
    )

    return {
        "lesson_acceptance_rate": acceptance_rate,
        "stm_to_ltm_promotion_rate": _bounded_rate(len(promoted), len(processed)),
        "merge_rate": _bounded_rate(len(merged), len(processed)),
        "memory_conflict_rate": _bounded_rate(len({stm_id for event in conflicted_events for stm_id in event.stm_ids}), len(processed)),
        "semantic_promotion_rate": (
            _bounded_rate(len([record for record in active_semantic if not record.metadata.get("generalized_from_stm_ids")]), len(semantic_candidates))
        ),
        # Denominator: all persisted generalized-rule outcomes.
        "semantic_generalization_rate": _bounded_rate(
            len(active_generalized),
            len(active_generalized) + len(generalization_proposals),
        ),
        "semantic_generalization_proposal_count": len(generalization_proposals),
        "semantic_generalization_recovery_count": len({
            event.memory_id for event in store.load_consolidation_events()
            if event.event_type == "generalization_recovered" and event.memory_id
        }),
        "active_semantic_rule_count": sum(
            1 for record in long_term
            if record.memory_type == "semantic_rule" and record.status == "active"
        ),
        "under_review_semantic_proposal_count": len(generalization_proposals),
        "average_independent_support": average_support,
        "active_memory_count": sum(1 for record in long_term if record.status == "active"),
        "under_review_memory_count": sum(1 for record in long_term if record.status == "under_review"),
        "deprecated_memory_count": sum(1 for record in long_term if record.status == "deprecated"),
        "failure_recurrence_rate": _failure_recurrence_rate(short_term),
    }


def _bounded_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return max(0.0, min(1.0, numerator / denominator))


def _failure_recurrence_rate(short_term) -> float | None:
    """Share of failure observations whose failure_type already occurred in an
    earlier, different case."""
    failure_rows = sorted(
        (row for row in short_term if row.memory_type == "failure" and row.failure_type),
        key=lambda row: row.created_at or "",
    )
    if not failure_rows:
        return None
    seen_cases_by_type: dict[str, set[str]] = {}
    recurrences = 0
    for row in failure_rows:
        earlier_cases = seen_cases_by_type.setdefault(row.failure_type, set())
        if earlier_cases - {row.source_case_id}:
            recurrences += 1
        earlier_cases.add(row.source_case_id)
    return recurrences / len(failure_rows)
