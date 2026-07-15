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

    metrics = {
        "memory_retrieval_rate": sum(1 for count in retrieved if count > 0) / len(retrieved) if retrieved else 0.0,
        "average_memories_retrieved_per_case": sum(retrieved) / len(retrieved) if retrieved else 0.0,
        "memory_citation_rate": len(used_cases) / len(predictions) if predictions else 0.0,
        "memory_usage_rate": len(used_cases) / len(predictions) if predictions else 0.0,
        "memory_associated_correctness_rate": (
            len(helpful) / len(used_cases) if used_cases else None
        ),
        "memory_associated_error_rate": len(harmful) / len(used_cases) if used_cases else None,
        # Association is not causation: without a paired memory-off run this
        # must not be reported as negative transfer.
        "negative_transfer_rate": _paired_negative_transfer(
            case_metrics, paired_baseline_case_metrics
        ),
    }
    metrics.update(_store_metrics(store))
    return metrics


def _paired_negative_transfer(
    case_metrics: list[dict],
    baseline_case_metrics: list[dict] | None,
) -> float | None:
    """Fraction of paired cases correct without memory but wrong with memory."""
    if not baseline_case_metrics:
        return None
    baseline_by_case = {row["case_id"]: row.get("final_label_correct") for row in baseline_case_metrics}
    paired = [
        (row.get("final_label_correct"), baseline_by_case[row["case_id"]])
        for row in case_metrics
        if row["case_id"] in baseline_by_case
    ]
    if not paired:
        return None
    flipped_bad = sum(1 for with_mem, without_mem in paired if without_mem is True and with_mem is False)
    return flipped_bad / len(paired)


def _store_metrics(store) -> dict:
    if store is None:
        return {
            "lesson_acceptance_rate": None,
            "stm_to_ltm_promotion_rate": None,
            "merge_rate": None,
            "memory_conflict_rate": None,
            "semantic_promotion_rate": None,
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

    semantic_promoted = [
        record
        for record in long_term
        if record.memory_type == "semantic_rule" and record.origin == "consolidated"
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
            _bounded_rate(len([record for record in semantic_promoted if not record.metadata.get("generalized_from_stm_ids")]), len(semantic_candidates))
        ),
        "semantic_generalization_rate": _bounded_rate(
            len([record for record in semantic_promoted if record.metadata.get("generalized_from_stm_ids")]),
            len([row for row in short_term if row.memory_type in {"episodic", "failure"}]),
        ),
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
