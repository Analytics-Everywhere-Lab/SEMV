from __future__ import annotations

from src.schemas.argument_schema import Argument
from src.schemas.case_trace_schema import CaseTrace
from src.schemas.contestation_schema import HumanReviewBatch, RevisionPlan, RevisionTarget


STEP_ORDER: list[RevisionTarget] = [
    "claim_decomposition",
    "evidence_retrieval",
    "evidence_validation",
    "argument_construction",
    "qbaf_reasoning",
    "final_aggregation",
    "report_generation",
]

_EVIDENCE_RETRIEVAL_TERMS = {
    "retrieval error",
    "missing source",
    "missing evidence",
    "new external evidence",
    "new source",
    "source mismatch",
    "wrong source",
}

_EVIDENCE_VALIDATION_TERMS = {
    "wrong evidence",
    "unsupported evidence",
    "does not support",
    "irrelevant source",
    "wrong date",
    "wrong location",
    "wrong entity",
}


def route_revision(
    review_batch: HumanReviewBatch,
    current_arguments: list[Argument] | None = None,
    case_trace: CaseTrace | None = None,
    final_still_contested: bool = False,
) -> RevisionPlan:
    current_arguments = current_arguments or _arguments_from_trace(case_trace)
    arguments_by_id = {argument.argument_id: argument for argument in current_arguments}
    existing_evidence_ids = _existing_evidence_ids(current_arguments, case_trace)

    selected_targets: list[RevisionTarget] = []
    affected_argument_ids: set[str] = set()
    affected_subclaim_ids: set[str] = set()
    affected_evidence_ids: set[str] = set()
    affected_query_ids: set[str] = set()
    rationale_parts: list[str] = []

    for contestation in review_batch.contestations:
        target = _target_for_contestation(contestation, arguments_by_id, existing_evidence_ids)
        selected_targets.append(target)
        rationale_parts.append(
            f"{contestation.action} {contestation.target_argument_id or contestation.added_subclaim_id or contestation.contestation_id} -> {target}"
        )

        if contestation.target_argument_id:
            affected_argument_ids.add(contestation.target_argument_id)
            argument = arguments_by_id.get(contestation.target_argument_id)
            if argument:
                affected_subclaim_ids.add(argument.claim_id)
                affected_evidence_ids.update(argument.evidence_ids)
                if argument.provenance:
                    affected_evidence_ids.update(argument.provenance.evidence_ids)
                    affected_query_ids.update(argument.provenance.retrieval_query_ids)
        if contestation.added_subclaim_id:
            affected_subclaim_ids.add(contestation.added_subclaim_id)
        affected_evidence_ids.update(contestation.added_evidence_ids)

    if final_still_contested:
        selected_targets.append("qbaf_reasoning")
        rationale_parts.append("final decision remains contested -> qbaf_reasoning")

    rerun_from_step = _earliest(selected_targets or ["report_generation"])
    return RevisionPlan(
        case_id=review_batch.case_id,
        revision_target=rerun_from_step,
        rerun_from_step=rerun_from_step,
        affected_argument_ids=sorted(affected_argument_ids),
        affected_subclaim_ids=sorted(affected_subclaim_ids),
        affected_evidence_ids=sorted(affected_evidence_ids),
        affected_retrieval_query_ids=sorted(affected_query_ids),
        human_actions=[contestation.action for contestation in review_batch.contestations],
        rationale="; ".join(rationale_parts) or "No human contestation required upstream revision.",
        metadata={"final_still_contested": final_still_contested},
    )


def _target_for_contestation(contestation, arguments_by_id, existing_evidence_ids) -> RevisionTarget:
    reason = (contestation.reason or "").lower()
    if contestation.action == "accept":
        return "qbaf_reasoning"

    if contestation.action == "reject":
        if any(term in reason for term in _EVIDENCE_RETRIEVAL_TERMS):
            return "evidence_retrieval"
        if any(term in reason for term in _EVIDENCE_VALIDATION_TERMS):
            return "evidence_validation"
        return "argument_construction"

    if contestation.action == "edit":
        argument = arguments_by_id.get(contestation.target_argument_id)
        known_evidence = set(argument.evidence_ids) if argument else set()
        edited_evidence = set(contestation.metadata.get("edited_evidence_ids", []))
        if edited_evidence and not edited_evidence.issubset(known_evidence):
            return "evidence_retrieval"
        return "argument_construction"

    if contestation.action == "add":
        added_evidence = set(contestation.added_evidence_ids)
        if added_evidence and not added_evidence.issubset(existing_evidence_ids):
            return "evidence_retrieval"
        added_text = (contestation.added_text or "").lower()
        if any(term in added_text for term in {"http://", "https://", "new source", "independent source"}):
            return "evidence_retrieval"
        return "argument_construction"

    return "argument_construction"


def _earliest(targets: list[RevisionTarget]) -> RevisionTarget:
    return min(targets, key=STEP_ORDER.index)


def _arguments_from_trace(case_trace: CaseTrace | None) -> list[Argument]:
    if case_trace is None:
        return []
    return [
        item if isinstance(item, Argument) else Argument.model_validate(item)
        for item in case_trace.arguments
    ]


def _existing_evidence_ids(
    arguments: list[Argument],
    case_trace: CaseTrace | None,
) -> set[str]:
    evidence_ids = {evidence_id for argument in arguments for evidence_id in argument.evidence_ids}
    if case_trace is not None:
        for item in [*case_trace.evidence_items, *case_trace.validated_evidence_items]:
            if isinstance(item, dict):
                evidence_id = item.get("evidence_id")
            else:
                evidence_id = getattr(item, "evidence_id", None)
            if evidence_id:
                evidence_ids.add(evidence_id)
    return evidence_ids
