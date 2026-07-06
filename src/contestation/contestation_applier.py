from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.schemas.argument_schema import Argument
from src.schemas.contestation_schema import (
    ArgumentProvenance,
    HumanArgumentContestation,
    HumanReviewBatch,
)


def apply_human_contestations(
    arguments: list[Argument],
    review_batch: HumanReviewBatch | None,
) -> list[Argument]:
    if review_batch is None or not review_batch.contestations:
        return arguments

    result = list(arguments)
    for contestation in review_batch.contestations:
        if contestation.action == "accept":
            result = _replace_target(result, contestation, {"human_status": "accepted"})
        elif contestation.action == "reject":
            result = _replace_target(result, contestation, {"human_status": "rejected"})
        elif contestation.action == "edit":
            result = _apply_edit(result, contestation)
        elif contestation.action == "add":
            result.append(_human_added_argument(contestation, review_batch))
    return result


def filter_arguments_for_qbaf(
    arguments: Iterable[Argument],
    exclude_rejected_arguments: bool = True,
) -> list[Argument]:
    argument_list = list(arguments)
    edited_original_ids = {
        argument.human_original_argument_id
        for argument in argument_list
        if argument.human_original_argument_id
    }
    filtered = []
    for argument in argument_list:
        if exclude_rejected_arguments and argument.human_status == "rejected":
            continue
        if (
            argument.human_status == "edited"
            and argument.human_original_argument_id is None
            and argument.argument_id in edited_original_ids
        ):
            continue
        filtered.append(argument)
    return filtered


def human_status_counts(arguments: Iterable[Argument]) -> dict[str, int]:
    counts = {"accepted": 0, "rejected": 0, "edited": 0, "added": 0, "unreviewed": 0}
    for argument in arguments:
        counts[argument.human_status] = counts.get(argument.human_status, 0) + 1
    return counts


def contestation_summary(
    original_arguments: list[Argument],
    reviewed_arguments: list[Argument],
    review_batch: HumanReviewBatch | None,
) -> dict[str, Any]:
    original_ids = {argument.argument_id for argument in original_arguments}
    reviewed_ids = {argument.argument_id for argument in reviewed_arguments}
    return {
        "human_review_applied": review_batch is not None,
        "action_counts": _action_counts(review_batch),
        "status_counts": human_status_counts(reviewed_arguments),
        "changed_arguments": sorted(
            argument.argument_id
            for argument in reviewed_arguments
            if argument.human_status in {"accepted", "rejected", "edited", "added"}
        ),
        "removed_arguments": sorted(
            argument.argument_id
            for argument in reviewed_arguments
            if argument.human_status == "rejected"
        ),
        "added_arguments": sorted(reviewed_ids - original_ids),
        "edited_arguments": sorted(
            argument.argument_id
            for argument in reviewed_arguments
            if argument.human_status == "edited"
        ),
    }


def _replace_target(
    arguments: list[Argument],
    contestation: HumanArgumentContestation,
    update: dict[str, Any],
) -> list[Argument]:
    target_id = contestation.target_argument_id
    return [
        argument.model_copy(update=update) if argument.argument_id == target_id else argument
        for argument in arguments
    ]


def _apply_edit(
    arguments: list[Argument],
    contestation: HumanArgumentContestation,
) -> list[Argument]:
    target_id = contestation.target_argument_id
    if not target_id:
        return arguments

    result: list[Argument] = []
    for argument in arguments:
        if argument.argument_id != target_id:
            result.append(argument)
            continue

        original = argument.model_copy(
            update={
                "human_status": "edited",
                "metadata": {
                    **argument.metadata,
                    "human_edit_original": True,
                    "contestation_id": contestation.contestation_id,
                },
            }
        )
        result.append(original)

        replacement_id = _unique_argument_id(f"{argument.argument_id}_human", arguments)
        edited_evidence_ids = contestation.edited_evidence_ids
        if edited_evidence_ids is None:
            edited_evidence_ids = contestation.metadata.get("edited_evidence_ids")
        replacement_metadata = {
            **argument.metadata,
            "human_edit_replacement": True,
            "contestation_id": contestation.contestation_id,
            "human_reason": contestation.reason,
        }
        if edited_evidence_ids is not None:
            replacement_metadata["pre_contestation_evidence_ids"] = list(argument.evidence_ids)
        replacement = argument.model_copy(
            update={
                "argument_id": replacement_id,
                "text": contestation.edited_text or argument.text,
                "stance": _valid_stance(contestation.edited_stance) or argument.stance,
                "evidence_ids": (
                    list(edited_evidence_ids)
                    if edited_evidence_ids is not None
                    else argument.evidence_ids
                ),
                "intrinsic_score": (
                    contestation.edited_confidence
                    if contestation.edited_confidence is not None
                    else argument.intrinsic_score
                ),
                "score": (
                    contestation.edited_confidence
                    if contestation.edited_confidence is not None
                    else argument.score
                ),
                "human_status": "edited",
                "human_original_argument_id": argument.argument_id,
                "metadata": replacement_metadata,
            }
        )
        result.append(replacement)
    return result


def _human_added_argument(
    contestation: HumanArgumentContestation,
    review_batch: HumanReviewBatch,
) -> Argument:
    subclaim_id = contestation.added_subclaim_id or "human_added_subclaim"
    evidence_ids = list(contestation.added_evidence_ids)
    argument_id = contestation.metadata.get("argument_id") or f"{contestation.contestation_id}_human_arg"
    return Argument(
        argument_id=argument_id,
        claim_id=subclaim_id,
        case_id=review_batch.case_id,
        stance=_valid_stance(contestation.added_stance) or "neutral",
        title="Human added argument",
        text=contestation.added_text or "",
        evidence_ids=evidence_ids,
        intrinsic_score=contestation.metadata.get("confidence", 0.7),
        score=contestation.metadata.get("confidence", 0.7),
        provenance=ArgumentProvenance(
            source_step="argument_construction",
            subclaim_id=subclaim_id,
            evidence_ids=evidence_ids,
            metadata={"human_added": True},
        ),
        human_status="added",
        metadata={
            "human_added": True,
            "contestation_id": contestation.contestation_id,
            "human_reason": contestation.reason,
        },
    )


def _valid_stance(value: str | None) -> str | None:
    if value in {"support", "attack", "mixed", "neutral"}:
        return value
    return None


def _unique_argument_id(seed: str, arguments: list[Argument]) -> str:
    existing = {argument.argument_id for argument in arguments}
    if seed not in existing:
        return seed
    index = 2
    while f"{seed}_{index}" in existing:
        index += 1
    return f"{seed}_{index}"


def _action_counts(review_batch: HumanReviewBatch | None) -> dict[str, int]:
    counts = {"accept": 0, "reject": 0, "edit": 0, "add": 0}
    if review_batch is None:
        return counts
    for contestation in review_batch.contestations:
        counts[contestation.action] = counts.get(contestation.action, 0) + 1
    return counts
