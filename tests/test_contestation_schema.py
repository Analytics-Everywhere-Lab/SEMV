from __future__ import annotations

from src.schemas.contestation_schema import (
    ArgumentProvenance,
    HumanArgumentContestation,
    HumanReviewBatch,
    RevisionPlan,
)


def test_contestation_schemas_validate_defaults():
    provenance = ArgumentProvenance(
        source_step="argument_construction",
        subclaim_id="subclaim_1",
        evidence_ids=["ev_1"],
    )
    batch = HumanReviewBatch(
        case_id="case_1",
        contestations=[
            HumanArgumentContestation(
                contestation_id="c1",
                case_id="case_1",
                action="reject",
                target_argument_id="arg_1",
            )
        ],
    )
    plan = RevisionPlan(
        case_id="case_1",
        revision_target="evidence_validation",
        rerun_from_step="evidence_validation",
        rationale="Bad evidence was contested.",
    )

    assert provenance.upstream_steps == []
    assert batch.contestations[0].action == "reject"
    assert plan.affected_argument_ids == []
