from __future__ import annotations

from src.contestation.revision_router import route_revision
from src.schemas.argument_schema import Argument
from src.schemas.case_trace_schema import CaseTrace
from src.schemas.contestation_schema import HumanArgumentContestation, HumanReviewBatch
from src.schemas.evidence_schema import EvidenceItem


def _trace() -> CaseTrace:
    return CaseTrace(
        case_id="case_1",
        evidence_items=[EvidenceItem(evidence_id="ev_1", content="known")],
        arguments=[Argument(argument_id="arg_1", claim_id="subclaim_1", stance="support", text="a", evidence_ids=["ev_1"])],
    )


def test_reject_bad_evidence_routes_to_validation_or_earlier():
    batch = HumanReviewBatch(case_id="case_1", contestations=[HumanArgumentContestation(contestation_id="c1", case_id="case_1", action="reject", target_argument_id="arg_1", reason="The evidence source does not support this claim.")])
    plan = route_revision(batch, case_trace=_trace())
    assert plan.rerun_from_step in {"evidence_validation", "evidence_retrieval"}


def test_edit_wording_routes_to_argument_construction():
    batch = HumanReviewBatch(case_id="case_1", contestations=[HumanArgumentContestation(contestation_id="c1", case_id="case_1", action="edit", target_argument_id="arg_1", edited_text="clearer wording")])
    assert route_revision(batch, case_trace=_trace()).rerun_from_step == "argument_construction"


def test_add_existing_evidence_routes_to_argument_construction():
    batch = HumanReviewBatch(case_id="case_1", contestations=[HumanArgumentContestation(contestation_id="c1", case_id="case_1", action="add", added_subclaim_id="subclaim_1", added_text="new arg", added_stance="support", added_evidence_ids=["ev_1"])])
    assert route_revision(batch, case_trace=_trace()).rerun_from_step == "argument_construction"


def test_add_new_evidence_routes_to_retrieval():
    batch = HumanReviewBatch(case_id="case_1", contestations=[HumanArgumentContestation(contestation_id="c1", case_id="case_1", action="add", added_subclaim_id="subclaim_1", added_text="new arg", added_stance="support", added_evidence_ids=["ev_new"])])
    assert route_revision(batch, case_trace=_trace()).rerun_from_step == "evidence_retrieval"


def test_multiple_contestations_choose_earliest_step():
    batch = HumanReviewBatch(case_id="case_1", contestations=[
        HumanArgumentContestation(contestation_id="c1", case_id="case_1", action="edit", target_argument_id="arg_1", edited_text="clearer"),
        HumanArgumentContestation(contestation_id="c2", case_id="case_1", action="reject", target_argument_id="arg_1", reason="retrieval error returned the wrong source"),
    ])
    assert route_revision(batch, case_trace=_trace()).rerun_from_step == "evidence_retrieval"


def test_explicit_revision_target_metadata_overrides_inference():
    batch = HumanReviewBatch(case_id="case_1", contestations=[
        HumanArgumentContestation(
            contestation_id="c1",
            case_id="case_1",
            action="accept",
            target_argument_id="arg_1",
            metadata={"revision_target": "evidence_retrieval"},
        ),
    ])
    assert route_revision(batch, case_trace=_trace()).rerun_from_step == "evidence_retrieval"


def test_invalid_revision_target_metadata_falls_back_to_inference():
    batch = HumanReviewBatch(case_id="case_1", contestations=[
        HumanArgumentContestation(
            contestation_id="c1",
            case_id="case_1",
            action="accept",
            target_argument_id="arg_1",
            metadata={"revision_target": "not_a_real_step"},
        ),
    ])
    assert route_revision(batch, case_trace=_trace()).rerun_from_step == "qbaf_reasoning"
