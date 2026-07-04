from __future__ import annotations

from src.contestation.contestation_applier import apply_human_contestations, filter_arguments_for_qbaf
from src.schemas.argument_schema import Argument
from src.schemas.contestation_schema import HumanArgumentContestation, HumanReviewBatch


def test_contestation_applier_accept_reject_edit_add():
    args = [
        Argument(argument_id="arg_1", claim_id="subclaim_1", stance="support", text="one", evidence_ids=["ev_1"]),
        Argument(argument_id="arg_2", claim_id="subclaim_1", stance="attack", text="two", evidence_ids=["ev_2"]),
    ]
    batch = HumanReviewBatch(
        case_id="case_1",
        contestations=[
            HumanArgumentContestation(contestation_id="c1", case_id="case_1", action="accept", target_argument_id="arg_1"),
            HumanArgumentContestation(contestation_id="c2", case_id="case_1", action="reject", target_argument_id="arg_2"),
            HumanArgumentContestation(contestation_id="c3", case_id="case_1", action="edit", target_argument_id="arg_1", edited_text="better"),
            HumanArgumentContestation(contestation_id="c4", case_id="case_1", action="add", added_subclaim_id="subclaim_1", added_text="human", added_stance="attack", added_evidence_ids=["ev_2"]),
        ],
    )

    reviewed = apply_human_contestations(args, batch)

    assert any(arg.argument_id == "arg_1_human" and arg.text == "better" for arg in reviewed)
    assert any(arg.human_status == "rejected" for arg in reviewed if arg.argument_id == "arg_2")
    assert any(arg.human_status == "added" for arg in reviewed)
    included_ids = {arg.argument_id for arg in filter_arguments_for_qbaf(reviewed)}
    assert "arg_2" not in included_ids
    assert "arg_1_human" in included_ids
