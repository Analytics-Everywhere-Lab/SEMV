from __future__ import annotations

from src.argumentation.uncertainty_escalator import UncertaintyEscalator
from src.schemas.argument_schema import Argument
from src.schemas.evidence_schema import EvidenceItem


def test_neutral_score_triggers_escalation():
    decisions = UncertaintyEscalator().evaluate({"c1": 0.5}, [], [])
    assert decisions[0].should_escalate
    assert "neutral_score_band" in decisions[0].reason_codes


def test_strong_score_does_not_escalate_with_good_evidence():
    evidence = [EvidenceItem(evidence_id="e1", source_type="case_provided", source="s", content="ok", reliability=0.8)]
    args = [Argument(argument_id="a", claim_id="c1", stance="support", text="ok", evidence_ids=["e1"], score=0.9)]
    decisions = UncertaintyEscalator().evaluate({"c1": 0.9}, args, evidence)
    assert not decisions[0].should_escalate


def test_all_synthetic_evidence_triggers_escalation():
    evidence = [EvidenceItem(evidence_id="e1", source_type="synthetic_uncertainty", source="s", content="missing")]
    args = [Argument(argument_id="a", claim_id="c1", stance="support", text="ok", evidence_ids=["e1"])]
    decisions = UncertaintyEscalator().evaluate({"c1": 0.8}, args, evidence)
    assert "all_synthetic_uncertainty" in decisions[0].reason_codes


def test_reverse_old_reuse_conflict_triggers_escalation():
    evidence = [EvidenceItem(evidence_id="e1", source_type="reverse_image_local", source="s", content="previous old match")]
    args = [Argument(argument_id="a", claim_id="c1", stance="attack", text="old", evidence_ids=["e1"])]
    decisions = UncertaintyEscalator().evaluate({"c1": 0.8}, args, evidence)
    assert "reverse_search_context_conflict" in decisions[0].reason_codes


def test_forensic_source_conflict_triggers_escalation():
    evidence = [
        EvidenceItem(evidence_id="f", source_type="forensic_analysis", source="s", content="caution", uncertainty_flags=["software_tag_suspicious"]),
        EvidenceItem(evidence_id="w", source_type="case_provided", source="s", content="authentic", metadata={"stance": "support"}),
    ]
    args = [Argument(argument_id="a", claim_id="c1", stance="support", text="x", evidence_ids=["f", "w"])]
    decisions = UncertaintyEscalator().evaluate({"c1": 0.8}, args, evidence)
    assert "forensic_source_conflict" in decisions[0].reason_codes
