from __future__ import annotations

from src.main import run_from_step
from src.qbaf.graph_builder import QBAFGraphBuilder
from src.qbaf.propagator import QBAFPropagator
from src.reporting.markdown_renderer import MarkdownRenderer
from src.schemas.argument_schema import Argument
from src.schemas.case_bundle_schema import CaseBundle, Claim, DatasetInfo, InputMetadata, RunConfig, TaskInfo
from src.schemas.case_trace_schema import CaseTrace
from src.schemas.claim_schema import SubClaim
from src.schemas.contestation_schema import HumanArgumentContestation, HumanReviewBatch, RevisionPlan
from src.schemas.evidence_schema import EvidenceItem
from src.schemas.report_schema import VerificationReport

from tests.conftest import FakeLLMClient


def _bundle() -> CaseBundle:
    return CaseBundle(
        case_id="case_1",
        dataset=DatasetInfo(dataset_name="unit"),
        task=TaskInfo(task_type="multimedia_verification", media_type="image"),
        input=InputMetadata(caption="claim"),
        claims=[Claim(claim_id="subclaim_1", claim_type="what", statement="Scene happened")],
        run_config=RunConfig(allow_web_search=False, allow_reverse_search=False, allow_memory_retrieval=False),
    )


def test_qbaf_excludes_rejected_argument():
    claim = SubClaim(claim_id="subclaim_1", claim_type="what", statement="Scene happened")
    args = [
        Argument(argument_id="arg_support", claim_id="subclaim_1", stance="support", text="strong", score=0.9, human_status="rejected"),
        Argument(argument_id="arg_attack", claim_id="subclaim_1", stance="attack", text="weak", score=0.1),
    ]
    graph = QBAFPropagator().propagate(QBAFGraphBuilder().build(claim, args))
    assert "arg_support" not in graph.nodes
    assert graph.claim_score < 0.5


def test_adaptive_rerun_from_qbaf_applies_human_review():
    trace = CaseTrace(
        case_id="case_1",
        subclaims=[SubClaim(claim_id="subclaim_1", claim_type="what", statement="Scene happened")],
        validated_evidence_items=[EvidenceItem(evidence_id="ev_1", content="known", reliability=0.9, relevance=0.9)],
        arguments=[Argument(argument_id="arg_support", claim_id="subclaim_1", stance="support", text="strong", evidence_ids=["ev_1"], score=0.9)],
    )
    batch = HumanReviewBatch(case_id="case_1", contestations=[HumanArgumentContestation(contestation_id="c1", case_id="case_1", action="reject", target_argument_id="arg_support", reason="unsupported evidence")])

    report = run_from_step(_bundle(), "qbaf_reasoning", previous_state=trace, human_review_batch=batch, llm_client=FakeLLMClient())

    assert report.subclaim_reports[0].top_support_arguments == []
    assert report.final_status in {"uncertain", "insufficient_evidence"}


def test_report_renders_human_contestation():
    batch = HumanReviewBatch(case_id="case_1", contestations=[HumanArgumentContestation(contestation_id="c1", case_id="case_1", action="reject", target_argument_id="arg_1", reason="bad evidence")])
    plan = RevisionPlan(case_id="case_1", revision_target="evidence_validation", rerun_from_step="evidence_validation", affected_argument_ids=["arg_1"], rationale="bad evidence")
    report = VerificationReport(
        case_id="case_1",
        final_status="uncertain",
        final_confidence=0.5,
        human_review_applied=True,
        human_review_batch=batch,
        revision_plan=plan,
        contestation_summary={"original_final_status": "verified", "revised_final_status": "uncertain", "original_confidence": 0.8, "revised_confidence": 0.5},
    )

    markdown = MarkdownRenderer().render(report)

    assert "Human Contestation and Adaptive Revision" in markdown
    assert "Human review applied: yes" in markdown
    assert "Revision Plan" in markdown
