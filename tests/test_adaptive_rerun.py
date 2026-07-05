from __future__ import annotations

from src.main import run_from_step
from src.qbaf.graph_builder import QBAFGraphBuilder
from src.qbaf.propagator import QBAFPropagator
from src.reporting.markdown_renderer import MarkdownRenderer
from src.retrieval.deep_researcher import DeepResearcher
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
    batch = HumanReviewBatch(case_id="case_1", contestations=[HumanArgumentContestation(contestation_id="c1", case_id="case_1", action="reject", target_argument_id="arg_support", reason="argument is weak")])

    report = run_from_step(_bundle(), "qbaf_reasoning", previous_state=trace, human_review_batch=batch, llm_client=FakeLLMClient())

    assert report.revision_plan.rerun_from_step in {"argument_construction", "qbaf_reasoning"}
    assert all(arg.argument_id != "arg_support" for arg in report.subclaim_reports[0].top_support_arguments)


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


def _two_claim_bundle() -> CaseBundle:
    return CaseBundle(
        case_id="case_1",
        dataset=DatasetInfo(dataset_name="unit"),
        task=TaskInfo(task_type="multimedia_verification", media_type="image"),
        input=InputMetadata(caption="claim"),
        claims=[
            Claim(claim_id="subclaim_1", claim_type="what", statement="Scene happened"),
            Claim(claim_id="subclaim_2", claim_type="where", statement="Scene location"),
        ],
        run_config=RunConfig(allow_web_search=False, allow_reverse_search=False, allow_memory_retrieval=False),
    )


def _two_claim_trace() -> CaseTrace:
    return CaseTrace(
        case_id="case_1",
        subclaims=[
            SubClaim(claim_id="subclaim_1", claim_type="what", statement="Scene happened"),
            SubClaim(claim_id="subclaim_2", claim_type="where", statement="Scene location"),
        ],
        validated_evidence_items=[
            EvidenceItem(evidence_id="ev_1", content="known evidence one", reliability=0.9, relevance=0.9),
            EvidenceItem(evidence_id="ev_2", content="known evidence two", reliability=0.9, relevance=0.9),
        ],
        arguments=[
            Argument(argument_id="arg_1", claim_id="subclaim_1", stance="support", text="strong", evidence_ids=["ev_1"], score=0.9),
            Argument(argument_id="arg_2", claim_id="subclaim_2", stance="support", text="strong", evidence_ids=["ev_2"], score=0.9),
        ],
    )


def test_qbaf_only_contestation_does_not_call_retrieval(monkeypatch):
    def _raise_if_called(self, claim, plan, existing_evidence):
        raise AssertionError("DeepResearcher.research must not be called for a qbaf-only contestation")

    monkeypatch.setattr(DeepResearcher, "research", _raise_if_called)

    trace = _two_claim_trace()
    batch = HumanReviewBatch(
        case_id="case_1",
        contestations=[
            HumanArgumentContestation(
                contestation_id="c1",
                case_id="case_1",
                action="reject",
                target_argument_id="arg_1",
                reason="argument is weak",
            )
        ],
    )

    report = run_from_step(_two_claim_bundle(), "qbaf_reasoning", previous_state=trace, human_review_batch=batch, llm_client=FakeLLMClient())

    assert report.revision_plan.rerun_from_step in {"argument_construction", "qbaf_reasoning"}
    assert report.human_review_applied is True


def test_unsupported_evidence_reruns_validation_not_retrieval(monkeypatch):
    def _raise_if_called(self, claim, plan, existing_evidence):
        raise AssertionError("DeepResearcher.research must not be called for evidence_validation")

    monkeypatch.setattr(DeepResearcher, "research", _raise_if_called)

    trace = _two_claim_trace()
    batch = HumanReviewBatch(
        case_id="case_1",
        contestations=[
            HumanArgumentContestation(
                contestation_id="c1",
                case_id="case_1",
                action="reject",
                target_argument_id="arg_1",
                reason="The evidence does not support this claim.",
            )
        ],
    )

    report = run_from_step(_two_claim_bundle(), "evidence_validation", previous_state=trace, human_review_batch=batch, llm_client=FakeLLMClient())

    assert report.revision_plan.rerun_from_step == "evidence_validation"
    assert report.human_review_applied is True
    assert report.subclaim_reports


def test_wrong_source_reruns_retrieval_only_for_affected_claim(monkeypatch):
    calls: list[str] = []

    def _tracking_research(self, claim, plan, existing_evidence):
        calls.append(claim.claim_id)
        return [EvidenceItem(evidence_id="ev_new", content="freshly retrieved", reliability=0.8, relevance=0.8)]

    monkeypatch.setattr(DeepResearcher, "research", _tracking_research)

    trace = _two_claim_trace()
    batch = HumanReviewBatch(
        case_id="case_1",
        contestations=[
            HumanArgumentContestation(
                contestation_id="c1",
                case_id="case_1",
                action="reject",
                target_argument_id="arg_1",
                reason="retrieval error returned the wrong source",
            )
        ],
    )

    report = run_from_step(_two_claim_bundle(), "evidence_retrieval", previous_state=trace, human_review_batch=batch, llm_client=FakeLLMClient())

    assert report.revision_plan.rerun_from_step == "evidence_retrieval"
    assert calls == ["subclaim_1"]

    evidence_ids = {item.evidence_id for item in report.evidence}
    assert "ev_new" in evidence_ids

    subclaim_2_report = next(r for r in report.subclaim_reports if r.claim_id == "subclaim_2")
    subclaim_2_arg_ids = {arg.argument_id for arg in [*subclaim_2_report.top_support_arguments, *subclaim_2_report.top_attack_arguments]}
    assert "arg_2" in subclaim_2_arg_ids or subclaim_2_arg_ids == set()


def test_selective_rerun_leaves_unaffected_claim_untouched(monkeypatch):
    def _no_research(self, claim, plan, existing_evidence):
        return []

    monkeypatch.setattr(DeepResearcher, "research", _no_research)

    trace = _two_claim_trace()
    batch = HumanReviewBatch(
        case_id="case_1",
        contestations=[
            HumanArgumentContestation(
                contestation_id="c1",
                case_id="case_1",
                action="edit",
                target_argument_id="arg_1",
                edited_text="clearer wording",
            )
        ],
    )

    report = run_from_step(_two_claim_bundle(), "argument_construction", previous_state=trace, human_review_batch=batch, llm_client=FakeLLMClient())

    assert report.revision_plan.affected_subclaim_ids == ["subclaim_1"]
    claim_ids = {subclaim.claim_id for subclaim in report.subclaim_reports}
    assert {"subclaim_1", "subclaim_2"}.issubset(claim_ids)

    subclaim_2_report = next(r for r in report.subclaim_reports if r.claim_id == "subclaim_2")
    subclaim_2_arg_ids = {arg.argument_id for arg in [*subclaim_2_report.top_support_arguments, *subclaim_2_report.top_attack_arguments]}
    assert subclaim_2_arg_ids == {"arg_2"}


def test_human_added_argument_is_verified_and_scored():
    trace = _two_claim_trace()
    batch = HumanReviewBatch(
        case_id="case_1",
        contestations=[
            HumanArgumentContestation(
                contestation_id="c1",
                case_id="case_1",
                action="add",
                added_subclaim_id="subclaim_1",
                added_text="A newly observed corroborating detail.",
                added_stance="support",
                added_evidence_ids=["ev_1"],
            )
        ],
    )

    report = run_from_step(_two_claim_bundle(), "argument_construction", previous_state=trace, human_review_batch=batch, llm_client=FakeLLMClient())

    subclaim_1_report = next(r for r in report.subclaim_reports if r.claim_id == "subclaim_1")
    all_args = [*subclaim_1_report.top_support_arguments, *subclaim_1_report.top_attack_arguments]
    added = [arg for arg in all_args if arg.human_status == "added"]
    assert added, "human-added argument should remain present in the final arguments"
    assert added[0].score is not None
