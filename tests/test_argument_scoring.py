from __future__ import annotations

from src.argumentation.argument_scorer import ArgumentScorer
from src.argumentation.argument_verifier import ArgumentVerifier
import src.main as main_module
from src.schemas.argument_schema import Argument
from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem, Provenance
from src.schemas.qbaf_schema import QBAFGraph
from src.schemas.report_schema import SubClaimReport

from tests.conftest import FakeLLMClient


def test_argument_scoring_uses_evidence_quality_and_invalid_multiplier():
    claim = SubClaim(claim_id="c1", claim_type="when", statement="Image is recent")
    evidence = [
        EvidenceItem(
            evidence_id="e1",
            source_type="case_provided",
            source="manual",
            content="The image was published two years ago.",
            reliability=0.9,
            relevance=0.8,
            provenance=Provenance(source_id="e1", source_type="case_provided", source="manual"),
        )
    ]
    scorer = ArgumentScorer(FakeLLMClient())
    valid = scorer.score(
        claim,
        Argument(argument_id="a1", claim_id="c1", stance="attack", title="old", text="The image is old.", evidence_ids=["e1"]),
        evidence,
        EvidenceGraph(),
    )
    invalid = scorer.score(
        claim,
        Argument(argument_id="a2", claim_id="c1", stance="attack", title="old", text="The image is old.", evidence_ids=["e1"], verifier_valid=False),
        evidence,
        EvidenceGraph(),
    )

    assert 0.0 < valid.score <= 1.0
    assert invalid.score < valid.score


def test_batched_argument_verifier_preserves_argument_order(monkeypatch):
    monkeypatch.setenv("SEMV_BATCH_ARGUMENT_VERIFICATION", "true")
    claim = SubClaim(claim_id="c1", claim_type="what", statement="Claim statement")
    evidence = [EvidenceItem(evidence_id="e1", content="Evidence supports one argument.")]
    arguments = [
        Argument(argument_id="a2", claim_id="c1", stance="attack", text="Second", evidence_ids=["e1"]),
        Argument(argument_id="a1", claim_id="c1", stance="support", text="First", evidence_ids=["e1"]),
    ]

    verified = ArgumentVerifier(FakeLLMClient()).verify_all(claim, arguments, evidence)

    assert [argument.argument_id for argument in verified] == ["a2", "a1"]
    assert verified[0].verifier_valid is False
    assert verified[1].verifier_valid is True
    assert verified[0].verification_notes == "Batch grounded 2."


def test_argument_parallel_verification_is_disabled_by_default(monkeypatch):
    monkeypatch.setenv("SEMV_BATCH_ARGUMENT_VERIFICATION", "false")
    monkeypatch.delenv("SEMV_PARALLEL_ARGUMENT_VERIFICATION", raising=False)
    claim = SubClaim(claim_id="c1", claim_type="what", statement="Claim statement")
    evidence = [EvidenceItem(evidence_id="e1", content="Evidence supports both arguments.")]
    arguments = [
        Argument(argument_id="a1", claim_id="c1", stance="support", text="First", evidence_ids=["e1"]),
        Argument(argument_id="a2", claim_id="c1", stance="attack", text="Second", evidence_ids=["e1"]),
    ]
    llm_client = FakeLLMClient()

    verified = ArgumentVerifier(llm_client).verify_all(claim, arguments, evidence)

    assert [argument.argument_id for argument in verified] == ["a1", "a2"]
    assert all(argument.verification_notes == "Grounded in linked evidence." for argument in verified)
    assert sum(1 for call in llm_client.calls if call[0] == "generate_json") == 2


def test_parallel_claim_results_preserve_claim_order(monkeypatch):
    monkeypatch.setenv("SEMV_PARALLEL_CLAIMS", "true")
    monkeypatch.setenv("SEMV_MAX_WORKERS", "2")
    claims = [
        SubClaim(claim_id="c1", claim_type="what", statement="First claim"),
        SubClaim(claim_id="c2", claim_type="where", statement="Second claim"),
    ]

    def fake_process_claim(**kwargs):
        claim = kwargs["claim"]
        argument = Argument(
            argument_id=f"arg_{claim.claim_id}",
            claim_id=claim.claim_id,
            stance="support",
            text=claim.statement,
        )
        return (
            [argument],
            QBAFGraph(claim_id=claim.claim_id),
            SubClaimReport(
                claim_id=claim.claim_id,
                claim_type=claim.claim_type,
                statement=claim.statement,
                score=0.5,
                decision="uncertain",
                confidence=1.0,
            ),
        )

    monkeypatch.setattr(main_module, "_process_claim", fake_process_claim)

    results = main_module._process_claims_parallel(
        claims=claims,
        normalized_evidence=[],
        evidence_graph=EvidenceGraph(),
        memory_by_claim={},
        bundle=None,
        evidence_ranker=None,
        argument_generator=None,
        argument_verifier=None,
        argument_scorer=None,
        graph_builder=None,
        propagator=None,
        clash_resolver=None,
        decision_mapper=None,
    )

    assert [report.claim_id for _, _, report in results] == ["c1", "c2"]
