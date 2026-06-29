from __future__ import annotations

from src.argumentation.clash_resolver import ClashResolver
from src.qbaf.decision_mapper import DecisionMapper
from src.qbaf.graph_builder import QBAFGraphBuilder
from src.qbaf.propagator import QBAFPropagator
from src.schemas.argument_schema import Argument
from src.schemas.claim_schema import SubClaim

from tests.conftest import FakeLLMClient


def test_qbaf_support_minus_attack_maps_to_supported():
    claim = SubClaim(claim_id="c1", claim_type="authenticity", statement="Media is in context")
    arguments = [
        Argument(argument_id="a1", claim_id="c1", stance="support", title="s", text="support", score=0.8),
        Argument(argument_id="a2", claim_id="c1", stance="attack", title="a", text="attack", score=0.2),
    ]
    graph = QBAFPropagator().propagate(QBAFGraphBuilder().build(claim, arguments))

    assert graph.claim_score > 0.5
    assert DecisionMapper().map(graph.claim_score)[0] in {"weakly_supported", "supported"}


def test_attack_argument_lowers_claim_score():
    claim = SubClaim(claim_id="c1", claim_type="when", statement="The media was captured on the claimed date.")
    argument = Argument(
        argument_id="a1",
        case_id="case1",
        claim_id="c1",
        claim_type="when",
        text="Reverse search shows the media appeared earlier.",
        stance="attack",
        evidence_ids=["e1"],
        rationale="Earlier appearance refutes the claimed date.",
        intrinsic_score=0.9,
        score=0.9,
    )
    graph = QBAFPropagator().propagate(QBAFGraphBuilder().build(claim, [argument]))

    assert graph.claim_score < 0.5


def test_clash_resolver_detects_close_high_weight_opposition():
    claim = SubClaim(claim_id="c1", claim_type="what", statement="Media depicts event")
    arguments = [
        Argument(argument_id="a1", claim_id="c1", stance="support", title="s", text="support", score=0.7),
        Argument(argument_id="a2", claim_id="c1", stance="attack", title="a", text="attack", score=0.6),
    ]
    graph = QBAFPropagator().propagate(QBAFGraphBuilder().build(claim, arguments))

    assert ClashResolver(FakeLLMClient()).should_resolve(graph, arguments) is True
