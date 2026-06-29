from __future__ import annotations

from src.argumentation.argument_scorer import ArgumentScorer
from src.schemas.argument_schema import Argument
from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem, Provenance

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
