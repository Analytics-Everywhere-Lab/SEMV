from __future__ import annotations

from src.argumentation.argument_generator import ArgumentGenerator, intrinsic_strength_from_components
from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
    def generate_json(self, prompt, **kwargs):
        return self.payload


def test_invalid_evidence_id_is_repaired():
    evidence = [EvidenceItem(evidence_id="e1", source_type="case_provided", source="s", content="The image is old.")]
    llm = FakeLLM({"arguments": [{"stance": "attack", "text": "The image is old.", "evidence_ids": ["bad"]}]})
    claim = SubClaim(claim_id="c1", claim_type="authenticity", statement="authentic?")
    argument = ArgumentGenerator(llm).generate(claim, evidence, EvidenceGraph(), [])[0]
    assert argument.evidence_ids == ["e1"]
    assert "invalid_evidence_id_repaired" in argument.uncertainty_flags


def test_strength_components_sum_into_intrinsic_strength():
    components = {"source_reliability": 1.0, "cross_source_corroboration": 1.0, "cross_modal_consistency": 1.0, "claim_relevance": 1.0}
    assert intrinsic_strength_from_components(components) == 1.0


def test_contestability_is_present():
    evidence = [EvidenceItem(evidence_id="e1", source_type="case_provided", source="s", content="ok")]
    llm = FakeLLM({"arguments": [{"stance": "support", "text": "ok", "evidence_ids": ["e1"]}]})
    claim = SubClaim(claim_id="c1", claim_type="what", statement="what?")
    argument = ArgumentGenerator(llm).generate(claim, evidence, EvidenceGraph(), [])[0]
    assert argument.contestability["can_reject"] is True
    assert argument.provenance_summary
