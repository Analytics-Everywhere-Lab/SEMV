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


class RaisingLLM:
    def generate_json(self, prompt, **kwargs):
        raise RuntimeError("llm unavailable")


def test_fallback_tool_missing_uncertainty_is_neutral_not_attack():
    evidence = [
        EvidenceItem(
            evidence_id="unc1",
            source_type="synthetic_uncertainty",
            source="metadata",
            title="ExifTool metadata unavailable",
            content="ExifTool metadata unavailable; adapter did not run.",
            reliability=0.25,
            relevance=0.45,
            uncertainty_flags=["exiftool_missing"],
        )
    ]
    claim = SubClaim(claim_id="c1", claim_type="authenticity", statement="authentic?")
    arguments = ArgumentGenerator(RaisingLLM()).generate(claim, evidence, EvidenceGraph(), [])
    assert arguments[0].stance == "neutral"


def test_fallback_weak_metadata_flags_are_neutral_not_attack():
    evidence = [
        EvidenceItem(
            evidence_id="meta1",
            source_type="metadata_exiftool",
            source="exiftool",
            title="Image metadata",
            content="Metadata inspection for image.jpg.",
            reliability=0.6,
            relevance=0.5,
            uncertainty_flags=["gps_missing", "creation_time_missing"],
        )
    ]
    claim = SubClaim(claim_id="c1", claim_type="where", statement="location claim")
    arguments = ArgumentGenerator(RaisingLLM()).generate(claim, evidence, EvidenceGraph(), [])
    assert arguments[0].stance == "neutral"


def test_fallback_suspicious_forensic_flag_is_still_attack():
    evidence = [
        EvidenceItem(
            evidence_id="forensic1",
            source_type="metadata_exiftool",
            source="exiftool",
            title="Suspicious software tag",
            content="Metadata software tag indicates editing software.",
            reliability=0.6,
            relevance=0.5,
            uncertainty_flags=["software_tag_suspicious"],
        )
    ]
    claim = SubClaim(claim_id="c1", claim_type="authenticity", statement="authentic?")
    arguments = ArgumentGenerator(RaisingLLM()).generate(claim, evidence, EvidenceGraph(), [])
    assert arguments[0].stance == "attack"
