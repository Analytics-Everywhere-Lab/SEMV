from __future__ import annotations

from src.reporting.markdown_renderer import MarkdownRenderer
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem
from src.schemas.report_schema import VerificationReport


def test_markdown_has_media_analysis_and_escalation_sections():
    report = VerificationReport(
        case_id="c1",
        final_status="uncertain",
        final_confidence=0.5,
        evidence=[EvidenceItem(evidence_id="m", source_type="metadata_exiftool", source="img", content="metadata")],
        evidence_graph=EvidenceGraph(),
        escalation=[{"claim_id": "c1", "should_escalate": True, "reason_codes": ["neutral_score_band"], "affected_pipeline_stages": ["qbaf_reasoning"]}],
    )
    markdown = MarkdownRenderer().render(report)
    assert "## Media Analysis" in markdown
    assert "### Metadata" in markdown
    assert "## Escalation / Human Review" in markdown
    assert "neutral_score_band" in markdown
