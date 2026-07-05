from __future__ import annotations

from src.retrieval.deep_researcher import build_queries_from_evidence
from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem


def _claim(kind="where"):
    return SubClaim(claim_id="c1", claim_type=kind, statement="Where was this recorded?")


def test_ocr_evidence_contributes_query():
    evidence = [EvidenceItem(evidence_id="e1", source_type="ocr", content="Visible text: Halifax Harbour", source="img")]
    queries = build_queries_from_evidence(_claim("where"), ResearchPlan(claim_id="c1"), evidence)
    assert any("Halifax Harbour" in query for query in queries)


def test_asr_evidence_contributes_query():
    evidence = [EvidenceItem(evidence_id="e1", source_type="asr", content="This happened on July 4 in Halifax", source="vid")]
    queries = build_queries_from_evidence(_claim("when"), ResearchPlan(claim_id="c1"), evidence)
    assert any("July" in query or "202" in query or "This" in query for query in queries)


def test_vlm_search_queries_are_deduped():
    evidence = [EvidenceItem(evidence_id="e1", source_type="frame_analysis", content="scene", source="img", raw_output={"search_queries": ["same query", "same query"]})]
    queries = build_queries_from_evidence(_claim("what"), ResearchPlan(claim_id="c1", search_queries=["same query"]), evidence)
    assert queries.count("same query") == 1
