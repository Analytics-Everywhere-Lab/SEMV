from __future__ import annotations

from src.main import run_case
from src.reporting.json_renderer import JSONRenderer
from src.reporting.markdown_renderer import MarkdownRenderer
from src.schemas.case_schema import MultimediaCase
from src.utils.io import project_root, read_json

from tests.conftest import FakeLLMClient


def test_pipeline_end_to_end_inference_only_writes_reports(tmp_path):
    case_path = project_root() / "data" / "cases" / "sample_case.json"
    case = MultimediaCase.model_validate(read_json(case_path))

    report = run_case(
        case=case,
        mode="inference_only",
        llm_client=FakeLLMClient(),
        case_path=case_path,
    )

    assert report.case_id == case.case_id
    assert report.final_status in {"verified", "mostly_verified", "partially_verified", "false_context", "out_of_context_cheapfake", "manipulated_or_synthetic", "uncertain", "insufficient_evidence", "not_applicable"}
    assert len(report.subclaim_reports) == 6
    assert report.memory_update_candidates == []
    assert any(item.provenance for item in report.evidence)

    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    JSONRenderer().render_to_file(report, json_path)
    MarkdownRenderer().render_to_file(report, md_path)

    assert json_path.exists()
    assert "Verification Report" in md_path.read_text(encoding="utf-8")
