from __future__ import annotations

from src.evaluation.mv2026_gold_parser import parse_mv2026_gold_report


def test_mv2026_gold_parser_extracts_status_urls_and_sections(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(
        "# Case Summary\nA verified case.\n# Final Verification Status\nStatus: verified\n# Where\nLocation 44.65, -63.57\n# Other Evidence & Findings\nhttps://example.com/source",
        encoding="utf-8",
    )

    gold = parse_mv2026_gold_report(report, case_id="ID1")

    assert gold.case_id == "ID1"
    assert gold.gold_final_label == "verified"
    assert "https://example.com/source" in gold.gold_source_urls
    assert gold.gold_coordinates[0]["latitude"] == 44.65
