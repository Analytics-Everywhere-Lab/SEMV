from __future__ import annotations

import json

from scripts import check_case_outputs


def test_check_case_outputs_accepts_normalized_evidence_ids(tmp_path, monkeypatch, capsys):
    case_dir = tmp_path / "case1"
    case_dir.mkdir()
    (case_dir / "raw_evidence.json").write_text(
        json.dumps([{"evidence_id": "raw1", "source_type": "metadata_ffprobe"}]),
        encoding="utf-8",
    )
    (case_dir / "normalized_evidence.json").write_text(
        json.dumps([{"evidence_id": "geo1", "source_type": "geolocation_candidate"}]),
        encoding="utf-8",
    )
    (case_dir / "arguments.json").write_text(
        json.dumps([{"argument_id": "arg1", "evidence_ids": ["geo1"]}]),
        encoding="utf-8",
    )
    (case_dir / "report.md").write_text(
        "## Media Analysis\nmedia\n## Escalation / Human Review\nreview\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["check_case_outputs.py", str(case_dir)])

    assert check_case_outputs.main() == 0
    assert "case outputs look consistent" in capsys.readouterr().out
