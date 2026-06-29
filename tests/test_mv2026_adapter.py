from __future__ import annotations

import json

from src.ingestion.mv2026_adapter import MV2026Adapter


def test_multiple_media_assets_are_loaded(tmp_path):
    case_dir = tmp_path / "ID407"
    media_dir = case_dir / "input" / "media"
    media_dir.mkdir(parents=True)
    (case_dir / "output").mkdir()
    (case_dir / "input" / "ID407.json").write_text(
        json.dumps({"title": "Claim title", "description": "Desc", "location": "Halifax"}),
        encoding="utf-8",
    )
    (media_dir / "a.jpg").write_bytes(b"not really an image")
    (media_dir / "b.mp4").write_bytes(b"not really a video")
    (case_dir / "output" / "report.md").write_text("# Final Verification Status\nStatus: verified", encoding="utf-8")

    bundle = MV2026Adapter().load(case_dir, split="validation")

    assert len(bundle.media_assets) == 2
    assert any(media.role == "primary_claim_media" for media in bundle.media_assets)
    assert bundle.gold.gold_report_available is True
    assert bundle.gold.read_gold_before_prediction is False
    assert not bundle.provided_evidence
