from __future__ import annotations

from src.processing.asr_extractor import ASRExtractor
from src.schemas.case_schema import MediaItem


def test_asr_missing_ffmpeg_is_uncertainty(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr("shutil.which", lambda name: None)
    items = ASRExtractor({"media": {"enable_asr_adapter": True, "asr_engine": "faster_whisper"}}).extract(MediaItem(path=str(video), media_type="video"))
    assert items[0].source_type == "synthetic_uncertainty"
    assert "ffmpeg" in items[0].uncertainty_flags[0]
