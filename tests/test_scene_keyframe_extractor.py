from __future__ import annotations

from src.processing.scene_keyframe_extractor import SceneKeyframeExtractor
from src.schemas.case_schema import MediaItem


def test_scene_keyframe_missing_ffmpeg_is_uncertainty(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr("shutil.which", lambda name: None)
    items = SceneKeyframeExtractor().extract(MediaItem(path=str(video), media_type="video"), tmp_path / "frames")
    assert items[0].source_type == "synthetic_uncertainty"
