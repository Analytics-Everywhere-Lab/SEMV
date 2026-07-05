from __future__ import annotations

from PIL import Image

from src.processing.metadata_extractor import MetadataExtractor
from src.schemas.case_schema import MediaItem


def test_metadata_extractor_returns_evidence_items(tmp_path, monkeypatch):
    image = tmp_path / "img.jpg"
    Image.new("RGB", (32, 24)).save(image)
    monkeypatch.setattr("shutil.which", lambda name: None)
    items = MetadataExtractor().extract(MediaItem(path=str(image), media_type="image"))
    assert items
    assert all(item.provenance for item in items)
