from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.retrieval.local_reverse_image_search import LocalReverseImageSearch
from src.retrieval.visual_index import VisualIndex


def _image(path: Path, diagonal: bool = False) -> Path:
    img = Image.new("RGB", (220, 160), "white")
    draw = ImageDraw.Draw(img)
    if diagonal:
        draw.line((0, 0, 220, 160), fill="black", width=10)
    else:
        draw.rectangle((40, 30, 160, 120), fill="red")
    img.save(path)
    return path


def _config(tmp_path, methods=None):
    return {"media": {"local_reverse_methods": methods or ["phash"], "visual_index_dir": str(tmp_path), "phash_threshold": 10, "clip_similarity_threshold": 0.84}}


def test_phash_finds_duplicate(tmp_path):
    first = _image(tmp_path / "a.jpg")
    duplicate = _image(tmp_path / "b.jpg")
    index = VisualIndex(index_dir=tmp_path / "idx", config=_config(tmp_path / "idx"))
    search = LocalReverseImageSearch(index=index, config=_config(tmp_path / "idx"))
    search.add_assets([first], case_id="old")
    items = search.search([duplicate], case_id="new")
    assert items
    assert items[0].source_type == "reverse_image_local"
    assert items[0].raw_output["phash_distance"] <= 10


def test_phash_does_not_find_unrelated_image(tmp_path):
    first = _image(tmp_path / "a.jpg")
    unrelated = _image(tmp_path / "c.jpg", diagonal=True)
    index = VisualIndex(index_dir=tmp_path / "idx", config=_config(tmp_path / "idx"))
    search = LocalReverseImageSearch(index=index, config=_config(tmp_path / "idx"))
    search.add_assets([first], case_id="old")
    assert search.search([unrelated], case_id="new") == []


def test_clip_faiss_path_is_used_when_index_returns_match(tmp_path):
    query = _image(tmp_path / "q.jpg")

    class FakeIndex:
        def search(self, path, exclude_case_id=None):
            return [{"asset_id": "asset1", "path": "old.jpg", "case_id": "old", "phash_distance": None, "clip_similarity": 0.91, "methods": ["clip_faiss"]}]

    items = LocalReverseImageSearch(index=FakeIndex(), config=_config(tmp_path, ["clip_faiss"])).search([query], case_id="new")
    assert items[0].raw_output["clip_similarity"] == 0.91
    assert items[0].reliability == 0.80


def test_missing_clip_faiss_does_not_crash(tmp_path):
    query = _image(tmp_path / "q.jpg")
    index = VisualIndex(index_dir=tmp_path / "idx", config=_config(tmp_path / "idx", ["clip_faiss"]))
    assert index.search(query) == []


def test_index_persists_and_reloads(tmp_path):
    first = _image(tmp_path / "a.jpg")
    duplicate = _image(tmp_path / "b.jpg")
    cfg = _config(tmp_path / "idx")
    VisualIndex(index_dir=tmp_path / "idx", config=cfg).add_assets([first], case_id="old")
    reloaded = VisualIndex(index_dir=tmp_path / "idx", config=cfg)
    assert reloaded.search(duplicate, exclude_case_id="new")
