from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.retrieval.web_image_candidate_extractor import WebImageCandidateExtractor


def test_extracts_og_and_relative_image_urls():
    html = '<meta property="og:image" content="/og.jpg"><img src="img/a.png"><img data-src="https://x.test/b.jpg">'
    urls = WebImageCandidateExtractor().extract_image_urls(html, "https://example.test/page")
    assert "https://example.test/og.jpg" in urls
    assert "https://example.test/img/a.png" in urls
    assert "https://x.test/b.jpg" in urls


def test_skips_tiny_images(tmp_path):
    tiny = tmp_path / "tiny.jpg"
    Image.new("RGB", (20, 20)).save(tiny)
    assert WebImageCandidateExtractor()._is_tiny_or_unreadable(tiny)


def test_compare_candidates_emits_web_candidate(tmp_path):
    query = tmp_path / "query.jpg"
    candidate = tmp_path / "candidate.jpg"
    Image.new("RGB", (220, 160), "white").save(query)
    Image.new("RGB", (220, 160), "white").save(candidate)

    class Matcher:
        def compare_paths(self, query_path, candidate_path):
            return {"phash_distance": 0, "clip_similarity": None, "methods": ["phash"]}

    items = WebImageCandidateExtractor().compare_candidates([query], [candidate], Matcher(), page_url="https://news.test/a", source_title="News")
    assert items[0].source_type == "reverse_image_web_candidate"
    assert items[0].raw_output["phash_distance"] == 0


def test_download_offline_failure_does_not_crash(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("offline")
    monkeypatch.setattr("requests.get", fail)
    assert WebImageCandidateExtractor().download_candidate_images(["https://x.test/a.jpg"], tmp_path, 5) == []
