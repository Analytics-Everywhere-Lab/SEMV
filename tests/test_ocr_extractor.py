from __future__ import annotations

from PIL import Image

from src.processing.ocr_extractor import OCRExtractor


def test_ocr_extractor_with_mock_reader(tmp_path, monkeypatch):
    image = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(image)
    class Reader:
        def readtext(self, path):
            return [([[0,0],[10,0],[10,10],[0,10]], "Halifax", 0.9)]
    extractor = OCRExtractor({"media": {"enable_ocr_adapter": True, "ocr_engine": "easyocr"}})
    monkeypatch.setattr(extractor, "_get_reader", lambda: Reader())
    items = extractor.extract(image_paths=[image], case_id="c")
    assert items[0].source_type == "ocr"
    assert "Halifax" in items[0].content
