from __future__ import annotations

import json

import pytest

from scripts.prepare_cosmos import ConversionError, convert_records, inspect_schema, load_records


def _image(root, relative="val/0.jpg"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    return path


def test_real_shape_nested_articles_are_converted(tmp_path):
    _image(tmp_path)
    rows = [{
        "img_local_path": "val/0.jpg",
        "articles": [{"caption": "First article caption"}, {"caption": "Second article caption"}],
        "maskrcnn_bboxes": [],
    }]
    output = tmp_path / "converted.jsonl"
    summary = convert_records(rows, "val", tmp_path, output)
    converted = load_records(output)[0]
    assert converted["caption1"] == "First article caption"
    assert converted["caption2"] == "Second article caption"
    assert summary["conversion_ratio"] == 1.0


def test_failed_conversion_preserves_existing_output(tmp_path):
    output = tmp_path / "converted.jsonl"
    original = '{"case_id":"existing"}\n'
    output.write_text(original, encoding="utf-8")
    with pytest.raises(ConversionError, match="conversion_ratio"):
        convert_records([{"articles": []}], "val", tmp_path, output, minimum_ratio=1.0)
    assert output.read_text(encoding="utf-8") == original
    assert not output.with_suffix(".jsonl.tmp").exists()


def test_mixed_records_enforce_ratio_and_classify_failures(tmp_path):
    _image(tmp_path)
    rows = [
        {"id": "ok", "img_local_path": "val/0.jpg", "caption": "valid"},
        {"id": "missing", "img_local_path": "val/nope.jpg", "caption": "invalid image"},
    ]
    output = tmp_path / "mixed.jsonl"
    with pytest.raises(ConversionError):
        convert_records(rows, "val", tmp_path, output, minimum_ratio=0.75)
    summary = convert_records(rows, "val", tmp_path, output, minimum_ratio=0.5)
    assert summary["failures_by_reason"] == {"missing_image": 1}
    assert summary["failure_samples"]["missing_image"][0]["keys"]
    assert len(load_records(output)) == 1


def test_duplicate_ids_and_unresolved_images_are_reported(tmp_path):
    _image(tmp_path, "val/0.jpg")
    _image(tmp_path, "val/1.jpg")
    rows = [
        {"id": "dup", "image": "val/0.jpg", "caption": "one"},
        {"id": "dup", "image": "val/1.jpg", "caption": "two"},
        {"id": "missing", "image": "val/not-there.jpg", "caption": "three"},
    ]
    summary = convert_records(rows, "val", tmp_path, tmp_path / "out.jsonl", minimum_ratio=0.3)
    assert summary["failures_by_reason"] == {"duplicate_case_id": 1, "missing_image": 1}


def test_inspect_schema_reports_types_and_keys_without_values(tmp_path):
    source = tmp_path / "annotations.jsonl"
    secret = "private caption must not appear"
    source.write_text(json.dumps({"image": "x.jpg", "articles": [{"caption": secret}]}) + "\n", encoding="utf-8")
    report = inspect_schema(source, load_records(source))
    rendered = json.dumps(report)
    assert report["top_level_type"] == "dict"
    assert "articles[].caption" in rendered
    assert secret not in rendered
