#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

MEDIA_TYPES = {
    "metadata_exiftool",
    "metadata_ffprobe",
    "scene_keyframe",
    "ocr",
    "asr",
    "visual_caption",
    "visual_objects",
    "visual_vqa",
    "frame_analysis",
    "forensic_analysis",
    "reverse_image_local",
    "reverse_image_web_candidate",
    "geolocation_candidate",
}
PLACEHOLDERS = {
    "OCR adapter unavailable": "enable_ocr_adapter",
    "Audio transcription was not run": "enable_asr_adapter",
    "Visual analysis adapter unavailable": "enable_vlm_adapter",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_case_outputs.py data/outputs/cases/<case_id>")
        return 2
    case_dir = Path(sys.argv[1])
    errors = []
    raw_path = case_dir / "raw_evidence.json"
    args_path = case_dir / "arguments.json"
    report_path = case_dir / "report.md"
    for path in (raw_path, args_path, report_path):
        if not path.exists():
            errors.append(f"missing {path}")
    if errors:
        return _finish(errors)

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    normalized_path = case_dir / "normalized_evidence.json"
    normalized = json.loads(normalized_path.read_text(encoding="utf-8")) if normalized_path.exists() else []
    evidence_pool = [item for item in [*raw, *normalized] if isinstance(item, dict)]
    arguments = json.loads(args_path.read_text(encoding="utf-8"))
    markdown = report_path.read_text(encoding="utf-8")
    if "## Media Analysis" not in markdown:
        errors.append("report.md missing ## Media Analysis")
    if "## Escalation / Human Review" not in markdown:
        errors.append("report.md missing escalation section")
    if not any(item.get("source_type") in MEDIA_TYPES for item in evidence_pool):
        errors.append("evidence outputs have no media-derived evidence type")

    evidence_ids = {item.get("evidence_id") for item in evidence_pool}
    for index, argument in enumerate(arguments):
        if not isinstance(argument, dict):
            errors.append(f"arguments.json item {index} is not an object")
            continue
        for evidence_id in argument.get("evidence_ids", []):
            if evidence_id not in evidence_ids:
                errors.append(f"argument {argument.get('argument_id')} references missing evidence {evidence_id}")
    for text, flag in PLACEHOLDERS.items():
        if text in markdown and "disabled" not in markdown.lower() and "unavailable" not in markdown.lower():
            errors.append(f"placeholder text appears without disabled/unavailable context: {text}")
    return _finish(errors)


def _finish(errors: list[str]) -> int:
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("case outputs look consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
