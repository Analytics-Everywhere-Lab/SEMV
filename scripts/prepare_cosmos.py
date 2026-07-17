#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any


LIST_KEYS = ("data", "records", "samples", "annotations", "train", "val", "validation", "test")


class ConversionError(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next((payload[key] for key in LIST_KEYS if isinstance(payload.get(key), list)), None)
        if rows is None and payload and all(isinstance(value, dict) for value in payload.values()):
            rows = [{**value, "id": value.get("id", sample_id)} for sample_id, value in payload.items()]
    else:
        rows = None
    if rows is not None:
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"Expected annotation objects in {path}")
        return list(rows)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number}, found {type(row).__name__}")
            records.append(row)
    if not records:
        raise ValueError(f"No annotation records found in {path}")
    return records


def inspect_schema(path: Path, records: list[dict[str, Any]], sample_limit: int = 100) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").lstrip()
    try:
        payload = json.loads(text)
        top_level_type = type(payload).__name__
    except json.JSONDecodeError:
        top_level_type = "jsonl"
    key_counts: Counter[str] = Counter()
    nested_counts: Counter[str] = Counter()
    value_types: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records[:sample_limit]:
        for key, value in row.items():
            key_counts[key] += 1
            value_types[key][type(value).__name__] += 1
            if isinstance(value, dict):
                nested_counts.update(f"{key}.{nested}" for nested in value)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                nested_counts.update(f"{key}[].{nested}" for nested in value[0])
    return {
        "top_level_type": top_level_type,
        "record_count": len(records),
        "common_keys": key_counts.most_common(30),
        "common_nested_keys": nested_counts.most_common(30),
        "representative_value_types": {
            key: counts.most_common(3) for key, counts in sorted(value_types.items())[:30]
        },
    }


def _path_value(row: dict[str, Any], path: str | None) -> Any:
    if not path:
        return None
    value: Any = row
    for part in path.replace("[", ".").replace("]", "").split("."):
        if not part:
            continue
        if isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = _path_value(row, key)
        if value is not None and value != "":
            return value
    return None


def _nested_captions(row: dict[str, Any]) -> list[str]:
    captions: list[str] = []
    for container_key in ("articles", "annotations", "captions", "contexts"):
        container = row.get(container_key)
        items = container if isinstance(container, list) else [container]
        for item in items:
            if isinstance(item, str):
                value = item
            elif isinstance(item, dict):
                value = first(item, "caption", "text", "article.caption", "annotation.caption")
            else:
                value = None
            if value is not None and str(value) not in captions:
                captions.append(str(value))
    return captions


def normalize_label(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and int(value) in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "ooc", "out-of-context", "out_of_context", "out of context", "false_context", "misleading", "fake"}:
        return 1
    if text in {"0", "not-ooc", "not_ooc", "not out of context", "not_out_of_context", "in-context", "in_context", "verified", "real"}:
        return 0
    raise ConversionError("unsupported_label", f"Unsupported label: {value!r}")


def resolve_image_path(raw_value: Any, split: str, cosmos_root: Path) -> str:
    if raw_value is None:
        raise ConversionError("missing_image_path", "Annotation has no image path")
    root = cosmos_root.resolve()
    path = Path(str(raw_value).strip().replace("\\", "/"))
    candidates: list[Path] = [path] if path.is_absolute() else [cosmos_root / path, cosmos_root / split / path.name]
    parts = list(path.parts)
    if "images" in parts:
        candidates.append(cosmos_root / Path(*parts[parts.index("images") + 1:]))
    for marker in ("train", "val", "test"):
        if marker in parts:
            candidates.append(cosmos_root / Path(*parts[parts.index(marker):]))
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    raise ConversionError("missing_image", f"Could not resolve image path for record: {path.name!r}")


def convert_row(row: dict[str, Any], index: int, split: str, cosmos_root: Path,
                field_map: dict[str, str] | None = None) -> dict[str, Any]:
    mapping = field_map or {}
    raw_image = first(row, mapping.get("image"), "img_local_path", "image_path", "img_path", "image", "img", "file_name", "filename", "path")
    caption_1 = first(row, mapping.get("caption1"), "caption1", "caption_1", "caption", "text_1", "text")
    caption_2 = first(row, mapping.get("caption2"), "caption2", "caption_2", "text_2", "alt_caption")
    nested = _nested_captions(row)
    caption_1 = caption_1 if caption_1 is not None else (nested[0] if nested else None)
    caption_2 = caption_2 if caption_2 is not None else (nested[1] if len(nested) > 1 else None)
    case_id = first(row, mapping.get("case_id"), "case_id", "id", "uid", "image_id", "img_id")
    case_id = str(case_id) if case_id is not None else f"cosmos_{split}_{index:06d}"
    if caption_1 is None:
        raise ConversionError("missing_caption", f"Record {index} has no detectable caption")
    output: dict[str, Any] = {
        "case_id": case_id,
        "image_path": resolve_image_path(raw_image, split, cosmos_root),
        "caption1": str(caption_1),
        "split": split,
    }
    if caption_2 is not None:
        output["caption2"] = str(caption_2)
    label = normalize_label(first(row, mapping.get("label"), "label", "gold", "target", "class"))
    if label is not None:
        output["label"] = label
    return output


def convert_records(records: list[dict[str, Any]], split: str, cosmos_root: Path,
                    output_path: Path, minimum_ratio: float = 1.0,
                    field_map: dict[str, str] | None = None) -> dict[str, Any]:
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum conversion ratio must be between 0 and 1")
    converted: list[dict[str, Any]] = []
    failures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for index, row in enumerate(records):
        sample_keys = sorted(str(key) for key in row)[:8]
        try:
            item = convert_row(row, index, split, cosmos_root, field_map)
            if item["case_id"] in seen_ids:
                raise ConversionError("duplicate_case_id", f"Duplicate case ID: {item['case_id']}")
            seen_ids.add(item["case_id"])
            converted.append(item)
        except ConversionError as exc:
            failures[exc.reason].append({"index": index, "keys": sample_keys, "message": str(exc)})
        except (TypeError, KeyError, IndexError) as exc:
            failures["malformed_record"].append({"index": index, "keys": sample_keys, "message": str(exc)})
    ratio = len(converted) / len(records) if records else 0.0
    summary = {
        "records": len(records), "converted": len(converted),
        "failed": len(records) - len(converted), "conversion_ratio": ratio,
        "failures_by_reason": {reason: len(items) for reason, items in sorted(failures.items())},
        "failure_samples": {reason: items[:5] for reason, items in sorted(failures.items())},
    }
    if ratio < minimum_ratio:
        raise ConversionError("conversion_ratio", json.dumps(summary, ensure_ascii=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for item in converted:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        # Validate the complete temporary output before replacement.
        if len(load_records(temporary)) != len(converted):
            raise ConversionError("output_validation", "Temporary output record count mismatch")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return summary


def _parse_field_maps(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    allowed = {"image", "caption1", "caption2", "case_id", "label"}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --field-map {value!r}; expected FIELD=path")
        field, path = value.split("=", 1)
        if field not in allowed or not path:
            raise ValueError(f"Invalid field mapping {value!r}; fields: {sorted(allowed)}")
        result[field] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert downloaded COSMOS annotations for SEMV.")
    parser.add_argument("--cosmos-root", type=Path, default=Path("data/raw/cosmos"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--minimum-conversion-ratio", type=float, default=1.0)
    parser.add_argument("--field-map", action="append", default=[], metavar="FIELD=path")
    parser.add_argument("--inspect-schema", action="store_true")
    args = parser.parse_args()
    input_path = args.input or args.cosmos_root / "annotations" / f"{args.split}_data.json"
    output_path = args.output or args.cosmos_root / f"{args.split}.jsonl"
    records = load_records(input_path)
    if args.limit is not None:
        records = records[:args.limit]
    if args.inspect_schema:
        print(json.dumps(inspect_schema(input_path, records), indent=2, ensure_ascii=False))
        return
    try:
        summary = convert_records(
            records, args.split, args.cosmos_root, output_path,
            minimum_ratio=args.minimum_conversion_ratio,
            field_map=_parse_field_maps(args.field_map),
        )
    except ConversionError as exc:
        print(f"Conversion failed [{exc.reason}]: {exc}")
        raise SystemExit(1) from exc
    print(json.dumps({"input": str(input_path), "output": str(output_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
