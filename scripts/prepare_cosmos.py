#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LIST_KEYS = (
    "data",
    "records",
    "samples",
    "annotations",
    "train",
    "val",
    "validation",
    "test",
)


def load_records(path: Path) -> list[dict[str, Any]]:
    """
    Load COSMOS annotations.

    Supports:
    1. A normal JSON list
    2. A JSON dictionary containing a list
    3. A dictionary keyed by sample ID
    4. JSON Lines/JSONL, including files ending in .json
    """
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    # First try ordinary JSON.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value

        # Dictionary indexed by sample ID.
        if payload and all(isinstance(value, dict) for value in payload.values()):
            rows: list[dict[str, Any]] = []

            for sample_id, value in payload.items():
                row = dict(value)
                row.setdefault("id", sample_id)
                rows.append(row)

            return rows

    # Fall back to JSON Lines, even when the file extension is .json.
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}, "
                    f"but found {type(row).__name__}"
                )

            records.append(row)

    if records:
        return records

    raise ValueError(f"No annotation records found in {path}")


def first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def normalize_label(value: Any) -> int | None:
    if value is None or value == "":
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        integer = int(value)
        if integer in (0, 1):
            return integer

    text = str(value).strip().lower()

    positive = {
        "1",
        "ooc",
        "out-of-context",
        "out_of_context",
        "out of context",
        "false_context",
        "misleading",
        "fake",
    }

    negative = {
        "0",
        "not-ooc",
        "not_ooc",
        "not out of context",
        "not_out_of_context",
        "in-context",
        "in_context",
        "verified",
        "real",
    }

    if text in positive:
        return 1

    if text in negative:
        return 0

    raise ValueError(f"Unsupported label: {value!r}")


def resolve_image_path(
    raw_value: Any,
    split: str,
    cosmos_root: Path,
) -> str:
    if raw_value is None:
        raise ValueError("Annotation has no image path")

    raw = str(raw_value).strip().replace("\\", "/")
    path = Path(raw)

    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        # Already includes train/, val/, or test/.
        candidates.append(cosmos_root / path)

        # Annotation contains only the image filename.
        candidates.append(cosmos_root / split / path.name)

        # Annotation may include images/train/... from another layout.
        parts = list(path.parts)
        if "images" in parts:
            index = parts.index("images")
            suffix = Path(*parts[index + 1 :])
            candidates.append(cosmos_root / suffix)

        # Annotation may contain only train/123.jpg after a larger prefix.
        for marker in ("train", "val", "test"):
            if marker in parts:
                index = parts.index(marker)
                suffix = Path(*parts[index:])
                candidates.append(cosmos_root / suffix)

    existing = next((candidate for candidate in candidates if candidate.exists()), None)

    if existing is None:
        attempted = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            f"Could not resolve image {raw_value!r}. Tried:\n{attempted}"
        )

    try:
        return existing.resolve().relative_to(cosmos_root.resolve()).as_posix()
    except ValueError:
        return str(existing.resolve())


def convert_row(
    row: dict[str, Any],
    index: int,
    split: str,
    cosmos_root: Path,
) -> dict[str, Any]:
    raw_image = first(
        row,
        "img_local_path",
        "image_path",
        "img_path",
        "image",
        "img",
        "file_name",
        "filename",
        "path",
    )

    caption_1 = first(
        row,
        "caption1",
        "caption_1",
        "caption",
        "text_1",
        "text",
    )

    caption_2 = first(
        row,
        "caption2",
        "caption_2",
        "text_2",
        "alt_caption",
    )

    case_id = first(
        row,
        "case_id",
        "id",
        "uid",
        "image_id",
        "img_id",
    )

    if case_id is None:
        case_id = f"cosmos_{split}_{index:06d}"

    if caption_1 is None:
        raise ValueError(f"Record {index} has no first caption")

    output: dict[str, Any] = {
        "case_id": str(case_id),
        "image_path": resolve_image_path(
            raw_image,
            split=split,
            cosmos_root=cosmos_root,
        ),
        "caption1": str(caption_1),
        "split": split,
    }

    if caption_2 is not None:
        output["caption2"] = str(caption_2)

    label = normalize_label(
        first(row, "label", "gold", "target", "class")
    )
    if label is not None:
        output["label"] = label

    # Retain the original annotation for traceability.
    output["original_annotation"] = row

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert downloaded COSMOS annotations for SEMV."
    )
    parser.add_argument(
        "--cosmos-root",
        type=Path,
        default=Path("data/raw/cosmos"),
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        required=True,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    input_path = args.input or (
        args.cosmos_root / "annotations" / f"{args.split}_data.json"
    )
    output_path = args.output or (
        args.cosmos_root / f"{args.split}.jsonl"
    )

    records = load_records(input_path)

    if args.limit is not None:
        records = records[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    failed: list[tuple[int, str]] = []

    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(records):
            try:
                converted_row = convert_row(
                    row,
                    index=index,
                    split=args.split,
                    cosmos_root=args.cosmos_root,
                )
            except Exception as exc:
                failed.append((index, str(exc)))
                continue

            handle.write(
                json.dumps(converted_row, ensure_ascii=False) + "\n"
            )
            converted += 1

    print(f"Input:     {input_path}")
    print(f"Output:    {output_path}")
    print(f"Records:   {len(records)}")
    print(f"Converted: {converted}")
    print(f"Failed:    {len(failed)}")

    if failed:
        print("\nFirst conversion failures:")
        for index, message in failed[:20]:
            print(f"\nRecord {index}:")
            print(message)

        raise SystemExit(1)


if __name__ == "__main__":
    main()
