from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.adapter_registry import default_registry
from src.ingestion.canonical_writer import write_canonical_bundle
from src.main import run_case, run_case_bundle
from src.schemas.case_bundle_schema import load_case_bundle
from src.schemas.case_schema import MultimediaCase
from src.utils.io import project_root, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one multimedia verification case.")
    parser.add_argument("--case", default=None, help="Legacy MultimediaCase JSON path.")
    parser.add_argument("--case-bundle", default=None, help="Canonical CaseBundle JSON path.")
    parser.add_argument("--case-path", default=None, help="Native dataset case path.")
    parser.add_argument("--adapter", default="auto")
    parser.add_argument("--canonical-root", default="data/canonical")
    parser.add_argument("--split", default=None, help="Dataset split name for native case paths, e.g. training.")
    parser.add_argument("--mode", choices=["inference_only", "self_evolving", "test", "bootstrap_memory"], default="inference_only")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ground-truth-label", default=None)
    parser.add_argument("--human-feedback-json", default=None)
    args = parser.parse_args()

    if args.case_bundle:
        path = _resolve(args.case_bundle)
        bundle = load_case_bundle(path)
        report = run_case_bundle(bundle=bundle, mode=args.mode, config_path=args.config, case_path=path)
    elif args.case_path:
        native = _resolve(args.case_path)
        adapter = default_registry().get_adapter(native, args.adapter)
        bundle = adapter.load(native, split=args.split)
        write_canonical_bundle(bundle, args.canonical_root)
        report = run_case_bundle(bundle=bundle, mode=args.mode, config_path=args.config, case_path=native)
    elif args.case:
        path = _resolve(args.case)
        case = MultimediaCase.model_validate(read_json(path))
        report = run_case(
            case=case,
            mode="self_evolving" if args.mode == "self_evolving" else "inference_only",
            ground_truth_label=args.ground_truth_label,
            case_path=path,
        )
    else:
        parser.error("Provide --case-bundle, --case-path, or --case")
        return

    output_dir = project_root() / "data" / "outputs" / "cases" / report.case_id
    print(f"Wrote {output_dir / 'report.json'}")
    print(f"Wrote {output_dir / 'report.md'}")


def _resolve(path: str) -> Path:
    target = Path(path)
    return target if target.is_absolute() else project_root() / target


if __name__ == "__main__":
    main()
