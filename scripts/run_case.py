from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("run_case")

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
    parser.add_argument("--human_review_path", default=None)
    parser.add_argument("--enable_adaptive_revision", default=None)
    parser.add_argument("--save_case_trace", default="true")
    parser.add_argument("--exclude_rejected_arguments", default="true")
    args = parser.parse_args()
    human_review_path = args.human_review_path or args.human_feedback_json

    logger.info("Loading case: %s", args.case_path or args.case_bundle or args.case)
    logger.info("Adapter: %s | split: %s | mode: %s", args.adapter, args.split, args.mode)

    if args.case_bundle:
        path = _resolve(args.case_bundle)
        bundle = load_case_bundle(path)
        logger.info("Loaded case_id=%s with %d media item(s)", bundle.case_id, len(bundle.media_assets))
        logger.info("Starting pipeline...")
        report = run_case_bundle(bundle=bundle, mode=args.mode, config_path=args.config, case_path=path, human_review_path=human_review_path, enable_adaptive_revision=_parse_optional_bool(args.enable_adaptive_revision), save_case_trace=_parse_bool(args.save_case_trace), exclude_rejected_arguments=_parse_bool(args.exclude_rejected_arguments))
        logger.info("Pipeline finished")
    elif args.case_path:
        native = _resolve(args.case_path)
        adapter = default_registry().get_adapter(native, args.adapter)
        bundle = adapter.load(native, split=args.split)
        logger.info("Loaded case_id=%s with %d media asset(s)", bundle.case_id, len(bundle.media_assets))
        write_canonical_bundle(bundle, args.canonical_root)
        logger.info("Starting pipeline...")
        report = run_case_bundle(bundle=bundle, mode=args.mode, config_path=args.config, case_path=native, human_review_path=human_review_path, enable_adaptive_revision=_parse_optional_bool(args.enable_adaptive_revision), save_case_trace=_parse_bool(args.save_case_trace), exclude_rejected_arguments=_parse_bool(args.exclude_rejected_arguments))
        logger.info("Pipeline finished")
    elif args.case:
        path = _resolve(args.case)
        case = MultimediaCase.model_validate(read_json(path))
        logger.info("Loaded case_id=%s with %d media item(s)", case.case_id, len(case.media))
        logger.info("Starting pipeline...")
        report = run_case(
            case=case,
            mode="self_evolving" if args.mode == "self_evolving" else "inference_only",
            ground_truth_label=args.ground_truth_label,
            case_path=path,
            human_review_path=human_review_path,
            enable_adaptive_revision=_parse_optional_bool(args.enable_adaptive_revision),
            exclude_rejected_arguments=_parse_bool(args.exclude_rejected_arguments),
        )
        logger.info("Pipeline finished")
    else:
        parser.error("Provide --case-bundle, --case-path, or --case")
        return

    output_dir = project_root() / "data" / "outputs" / "cases" / report.case_id
    logger.info("Wrote outputs to %s", output_dir)
    print(f"Wrote {output_dir / 'report.json'}")
    print(f"Wrote {output_dir / 'report.md'}")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _parse_optional_bool(value: str | bool | None) -> bool | None:
    if value is None:
        return None
    return _parse_bool(value)


def _resolve(path: str) -> Path:
    target = Path(path)
    return target if target.is_absolute() else project_root() / target


if __name__ == "__main__":
    main()
