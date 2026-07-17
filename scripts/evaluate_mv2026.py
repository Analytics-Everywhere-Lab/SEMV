from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.runtime import load_runtime_config
from src.evaluation.mv2026_evaluator import evaluate_mv2026
from src.utils.llm_client import LoggingLLMClient, build_llm_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MV2026 report-style cases.")
    parser.add_argument("--raw-root", default="data/raw/mv2026")
    parser.add_argument("--output-dir", default="data/outputs/evaluation/mv2026_static")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--protocol", default="static")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--case-id", default=None, help="Optional MV2026 case id, e.g. ID333.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of cases to run.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("evaluation.mv2026")
    logger.info(
        "Starting MV2026 evaluation: raw_root=%s output_dir=%s protocol=%s split=%s case_id=%s limit=%s",
        args.raw_root,
        args.output_dir,
        args.protocol,
        args.split,
        args.case_id,
        args.limit,
    )
    runtime = load_runtime_config(args.config)
    llm_client = LoggingLLMClient(build_llm_client(config_path=args.config), logger_name="llm.output")

    result = evaluate_mv2026(
        args.raw_root,
        args.output_dir,
        args.protocol,
        args.split,
        case_id=args.case_id,
        limit=args.limit,
        llm_client=llm_client,
        runtime_config=runtime,
    )
    print(result)


if __name__ == "__main__":
    main()
