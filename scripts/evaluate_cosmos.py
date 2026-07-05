from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cosmos_evaluator import evaluate_cosmos
from src.utils.llm_client import LoggingLLMClient, build_llm_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate COSMOS image-caption rows.")
    parser.add_argument("--cosmos-metadata", default="data/raw/cosmos/test.jsonl")
    parser.add_argument("--image-root", default="data/raw/cosmos/images")
    parser.add_argument("--output-dir", default="data/outputs/evaluation/cosmos_static")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--mode", default="closed_world")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("evaluation.cosmos")
    logger.info(
        "Starting COSMOS evaluation: metadata=%s image_root=%s output_dir=%s mode=%s split=%s",
        args.cosmos_metadata,
        args.image_root,
        args.output_dir,
        args.mode,
        args.split,
    )
    llm_client = LoggingLLMClient(build_llm_client(), logger_name="llm.output")

    result = evaluate_cosmos(
        args.cosmos_metadata,
        args.image_root,
        args.output_dir,
        args.mode,
        args.split,
        llm_client=llm_client,
    )
    print(result)


if __name__ == "__main__":
    main()
