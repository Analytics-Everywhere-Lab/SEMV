from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cosmos_evaluator import evaluate_cosmos


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate COSMOS image-caption rows.")
    parser.add_argument("--cosmos-metadata", default="data/raw/cosmos/test.jsonl")
    parser.add_argument("--image-root", default="data/raw/cosmos/images")
    parser.add_argument("--output-dir", default="data/outputs/evaluation/cosmos_static")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--mode", default="closed_world")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    del args.config
    result = evaluate_cosmos(args.cosmos_metadata, args.image_root, args.output_dir, args.mode, args.split)
    print(result)


if __name__ == "__main__":
    main()
