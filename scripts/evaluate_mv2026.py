from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.mv2026_evaluator import evaluate_mv2026


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MV2026 report-style cases.")
    parser.add_argument("--raw-root", default="data/raw/mv2026")
    parser.add_argument("--canonical-root", default="data/canonical/mv2026")
    parser.add_argument("--output-dir", default="data/outputs/evaluation/mv2026_static")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--protocol", default="static")
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    del args.canonical_root, args.config
    result = evaluate_mv2026(args.raw_root, args.output_dir, args.protocol, args.split)
    print(result)


if __name__ == "__main__":
    main()
