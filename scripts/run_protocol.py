from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.protocol_runner import run_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MV2026/COSMOS evaluation protocols.")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--output-dir", default="data/outputs/evaluation/joint_mv_cosmos")
    parser.add_argument("--ablation-variant", choices=[f"A{i}" for i in range(11)] + ["all"], default=None)
    args = parser.parse_args()
    result = run_protocol(
        args.config, args.protocol, args.output_dir,
        ablation_variant=args.ablation_variant,
    )
    print(result)


if __name__ == "__main__":
    main()
