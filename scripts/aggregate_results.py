from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import project_root, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation result directories.")
    parser.add_argument("--evaluation-root", default="data/outputs/evaluation")
    parser.add_argument("--output", default="data/outputs/evaluation/aggregate_results.json")
    args = parser.parse_args()
    root = _resolve(args.evaluation_root)
    rows = []
    for path in sorted(root.glob("**/aggregate_metrics.json")):
        rows.append({"path": str(path), "metrics": json.loads(path.read_text(encoding="utf-8"))})
    write_json(_resolve(args.output), {"runs": rows})
    print({"runs": len(rows), "output": str(_resolve(args.output))})


def _resolve(path: str) -> Path:
    target = Path(path)
    return target if target.is_absolute() else project_root() / target


if __name__ == "__main__":
    main()
