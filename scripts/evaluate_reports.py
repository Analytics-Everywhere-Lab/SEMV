from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generated verification reports.")
    parser.add_argument("--outputs-dir", default="data/outputs")
    args = parser.parse_args()
    reports = sorted((project_root() / args.outputs_dir).glob("*/report.json"))
    total = len(reports)
    counts: dict[str, int] = {}
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        counts[data["final_status"]] = counts.get(data["final_status"], 0) + 1
    print({"total_reports": total, "status_counts": counts})


if __name__ == "__main__":
    main()
