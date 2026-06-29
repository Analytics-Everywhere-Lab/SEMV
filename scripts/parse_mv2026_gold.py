from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.mv2026_gold_parser import parse_mv2026_gold_report
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse one MV2026 gold report after prediction.")
    parser.add_argument("--gold-report", required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    record = parse_mv2026_gold_report(args.gold_report, case_id=args.case_id)
    if args.output:
        write_json(args.output, record)
    print(record.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
