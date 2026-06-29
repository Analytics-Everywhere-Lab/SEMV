from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run_case
from src.schemas.case_schema import MultimediaCase
from src.utils.io import project_root, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all case JSON files in a directory.")
    parser.add_argument("--cases-dir", default="data/cases")
    parser.add_argument("--mode", choices=["inference_only", "self_evolving"], default="inference_only")
    args = parser.parse_args()
    cases_dir = project_root() / args.cases_dir
    for case_path in sorted(cases_dir.glob("*.json")):
        case = MultimediaCase.model_validate(read_json(case_path))
        report = run_case(case=case, mode=args.mode, case_path=case_path)
        print(f"{case.case_id}: {report.final_status} ({report.final_confidence:.2f})")


if __name__ == "__main__":
    main()
