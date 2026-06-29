from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.adapter_registry import default_registry
from src.ingestion.canonical_writer import write_canonical_bundle
from src.utils.io import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a native dataset case to CaseBundle format.")
    parser.add_argument("--case-path", required=True)
    parser.add_argument("--adapter", default="auto")
    parser.add_argument("--split", default=None)
    parser.add_argument("--canonical-root", default="data/canonical")
    parser.add_argument("--copy-media", default="false")
    args = parser.parse_args()
    del args.copy_media
    case_path = Path(args.case_path)
    if not case_path.is_absolute():
        case_path = project_root() / case_path
    adapter = default_registry().get_adapter(case_path, args.adapter)
    bundle = adapter.load(case_path, split=args.split)
    output = write_canonical_bundle(bundle, args.canonical_root)
    print(output)


if __name__ == "__main__":
    main()
