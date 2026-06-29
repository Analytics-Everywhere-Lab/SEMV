from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.seed_memory import seed_semantic_rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed semantic memory rules idempotently.")
    parser.add_argument("--memory-dir", default=None)
    args = parser.parse_args()
    inserted = seed_semantic_rules(args.memory_dir)
    print({"inserted": [item.memory_id for item in inserted]})


if __name__ == "__main__":
    main()
