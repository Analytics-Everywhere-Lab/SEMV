from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.memory_store import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Print append-only memory store counts.")
    parser.parse_args()
    store = MemoryStore()
    counts: dict[str, int] = {}
    for record in store.load_all():
        counts[record.memory_type] = counts.get(record.memory_type, 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
