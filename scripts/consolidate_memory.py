from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.memory_service import MemoryService


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate verified short-term memory into long-term memory. "
            "Defaults to a dry run; pass --apply to persist changes."
        )
    )
    parser.add_argument("--config", default="configs/memory.yaml", help="Memory config path.")
    parser.add_argument(
        "--memory-dir",
        default=None,
        help="Optional override for the memory directory (defaults to the config paths).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what consolidation would do without mutating any store (default).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply consolidation (required for any mutation).",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="After applying, write a frozen snapshot with manifest and state hash.",
    )
    args = parser.parse_args()

    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive.")
    dry_run = not args.apply

    output: dict = {"dry_run": dry_run, "errors": []}
    try:
        service = MemoryService.from_config_path(args.config, memory_dir=args.memory_dir)
        result = service.consolidate(dry_run=dry_run)
        output.update(
            {
                "counts_before": result.counts_before,
                "counts_after": result.counts_after,
                "stm_candidates_considered": result.stm_considered,
                "promoted_records": result.promoted,
                "merged_records": result.merged,
                "support_increments": result.support_increments,
                "conflicts": result.conflicted,
                "under_review_records": result.under_review,
                "deprecated_records": result.deprecated,
                "expired_archived_stm_records": result.expired,
                "unchanged_records": result.unchanged,
                "changed_long_term_ids": result.changed_long_term_ids,
                "errors": result.errors,
                "snapshot_path": None,
                "state_hash": service.state_hash(),
            }
        )
        if args.snapshot and not dry_run:
            snapshot_path = service.snapshot()
            output["snapshot_path"] = str(snapshot_path)
            output["state_hash"] = service.state_hash()
    except Exception as exc:  # surface errors as structured JSON, not a traceback
        output["errors"].append(str(exc))
        print(json.dumps(output, indent=2, ensure_ascii=False))
        raise SystemExit(1)

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
