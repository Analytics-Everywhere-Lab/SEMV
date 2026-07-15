from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.memory.memory_store import MemoryStore

from tests.memory_test_utils import make_candidate, make_memory_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "consolidate_memory.py"


def _prepare_memory(tmp_path) -> MemoryStore:
    config = make_memory_config(tmp_path)
    store = MemoryStore(config=config)
    from src.memory.memory_consolidator import MemoryConsolidator

    consolidator = MemoryConsolidator(store=store, config=config)
    text = "When reverse search finds an earlier upload, attack the temporal claim."
    consolidator.apply(
        [
            make_candidate(case_id="case1", text=text, verified=True),
            make_candidate(case_id="case2", text=text, verified=True),
        ]
    )
    return store


def _run_cli(memory_dir: Path, *args: str) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            "configs/memory.yaml",
            "--memory-dir",
            str(memory_dir),
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=True,
    )
    return json.loads(completed.stdout)


def test_cli_dry_run_reports_but_does_not_mutate(tmp_path):
    store = _prepare_memory(tmp_path)
    hash_before = store.state_hash(include_short_term=True)

    output = _run_cli(store.memory_dir, "--dry-run")

    assert output["dry_run"] is True
    assert output["stm_candidates_considered"] == 2
    assert output["promoted_records"]
    assert output["errors"] == []
    assert store.state_hash(include_short_term=True) == hash_before
    assert store.load_long_term() == []


def test_cli_apply_is_idempotent(tmp_path):
    store = _prepare_memory(tmp_path)

    first = _run_cli(store.memory_dir, "--apply", "--snapshot")
    assert first["dry_run"] is False
    assert first["promoted_records"]
    assert first["snapshot_path"]
    assert first["state_hash"]
    assert len(store.load_long_term()) == 1

    second = _run_cli(store.memory_dir, "--apply")
    assert second["promoted_records"] == []
    assert second["support_increments"] == {}
    assert second["merged_records"] == []
    assert len(store.load_long_term()) == 1
    assert second["counts_after"]["long_term"] == first["counts_after"]["long_term"]


def test_cli_dry_run_on_missing_directory_creates_nothing(tmp_path):
    missing = tmp_path / "never-created"
    output = _run_cli(missing, "--dry-run")
    assert output["dry_run"] is True
    assert output["consolidation_mode"] == "deterministic-only"
    assert not missing.exists()


def test_cli_snapshot_requires_apply(tmp_path):
    import pytest

    missing = tmp_path / "never-created"
    with pytest.raises(subprocess.CalledProcessError):
        _run_cli(missing, "--snapshot")
    assert not missing.exists()
