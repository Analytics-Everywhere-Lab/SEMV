from __future__ import annotations

import json

from src.memory.memory_store import MemoryStore
from src.memory.seed_memory import SEED_SEMANTIC_RULES, seed_semantic_rules
from src.schemas.memory_schema import ShortTermMemoryRecord

from tests.memory_test_utils import make_candidate, make_memory_config, make_record


LEGACY_ROW = {
    "memory_id": "rule_legacy_001",
    "memory_type": "semantic_rule",
    "case_id": None,
    "claim_type": "when",
    "task_type": "multimedia_verification",
    "text": "A publication timestamp establishes a latest-known occurrence bound.",
    "trigger_pattern": "only publication time is known",
    "lesson": "A publication timestamp establishes a latest-known occurrence bound.",
    "recommended_action": "Map When to partially_verified.",
    "failure_type": None,
    "source_case_ids": [],
    "tags": [],
    "confidence": 0.93,
    "support_count": 3,
    "conflict_count": 0,
    "usage_count": 0,
    "last_used_at": None,
    "created_at": "2026-07-03T18:40:04.475001Z",
    "updated_at": None,
    "verified_by": None,
    "status": "active",
    "metadata": {},
}


def test_legacy_jsonl_rows_load_with_safe_defaults(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "semantic_rules.jsonl").write_text(
        json.dumps(LEGACY_ROW) + "\n", encoding="utf-8"
    )
    store = MemoryStore(memory_dir)

    records = store.load_all()

    assert len(records) == 1
    record = records[0]
    assert record.memory_id == "rule_legacy_001"
    assert record.memory_level == "long_term"
    assert record.origin == "legacy"
    assert record.version == 1
    assert record.source_fingerprints == []
    assert record.support_weight == 0.0
    assert record.status == "active"


def test_repo_seed_rules_still_load(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    inserted = seed_semantic_rules(store=store)
    assert len(inserted) == len(SEED_SEMANTIC_RULES)
    loaded = store.load_all()
    assert {record.memory_id for record in loaded} == {
        row["memory_id"] for row in SEED_SEMANTIC_RULES
    }
    assert all(record.origin == "seed" for record in loaded)


def test_seed_rules_are_never_duplicated(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    first = seed_semantic_rules(store=store)
    second = seed_semantic_rules(store=store)

    assert len(first) == len(SEED_SEMANTIC_RULES)
    assert second == []
    assert len(store.load_all()) == len(SEED_SEMANTIC_RULES)


def test_upsert_long_term_preserves_ids_and_backs_up(tmp_path):
    config = make_memory_config(tmp_path)
    store = MemoryStore(config=config)
    record = make_record(memory_id="mem_1", support_count=1)
    store.append(record)

    updated = record.model_copy(update={"support_count": 5, "version": 2})
    store.upsert_long_term([updated])

    records = store.load_long_term()
    assert len(records) == 1
    assert records[0].memory_id == "mem_1"
    assert records[0].support_count == 5
    backups = list((store.archive_dir / "backups").glob("*failure_memory.jsonl"))
    assert backups, "rewriting an existing store must create a timestamped backup"


def test_stage_candidate_upserts_by_id(tmp_path):
    store = MemoryStore(config=make_memory_config(tmp_path))
    candidate = make_candidate(verified=True)
    record = ShortTermMemoryRecord.from_candidate(candidate)

    store.stage_candidate(record)
    store.stage_candidate(record)

    assert len(store.load_short_term()) == 1


def test_snapshot_writes_manifest_with_state_hash(tmp_path):
    store = MemoryStore(config=make_memory_config(tmp_path))
    store.append(make_record())
    snapshot_dir = store.snapshot("frozen")

    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state_hash"] == store.state_hash()
    assert "semantic_rules.jsonl" in manifest["files"] or "failure_memory.jsonl" in manifest["files"]
    snapshot_store = MemoryStore(snapshot_dir)
    assert snapshot_store.state_hash() == manifest["state_hash"]


def test_state_hash_changes_with_content(tmp_path):
    store = MemoryStore(config=make_memory_config(tmp_path))
    before = store.state_hash()
    store.append(make_record())
    assert store.state_hash() != before
