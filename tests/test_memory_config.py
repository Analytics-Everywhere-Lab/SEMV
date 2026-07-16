from __future__ import annotations

from src.memory.memory_config import MemoryConfig, load_memory_config


def test_defaults_without_file():
    config = load_memory_config(config_path=None)
    assert config.verification.min_confidence == 0.60
    assert config.consolidation.every_n_cases == 25
    assert config.consolidation.semantic_rule.min_distinct_cases == 3
    assert config.retrieval.top_k == 5
    assert config.similarity.backend == "hybrid"


def test_loads_repo_memory_yaml():
    config = load_memory_config("configs/memory.yaml")
    assert config.paths.memory_dir == "data/memory"
    assert config.paths.short_term_file == "short_term_memory.jsonl"
    assert config.short_term.ttl_days == 30
    assert config.verification.fail_policy == "under_review"
    assert config.similarity.duplicate_similarity == 0.85
    assert config.consolidation.failure.min_distinct_cases == 2
    assert config.retrieval.include_memory_types == ["episodic", "failure", "semantic_rule"]
    assert config.similarity.optional_embedding_backend is None


def test_run_specific_override_file(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text(
        "retrieval:\n  top_k: 2\nconsolidation:\n  every_n_cases: 3\n",
        encoding="utf-8",
    )
    config = load_memory_config("configs/memory.yaml", override_path=override)
    assert config.retrieval.top_k == 2
    assert config.consolidation.every_n_cases == 3
    # untouched values keep the base file's settings
    assert config.retrieval.min_similarity == 0.05


def test_legacy_flat_config_still_loads(tmp_path):
    legacy = tmp_path / "memory.yaml"
    legacy.write_text(
        "retrieval:\n  top_k: 7\nverification:\n  min_confidence: 0.5\n"
        "consolidation:\n  duplicate_similarity: 0.9\n",
        encoding="utf-8",
    )
    config = load_memory_config(legacy)
    assert config.retrieval.top_k == 7
    assert config.verification.min_confidence == 0.5
    assert config.similarity.duplicate_similarity == 0.9


def test_with_memory_dir_rebases_archive_and_snapshots(tmp_path):
    config = MemoryConfig().with_memory_dir(tmp_path / "run_memory")
    assert config.paths.memory_dir == str(tmp_path / "run_memory")
    assert config.paths.archive_dir == str(tmp_path / "run_memory" / "archive")
    assert config.paths.snapshot_dir == str(tmp_path / "run_memory" / "snapshots")



def test_legacy_global_source_threshold_maps_to_missing_per_type_values(tmp_path):
    legacy = tmp_path / "legacy_memory.yaml"
    legacy.write_text(
        "consolidation:\n"
        "  min_distinct_sources: 4\n"
        "  episodic:\n"
        "    min_distinct_sources: 1\n"
        "verification:\n"
        "  reject_on_conflict: false\n",
        encoding="utf-8",
    )

    config = load_memory_config(legacy)

    assert config.consolidation.episodic.min_distinct_sources == 1
    assert config.consolidation.failure.min_distinct_sources == 4
    assert config.consolidation.semantic_rule.min_distinct_sources == 4
    assert not hasattr(config.consolidation, "min_distinct_sources")
    assert not hasattr(config.verification, "reject_on_conflict")


def test_current_memory_yaml_has_no_disconnected_legacy_keys():
    from src.utils.io import read_yaml

    raw = read_yaml("configs/memory.yaml")
    assert "reject_on_conflict" not in raw["verification"]
    assert "min_distinct_sources" not in {
        key: value
        for key, value in raw["consolidation"].items()
        if key not in {"episodic", "failure", "semantic_rule"}
    }
    assert "contradiction_policy" not in raw["verification"]



def test_legacy_reject_on_conflict_logs_deprecation(tmp_path, caplog):
    legacy = tmp_path / "legacy_reject.yaml"
    legacy.write_text(
        "verification:\n  reject_on_conflict: true\n", encoding="utf-8"
    )

    with caplog.at_level("WARNING", logger="run_case"):
        config = load_memory_config(legacy)

    assert not hasattr(config.verification, "reject_on_conflict")
    assert "reject_on_conflict" in caplog.text
    assert "Deprecated" in caplog.text


def test_legacy_contradiction_policy_logs_deprecation(tmp_path, caplog):
    legacy = tmp_path / "legacy_policy.yaml"
    legacy.write_text(
        "verification:\n  contradiction_policy: verified_evidence\n",
        encoding="utf-8",
    )

    with caplog.at_level("WARNING", logger="run_case"):
        config = load_memory_config(legacy)

    assert not hasattr(config.verification, "contradiction_policy")
    assert "contradiction_policy" in caplog.text
    assert "Deprecated" in caplog.text
