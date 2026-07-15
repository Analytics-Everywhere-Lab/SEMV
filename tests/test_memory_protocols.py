from __future__ import annotations

import pytest

from src.main import run_case_bundle
from src.memory.memory_service import MemoryFrozenError
from src.memory.seed_memory import SEED_SEMANTIC_RULES, seed_semantic_rules
from src.schemas.case_bundle_schema import multimedia_case_to_case_bundle
from src.schemas.case_schema import MultimediaCase
from src.utils.io import project_root, read_json

from tests.memory_test_utils import CitingFakeLLM, make_candidate, make_service


STRUCTURED_LESSONS = {
    "episodic": {
        "observation": "Cached reverse-search evidence resolved the location sub-claim.",
        "what_worked_or_failed": "Provenance checks kept weak captions from dominating.",
        "confidence": 0.9,
        "evidence_ids": [],
        "argument_ids": [],
    },
    "failures": [
        {
            "failure_type": "weak_or_missing_provenance",
            "trigger_pattern": "caption evidence without provenance",
            "lesson": "Weak provenance should keep support strength low.",
            "recommended_action": "Preserve uncertainty when provenance is missing.",
            "confidence": 0.8,
            "evidence_ids": [],
            "argument_ids": [],
        }
    ],
    "semantic": None,
}


def _bootstrap_bundle(gold_label="false_context", allow_update=True):
    case_path = project_root() / "data" / "cases" / "sample_case.json"
    case = MultimediaCase.model_validate(read_json(case_path))
    bundle = multimedia_case_to_case_bundle(case)
    return (
        bundle.model_copy(
            update={
                "gold": bundle.gold.model_copy(update={"gold_final_label": gold_label}),
                "run_config": bundle.run_config.model_copy(
                    update={"allow_memory_update": allow_update}
                ),
            }
        ),
        case_path,
    )


def test_bootstrap_updates_short_term_memory_but_not_ltm_directly(tmp_path):
    service = make_service(tmp_path, retrieval={"min_similarity": 0.0})
    bundle, case_path = _bootstrap_bundle()

    report = run_case_bundle(
        bundle=bundle,
        mode="bootstrap_memory",
        llm_client=CitingFakeLLM(lessons=STRUCTURED_LESSONS),
        case_path=case_path,
        memory_service=service,
        save_case_trace=False,
    )

    stm = service.store.load_short_term()
    assert stm, "training/bootstrap must stage short-term memory"
    assert report.memory_updates_staged
    assert {row.memory_type for row in stm} == {"episodic"}
    assert all(
        candidate.failure_type != "weak_or_missing_provenance"
        for candidate in report.memory_update_candidates
    )
    assert all(row.verification_status == "verified" for row in stm)
    # No direct LTM append: promotion only happens through consolidation.
    assert service.store.load_long_term() == []
    assert report.memory_updates_applied == []


def test_test_mode_never_updates_memory(tmp_path):
    service = make_service(tmp_path, retrieval={"min_similarity": 0.0})
    seed_semantic_rules(store=service.store)
    hash_before = service.state_hash(include_short_term=True)
    bundle, case_path = _bootstrap_bundle(allow_update=True)

    run_case_bundle(
        bundle=bundle,
        mode="test",
        llm_client=CitingFakeLLM(lessons=STRUCTURED_LESSONS),
        case_path=case_path,
        memory_service=service,
        save_case_trace=False,
    )

    assert service.store.load_short_term() == []
    assert service.state_hash(include_short_term=True) == hash_before


def test_frozen_service_refuses_mutation(tmp_path):
    service = make_service(tmp_path, frozen=True)
    with pytest.raises(MemoryFrozenError):
        service.stage_candidates([make_candidate(verified=True)])
    with pytest.raises(MemoryFrozenError):
        service.register_case_processed()
    with pytest.raises(MemoryFrozenError):
        service.consolidate()


def test_frozen_snapshot_hash_unchanged_after_validation_run(tmp_path):
    train_service = make_service(tmp_path, retrieval={"min_similarity": 0.0})
    seed_semantic_rules(store=train_service.store)
    text = "When reverse search finds an earlier upload, attack the temporal claim."
    train_service.stage_candidates(
        [
            make_candidate(case_id="case1", text=text, verified=True),
            make_candidate(case_id="case2", text=text, verified=True),
        ]
    )
    train_service.consolidate()
    snapshot_dir = train_service.snapshot("frozen")

    frozen = make_service(
        tmp_path / "unused",
        frozen=True,
        usage_log_path=tmp_path / "eval" / "usage.jsonl",
        retrieval={"min_similarity": 0.0},
    )
    # Point the frozen service at the snapshot directory.
    frozen = type(frozen)(
        config=frozen.config.with_memory_dir(snapshot_dir),
        frozen=True,
        usage_log_path=tmp_path / "eval" / "usage.jsonl",
    )
    hash_before = frozen.state_hash(include_short_term=True)
    assert frozen.store.load_long_term(statuses=["active"])

    bundle, case_path = _bootstrap_bundle(allow_update=True)
    run_case_bundle(
        bundle=bundle,
        mode="test",
        llm_client=CitingFakeLLM(lessons=STRUCTURED_LESSONS),
        case_path=case_path,
        memory_service=frozen,
        save_case_trace=False,
    )

    assert frozen.state_hash(include_short_term=True) == hash_before
    # Usage events went to the evaluation output dir, not the snapshot.
    assert not (snapshot_dir / "memory_usage_events.jsonl").exists() or (
        (snapshot_dir / "memory_usage_events.jsonl").stat().st_size == 0
    )


def test_gold_is_never_read_before_prediction(tmp_path):
    service = make_service(tmp_path, retrieval={"min_similarity": 0.0})
    sentinel = "sentinel_gold_label_zzz"
    bundle, case_path = _bootstrap_bundle(gold_label=sentinel)
    llm = CitingFakeLLM(lessons=STRUCTURED_LESSONS)

    run_case_bundle(
        bundle=bundle,
        mode="bootstrap_memory",
        llm_client=llm,
        case_path=case_path,
        memory_service=service,
        save_case_trace=False,
    )

    reflection_markers = ("You are the reflection module", "Verify whether this memory lesson")
    for call in llm.calls:
        prompt = " ".join(str(part) for part in call)
        if sentinel in prompt:
            assert any(marker in prompt for marker in reflection_markers), (
                "gold label leaked into a pre-prediction prompt"
            )


def test_validation_split_candidates_never_enter_memory(tmp_path):
    service = make_service(tmp_path)
    verifier = service.verifier(CitingFakeLLM())
    candidate = verifier.verify(make_candidate(split="validation"))
    assert candidate.verified is False
    staged = service.consolidator.apply([candidate])
    assert staged == []
    assert service.store.load_short_term() == []


def test_protocol_runner_freeze_test_bootstraps_and_freezes(tmp_path, monkeypatch):
    import src.evaluation.protocol_runner as protocol_runner

    text = "When reverse search finds an earlier upload, attack the temporal claim."

    def fake_evaluate_mv2026(**kwargs):
        service = kwargs["memory_service"]
        if kwargs["update_memory"]:
            service.stage_candidates(
                [
                    make_candidate(case_id="train1", text=text, verified=True),
                    make_candidate(case_id="train2", text=text, verified=True),
                ]
            )
            service.register_case_processed()
            service.register_case_processed()
        else:
            assert service.frozen
            assert service.store.load_long_term(statuses=["active"])
        return {"dataset": "mv2026", "update_memory": kwargs["update_memory"]}

    def fake_evaluate_cosmos(**kwargs):
        assert kwargs["update_memory"] is False
        assert kwargs["memory_service"].frozen
        return {"dataset": "cosmos"}

    monkeypatch.setattr(protocol_runner, "evaluate_mv2026", fake_evaluate_mv2026)
    monkeypatch.setattr(protocol_runner, "evaluate_cosmos", fake_evaluate_cosmos)

    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        """
evaluation:
  memory_config: "configs/memory.yaml"
  datasets:
    mv2026:
      enabled: true
      raw_root: "data/raw/mv2026"
    cosmos:
      enabled: true
      metadata: "data/raw/cosmos/test.jsonl"
  protocol:
    name: "train_memory_freeze_test"
    consolidate_every_n_cases: 2
    train:
      dataset: "mv2026"
      split: "train"
    eval:
      - dataset: "mv2026"
        split: "validation"
      - dataset: "cosmos"
        split: "test"
""",
        encoding="utf-8",
    )

    results = protocol_runner.run_protocol(
        config_path=config_path,
        output_dir=tmp_path / "out",
    )

    assert results["memory_state_unchanged"] is True
    assert results["state_hash"] == results["state_hash_after_eval"]
    snapshot_dir = tmp_path / "out" / "memory" / "snapshots" / "frozen"
    assert snapshot_dir.exists()
    assert (snapshot_dir / "manifest.json").exists()
    # Training seeded + consolidated memory into the run-specific directory.
    from src.memory.memory_store import MemoryStore

    run_store = MemoryStore(tmp_path / "out" / "memory")
    ids = {record.memory_id for record in run_store.load_long_term()}
    assert {row["memory_id"] for row in SEED_SEMANTIC_RULES} <= ids
    assert any(record.origin == "consolidated" for record in run_store.load_long_term())
