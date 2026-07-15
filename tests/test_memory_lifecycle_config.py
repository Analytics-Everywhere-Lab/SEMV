from __future__ import annotations

from src.memory.memory_consolidator import MemoryConsolidator
from src.memory.memory_store import MemoryStore
from src.memory.memory_verifier import MemoryVerifier
from src.schemas.case_bundle_schema import (
    CaseBundle,
    Claim,
    DatasetInfo,
    InputMetadata,
    TaskInfo,
)

from tests.memory_test_utils import (
    MemoryFakeLLM,
    make_candidate,
    make_memory_config,
    make_record,
    make_service,
)


def test_short_term_max_records_overflow_is_archived(tmp_path):
    config = make_memory_config(tmp_path, short_term={"max_records": 2})
    store = MemoryStore(config=config)
    consolidator = MemoryConsolidator(store=store, config=config)

    consolidator.apply(
        [
            make_candidate(case_id=f"case{i}", text=f"Lesson number {i} about provenance {'x' * i}.", verified=True)
            for i in range(3)
        ]
    )

    assert len(store.load_short_term()) == 2
    overflow = store.archive_dir / "short_term_overflow.jsonl"
    assert overflow.exists()


def test_every_n_cases_triggers_scheduled_consolidation(tmp_path):
    service = make_service(tmp_path, consolidation={"every_n_cases": 2})
    text = "When reverse search finds an earlier upload, attack the temporal claim."
    service.stage_candidates(
        [
            make_candidate(case_id="case1", text=text, verified=True),
            make_candidate(case_id="case2", text=text, verified=True),
        ]
    )

    assert service.register_case_processed() is None
    result = service.register_case_processed()

    assert result is not None
    assert result.promoted
    assert len(service.store.load_long_term()) == 1


def test_reject_on_conflict_false_still_routes_verified_evidence(tmp_path):
    config = make_memory_config(tmp_path, verification={"reject_on_conflict": False})
    store = MemoryStore(config=config)
    store.append(
        make_record(
            text="Trust reverse image search results when locating an event.",
            confidence=0.95,
            support_count=3,
            source_case_ids=["a", "b", "c"],
            source_fingerprints=["fa", "fb", "fc"],
        )
    )
    verifier = MemoryVerifier(MemoryFakeLLM(), store=store, config=config)

    result = verifier.verify(
        make_candidate(
            case_id="case9",
            text="Do not trust reverse image search results when locating an event.",
        )
    )

    assert result.verified is True
    assert result.verification_status == "verified"
    assert result.semantic_relation == "contradicts"
    assert result.related_memory_id == "mem_existing"


def test_repeated_contradictions_deprecate_but_archive(tmp_path):
    config = make_memory_config(tmp_path)
    store = MemoryStore(config=config)
    consolidator = MemoryConsolidator(store=store, config=config)
    store.append(
        make_record(
            text="Trust reverse image search results when locating an event.",
            confidence=0.5,
            support_count=1,
            source_case_ids=["case0"],
            source_fingerprints=["fp_case0"],
        )
    )
    consolidator.apply(
        [
            make_candidate(
                case_id="case1",
                text="Do not trust reverse image search results when locating an event.",
                confidence=0.9,
                verified=True,
            )
        ]
    )

    result = consolidator.consolidate()

    record = store.load_long_term()[0]
    assert record.status == "deprecated"
    assert result.deprecated == [record.memory_id]
    # Archived, not silently deleted: still in the LTM file plus an archive copy.
    archive = store.archive_dir / "deprecated_long_term.jsonl"
    assert archive.exists()


def test_conflict_ratio_recovery_reactivates_under_review_memory(tmp_path):
    config = make_memory_config(tmp_path)
    store = MemoryStore(config=config)
    consolidator = MemoryConsolidator(store=store, config=config)
    store.append(
        make_record(
            text="Trust reverse image search results when locating an event.",
            status="under_review",
            confidence=0.6,
            support_count=2,
            conflict_count=1,
            source_case_ids=["case0", "case00"],
            source_fingerprints=["fp_case0", "fp_case00"],
        )
    )
    consolidator.apply(
        [
            make_candidate(
                case_id=f"case{i}",
                text="Trust reverse image search results when locating an event.",
                confidence=0.9,
                verified=True,
            )
            for i in range(1, 4)
        ]
    )

    consolidator.consolidate()

    record = store.load_long_term()[0]
    assert record.support_count == 5
    ratio = record.conflict_count / (record.support_count + record.conflict_count)
    assert ratio <= config.consolidation.max_conflict_ratio
    assert record.status == "active"


def test_generalization_failure_creates_one_stable_review_proposal(tmp_path):
    config = make_memory_config(tmp_path)
    store = MemoryStore(config=config)
    consolidator = MemoryConsolidator(
        store=store, config=config, llm_client=MemoryFakeLLM()
    )
    text = "Reverse search revealed the footage was older than claimed."
    consolidator.apply(
        [
            make_candidate(case_id=f"case{i}", text=text, confidence=0.65, verified=True)
            for i in range(1, 4)
        ]
    )

    result = consolidator.consolidate()

    assert any(event.event_type == "generalization_failed" for event in result.events)
    proposals = store.load_long_term()
    assert len(proposals) == 1
    assert proposals[0].status == "under_review"
    assert all(row.status == "under_review" for row in store.load_short_term())
    assert all(row.promoted_to_memory_id == proposals[0].memory_id for row in store.load_short_term())
    state_hash = store.state_hash()
    second = consolidator.consolidate()
    assert second.events == []
    assert store.state_hash() == state_hash


def test_generalization_synthesizes_verified_semantic_rule(tmp_path):
    class GeneralizingLLM(MemoryFakeLLM):
        def generate_json(self, prompt, **kwargs):
            if "Synthesize ONE generalized rule" in prompt:
                return {
                    "trigger_pattern": "reverse search finds an earlier upload of the media",
                    "lesson": "Older reverse-search matches indicate recycled footage.",
                    "recommended_action": "Attack the temporal claim with the earlier upload.",
                    "confidence": 0.8,
                }
            return super().generate_json(prompt, **kwargs)

    config = make_memory_config(tmp_path)
    store = MemoryStore(config=config)
    consolidator = MemoryConsolidator(store=store, config=config, llm_client=GeneralizingLLM())
    text = "Reverse search revealed the footage was older than claimed."
    consolidator.apply(
        [
            make_candidate(case_id=f"case{i}", text=text, confidence=0.65, verified=True)
            for i in range(1, 4)
        ]
    )

    result = consolidator.consolidate()

    rules = [r for r in store.load_long_term() if r.memory_type == "semantic_rule"]
    assert len(rules) == 1
    assert rules[0].status == "active"
    assert rules[0].origin == "consolidated"
    assert "generalized" in rules[0].tags
    assert rules[0].support_count == 3
    assert any(event.event_type == "generalized" for event in result.events)


def test_retrieve_during_bootstrap_config_controls_stm_visibility(tmp_path):
    bundle = CaseBundle(
        case_id="c1",
        dataset=DatasetInfo(dataset_name="mv2026"),
        task=TaskInfo(task_type="multimedia_verification", media_type="image"),
        input=InputMetadata(title="Explosion at the port filmed from a rooftop"),
    )
    claim = Claim(
        claim_id="c1_where",
        claim_type="where",
        statement="The explosion happened at the claimed port.",
    )
    candidate = make_candidate(
        case_id="case1",
        text="Explosion port footage lessons about rooftop camera angles.",
        verified=True,
    )

    hidden = make_service(tmp_path / "off", retrieval={"min_similarity": 0.0})
    hidden.stage_candidates([candidate])
    results = hidden.retrieve_for_claims(bundle, [claim], [], include_short_term=True)
    assert results["c1_where"] == []  # default: STM is not retrievable

    visible = make_service(
        tmp_path / "on",
        retrieval={"min_similarity": 0.0},
        short_term={"retrieve_during_bootstrap": True},
    )
    visible.stage_candidates([candidate])
    results = visible.retrieve_for_claims(bundle, [claim], [], include_short_term=True)
    assert [record.memory_id for record in results["c1_where"]] == [f"stm_{candidate.candidate_id}"]
    # And even then, only bootstrap-phase retrieval sees it.
    results = visible.retrieve_for_claims(bundle, [claim], [], include_short_term=False)
    assert results["c1_where"] == []
