from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.memory.memory_consolidator import MemoryConsolidator
from src.memory.memory_store import MemoryStore
from src.schemas.memory_schema import ShortTermMemoryRecord

from tests.memory_test_utils import make_candidate, make_memory_config, make_record


def _setup(tmp_path, **overrides):
    config = make_memory_config(tmp_path, **overrides)
    store = MemoryStore(config=config)
    return MemoryConsolidator(store=store, config=config), store


def test_verified_candidates_enter_stm_not_ltm(tmp_path):
    consolidator, store = _setup(tmp_path)
    staged = consolidator.apply([make_candidate(verified=True)])

    assert len(staged) == 1
    assert staged[0].status == "staged"
    assert len(store.load_short_term()) == 1
    assert store.load_long_term() == []


def test_unverified_candidates_are_not_staged(tmp_path):
    consolidator, store = _setup(tmp_path)
    rejected = make_candidate().model_copy(
        update={"verification_status": "rejected", "rejected_reason": "nope"}
    )
    staged = consolidator.apply([rejected])

    assert staged == []
    assert store.load_short_term() == []


def test_equivalent_candidates_from_distinct_cases_merge_and_promote(tmp_path):
    consolidator, store = _setup(tmp_path)
    text = "When reverse search finds an earlier upload, attack the temporal claim."
    consolidator.apply(
        [
            make_candidate(case_id="case1", text=text, verified=True),
            make_candidate(case_id="case2", text=text, verified=True),
        ]
    )
    result = consolidator.consolidate()

    assert len(result.promoted) == 2
    long_term = store.load_long_term()
    assert len(long_term) == 1
    record = long_term[0]
    assert record.support_count == 2
    assert set(record.source_case_ids) == {"case1", "case2"}
    assert record.origin == "consolidated"
    assert record.status == "active"
    # STM records reached a terminal state linked to the LTM id.
    for row in store.load_short_term():
        assert row.status == "promoted"
        assert row.promoted_to_memory_id == record.memory_id


def test_rerunning_same_case_does_not_increase_support(tmp_path):
    consolidator, store = _setup(tmp_path)
    text = "When reverse search finds an earlier upload, attack the temporal claim."
    candidates = [
        make_candidate(case_id="case1", text=text, verified=True),
        make_candidate(case_id="case2", text=text, verified=True),
    ]
    consolidator.apply(candidates)
    consolidator.consolidate()
    support_before = store.load_long_term()[0].support_count

    # Same case rerun: identical candidate is staged and consolidated again.
    consolidator.apply([make_candidate(case_id="case1", text=text, verified=True)])
    result = consolidator.consolidate()

    assert result.support_increments == {}
    assert store.load_long_term()[0].support_count == support_before


def test_duplicate_source_fingerprints_do_not_count_as_independent_support(tmp_path):
    consolidator, store = _setup(tmp_path)
    text = "When reverse search finds an earlier upload, attack the temporal claim."
    # Two near-duplicate dataset rows share one source fingerprint.
    consolidator.apply(
        [
            make_candidate(case_id="case1", text=text, fingerprint="fp_shared", verified=True),
            make_candidate(case_id="case2", text=text, fingerprint="fp_shared", verified=True),
        ]
    )
    consolidator.consolidate()

    # failure memory needs 2 independent sources; one shared fingerprint is not enough.
    assert store.load_long_term() == []


def test_failure_memory_requires_configured_case_count(tmp_path):
    consolidator, store = _setup(tmp_path)
    consolidator.apply([make_candidate(case_id="case1", verified=True)])
    consolidator.consolidate()
    assert store.load_long_term() == []

    consolidator.apply([make_candidate(case_id="case2", verified=True)])
    consolidator.consolidate()
    records = store.load_long_term()
    assert len(records) == 1
    assert records[0].memory_type == "failure"


def test_semantic_rule_requires_configured_case_count(tmp_path):
    consolidator, store = _setup(tmp_path)
    text = "Prefer bounded temporal claims when only publication time is known."
    for index, case_id in enumerate(["case1", "case2"]):
        consolidator.apply(
            [make_candidate(case_id=case_id, text=text, memory_type="semantic_rule", verified=True)]
        )
    consolidator.consolidate()
    assert store.load_long_term() == []

    consolidator.apply(
        [make_candidate(case_id="case3", text=text, memory_type="semantic_rule", verified=True)]
    )
    result = consolidator.consolidate()
    records = store.load_long_term()
    assert len(records) == 1
    assert records[0].memory_type == "semantic_rule"
    assert records[0].support_count == 3
    assert result.promoted


def test_low_confidence_episodic_stays_in_stm(tmp_path):
    consolidator, store = _setup(tmp_path)
    consolidator.apply(
        [make_candidate(case_id="case1", memory_type="episodic", confidence=0.7, verified=True)]
    )
    result = consolidator.consolidate()

    assert store.load_long_term() == []
    assert result.unchanged
    assert store.load_short_term()[0].status == "staged"


def test_high_confidence_grounded_episode_promotes_alone(tmp_path):
    consolidator, store = _setup(tmp_path)
    consolidator.apply(
        [make_candidate(case_id="case1", memory_type="episodic", confidence=0.9, verified=True)]
    )
    consolidator.consolidate()

    records = store.load_long_term()
    assert len(records) == 1
    assert records[0].memory_type == "episodic"


def test_contradiction_increments_conflict_and_can_trigger_under_review(tmp_path):
    consolidator, store = _setup(tmp_path)
    store.append(
        make_record(
            text="Trust reverse image search results when locating an event.",
            confidence=0.8,
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
                verified=True,
            )
        ]
    )
    result = consolidator.consolidate()

    record = store.load_long_term()[0]
    assert record.conflict_count == 1
    assert record.status == "under_review"
    assert result.conflicted
    assert any(event.event_type == "conflict" for event in result.events)
    # The contradicting observation is held for review, not merged.
    stm = store.load_short_term()
    assert stm[0].status == "under_review"


def test_confidence_recalibration_is_deterministic(tmp_path):
    consolidator, store = _setup(tmp_path)
    store.append(
        make_record(
            text="Trust reverse image search results when locating an event.",
            confidence=0.8,
            support_count=2,
            source_case_ids=["case0", "case00"],
            source_fingerprints=["fp_case0", "fp_case00"],
        )
    )
    consolidator.apply(
        [
            make_candidate(
                case_id="case1",
                text="Trust reverse image search results when locating an event.",
                confidence=0.9,
                verified=True,
            )
        ]
    )
    consolidator.consolidate()

    record = store.load_long_term()[0]
    # Prior: alpha = 0.8 * 2 = 1.6, beta = 0.4; support adds c=0.9:
    # alpha=2.5, beta=0.5 -> confidence = 2.5/3.
    assert abs(record.metadata["alpha"] - 2.5) < 1e-9
    assert abs(record.metadata["beta"] - 0.5) < 1e-9
    assert abs(record.confidence - record.metadata["alpha"] / (record.metadata["alpha"] + record.metadata["beta"])) < 1e-9
    assert record.support_count == 3


def test_expired_stm_is_archived_not_silently_deleted(tmp_path):
    consolidator, store = _setup(tmp_path)
    old_time = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    candidate = make_candidate(case_id="case1", verified=True)
    stale = ShortTermMemoryRecord.from_candidate(candidate).model_copy(
        update={"staged_at": old_time}
    )
    store.stage_candidate(stale)

    result = consolidator.consolidate()

    assert result.expired == [stale.stm_id]
    assert store.load_short_term() == []
    archive = store.archive_dir / "short_term_expired.jsonl"
    assert archive.exists()
    assert stale.stm_id in archive.read_text(encoding="utf-8")


def test_consolidation_is_idempotent(tmp_path):
    consolidator, store = _setup(tmp_path)
    text = "When reverse search finds an earlier upload, attack the temporal claim."
    consolidator.apply(
        [
            make_candidate(case_id="case1", text=text, verified=True),
            make_candidate(case_id="case2", text=text, verified=True),
        ]
    )
    consolidator.consolidate()
    hash_after_first = store.state_hash(include_short_term=True)

    second = consolidator.consolidate()

    assert second.support_increments == {}
    assert second.promoted == []
    assert second.merged == []
    assert store.state_hash(include_short_term=True) == hash_after_first
    assert len(store.load_long_term()) == 1


def test_dry_run_causes_no_mutation(tmp_path):
    consolidator, store = _setup(tmp_path)
    consolidator.apply(
        [
            make_candidate(case_id="case1", verified=True),
            make_candidate(case_id="case2", verified=True),
        ]
    )
    hash_before = store.state_hash(include_short_term=True)

    result = consolidator.consolidate(dry_run=True)

    assert result.dry_run is True
    assert result.promoted  # it would promote...
    assert store.state_hash(include_short_term=True) == hash_before  # ...but wrote nothing
    assert store.load_long_term() == []
