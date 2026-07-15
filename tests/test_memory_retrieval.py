from __future__ import annotations

from src.memory.memory_retriever import MemoryRetriever
from src.memory.memory_store import MemoryStore
from src.schemas.case_schema import MultimediaCase
from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceItem

from tests.memory_test_utils import make_memory_config, make_record


def _retriever(tmp_path, records, **overrides):
    config = make_memory_config(tmp_path, **overrides)
    store = MemoryStore(config=config)
    for record in records:
        store.append(record)
    return MemoryRetriever(store=store, config=config)


def _case_and_claim():
    case = MultimediaCase(
        case_id="c1",
        claim="A video shows an explosion at the port filmed from a rooftop.",
    )
    claim = SubClaim(
        claim_id="c1_where",
        claim_type="where",
        statement="The explosion video was filmed at the claimed port location.",
    )
    return case, claim


def test_only_active_ltm_is_retrieved(tmp_path):
    text = "Separate camera location from target event location in explosion videos."
    retriever = _retriever(
        tmp_path,
        [
            make_record(memory_id="mem_active", text=text, status="active"),
            make_record(memory_id="mem_deprecated", text=text + " v2", status="deprecated"),
            make_record(memory_id="mem_review", text=text + " v3", status="under_review"),
        ],
        retrieval={"min_similarity": 0.0},
    )
    case, claim = _case_and_claim()

    results = retriever.retrieve(case, claim, evidence=[])

    assert [record.memory_id for record in results] == ["mem_active"]


def test_evidence_affects_retrieval_ranking(tmp_path):
    shared_text = "Check the port explosion location claim against independent sources."
    retriever = _retriever(
        tmp_path,
        [
            make_record(
                memory_id="mem_reverse",
                text=shared_text,
                evidence_pattern="reverse search earlier appearance geolocation mismatch",
            ),
            make_record(
                memory_id="mem_forensic",
                text=shared_text,
                evidence_pattern="forensic manipulation artifacts splicing",
            ),
        ],
        retrieval={"min_similarity": 0.0},
    )
    case, claim = _case_and_claim()
    evidence = [
        EvidenceItem(
            evidence_id="ev1",
            source_type="reverse_image_local",
            content="Reverse search found an earlier appearance with a geolocation mismatch.",
            uncertainty_flags=["earlier_appearance"],
        )
    ]

    results = retriever.retrieve(case, claim, evidence=evidence)

    assert results[0].memory_id == "mem_reverse"
    # And the ranking is evidence-driven: without evidence the tie collapses.
    no_evidence = retriever.retrieve(case, claim, evidence=[])
    scores = {r.memory_id: r.metadata["retrieval_score"] for r in results}
    baseline = {r.memory_id: r.metadata["retrieval_score"] for r in no_evidence}
    assert scores["mem_reverse"] > baseline["mem_reverse"]


def test_min_similarity_and_top_k_are_honored(tmp_path):
    records = [
        make_record(
            memory_id=f"mem_{index}",
            text="Separate camera location from target event location in explosion videos.",
            canonical_key=f"key_{index}",
            metadata={},
        )
        for index in range(4)
    ]
    # Distinct texts so dedup keeps them apart.
    records = [
        record.model_copy(update={"text": record.text + f" variant {index} {'x' * index}"})
        for index, record in enumerate(records)
    ]
    retriever = _retriever(tmp_path, records, retrieval={"min_similarity": 0.0, "top_k": 2})
    case, claim = _case_and_claim()
    assert len(retriever.retrieve(case, claim, evidence=[])) == 2

    strict = _retriever(
        tmp_path.joinpath("strict"),
        records,
        retrieval={"min_similarity": 0.99, "top_k": 5},
    )
    assert strict.retrieve(case, claim, evidence=[]) == []


def test_scores_are_normalized_and_recorded(tmp_path):
    retriever = _retriever(
        tmp_path,
        [
            make_record(
                memory_id="mem_1",
                text="Separate camera location from target event location in explosion videos.",
            )
        ],
        retrieval={"min_similarity": 0.0},
    )
    case, claim = _case_and_claim()

    results = retriever.retrieve(case, claim, evidence=[])

    assert len(results) == 1
    score = results[0].metadata["retrieval_score"]
    assert 0.0 <= score <= 1.0


def test_equivalent_retrieved_records_are_deduplicated(tmp_path):
    text = "Separate camera location from target event location in explosion videos."
    retriever = _retriever(
        tmp_path,
        [
            make_record(memory_id="mem_a", text=text),
            make_record(memory_id="mem_b", text=text),
        ],
        retrieval={"min_similarity": 0.0},
    )
    case, claim = _case_and_claim()

    results = retriever.retrieve(case, claim, evidence=[])

    assert len(results) == 1


def test_configured_memory_types_filter(tmp_path):
    retriever = _retriever(
        tmp_path,
        [
            make_record(memory_id="mem_fail", memory_type="failure", text="port explosion lesson"),
            make_record(
                memory_id="mem_rule", memory_type="semantic_rule", text="port explosion rule"
            ),
        ],
        retrieval={"min_similarity": 0.0, "include_memory_types": ["semantic_rule"]},
    )
    case, claim = _case_and_claim()

    results = retriever.retrieve(case, claim, evidence=[])

    assert [record.memory_id for record in results] == ["mem_rule"]
