from __future__ import annotations

from src.argumentation.argument_generator import ArgumentGenerator
from src.argumentation.argument_verifier import ArgumentVerifier
from src.main import run_case_bundle
from src.schemas.case_bundle_schema import multimedia_case_to_case_bundle
from src.schemas.case_schema import MultimediaCase
from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem
from src.utils.io import project_root, read_json

from tests.conftest import FakeLLMClient
from tests.memory_test_utils import CitingFakeLLM, make_record, make_service


def _sample_bundle():
    case_path = project_root() / "data" / "cases" / "sample_case.json"
    case = MultimediaCase.model_validate(read_json(case_path))
    return multimedia_case_to_case_bundle(case), case_path


def test_used_memory_ids_are_distinguished_from_retrieved(tmp_path):
    service = make_service(tmp_path, retrieval={"min_similarity": 0.0, "top_k": 3})
    service.store.append(
        make_record(memory_id="mem_cited", text="Verify provenance before trusting captions.")
    )
    service.store.append(
        make_record(
            memory_id="mem_uncited",
            text="A completely different unrelated lesson about audio transcripts.",
        )
    )
    bundle, case_path = _sample_bundle()

    report = run_case_bundle(
        bundle=bundle,
        mode="inference_only",
        llm_client=CitingFakeLLM(),
        case_path=case_path,
        memory_service=service,
        save_case_trace=False,
    )

    retrieved_ids = {record.memory_id for record in report.memory_retrieved}
    used_ids = {record.memory_id for record in report.memory_used}
    assert retrieved_ids == {"mem_cited", "mem_uncited"}
    # The fake LLM only ever cites the first offered memory id.
    assert used_ids
    assert used_ids < retrieved_ids
    assert set(report.metadata["used_memory_ids"]) == used_ids
    # Cited ids are validated: nothing outside the retrieved set is recorded.
    assert used_ids <= retrieved_ids


def test_usage_events_record_retrieved_and_cited_stages(tmp_path):
    service = make_service(tmp_path, retrieval={"min_similarity": 0.0, "top_k": 3})
    service.store.append(
        make_record(memory_id="mem_cited", text="Verify provenance before trusting captions.")
    )
    bundle, case_path = _sample_bundle()

    run_case_bundle(
        bundle=bundle,
        mode="inference_only",
        llm_client=CitingFakeLLM(),
        case_path=case_path,
        memory_service=service,
        save_case_trace=False,
    )

    events = service.store.load_usage_events()
    stages = {event.stage for event in events}
    assert "retrieved" in stages
    assert "argument_cited" in stages or "planner_cited" in stages
    cited = [event for event in events if event.stage != "retrieved"]
    assert all(event.memory_id == "mem_cited" for event in cited)


def test_memory_only_arguments_are_rejected_as_ungrounded():
    claim = SubClaim(claim_id="c1", claim_type="what", statement="The event happened as claimed.")
    evidence = [
        EvidenceItem(evidence_id="ev1", content="An independent article describes the event.")
    ]
    memory = [make_record(memory_id="mem_1", text="Some past lesson about events.")]

    class MemoryOnlyLLM(FakeLLMClient):
        def generate_json(self, prompt, system=None, schema=None, **kwargs):
            if "Generate concise support and attack arguments" in prompt:
                return {
                    "arguments": [
                        {
                            "stance": "support",
                            "title": "Memory-only claim",
                            "text": "A stored lesson zqxwv says this claim pattern is usually true.",
                            "evidence_ids": [],
                            "used_memory_ids": ["mem_1"],
                            "rationale": "based on memory only",
                        }
                    ]
                }
            return super().generate_json(prompt, system=system, schema=schema, **kwargs)

    arguments = ArgumentGenerator(MemoryOnlyLLM()).generate(
        claim=claim, evidence=evidence, evidence_graph=EvidenceGraph(), memory_items=memory
    )
    argument = arguments[0]
    if not argument.evidence_ids:
        assert "memory_only_grounding" in argument.uncertainty_flags

    # Strip any repaired evidence link to simulate pure memory grounding, then
    # the argument verifier must reject it.
    ungrounded = argument.model_copy(update={"evidence_ids": []})
    verified = ArgumentVerifier(FakeLLMClient()).verify(claim, ungrounded, evidence)
    assert verified.verifier_valid is False
    assert "argument_without_evidence" in verified.uncertainty_flags
