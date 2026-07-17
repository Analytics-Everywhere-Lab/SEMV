from __future__ import annotations

from src.config.runtime import PipelineRuntimeConfig, RuntimeFeatures
from src.main import _process_claim
from src.qbaf.decision_mapper import DecisionMapper
from src.qbaf.graph_builder import QBAFGraphBuilder
from src.schemas.argument_schema import Argument
from src.schemas.case_bundle_schema import CaseBundle, DatasetInfo, InputMetadata, TaskInfo
from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem


class Ranker:
    def __init__(self): self.top_k = None
    def select_for_claim(self, **kwargs):
        self.top_k = kwargs["top_k"]
        return kwargs["evidence"][: self.top_k]


class Generator:
    def generate(self, claim, **kwargs):
        return [Argument(argument_id="a1", claim_id=claim.claim_id, stance="support",
                         text="support", evidence_ids=["e1"], intrinsic_score=0.8)]


class Verifier:
    def __init__(self): self.calls = 0
    def verify_all(self, **kwargs):
        self.calls += 1
        return kwargs["arguments"]


class Scorer:
    def score_all(self, **kwargs):
        return [row.model_copy(update={"score": 0.8}) for row in kwargs["arguments"]]


class Propagator:
    def __init__(self): self.calls = 0
    def propagate(self, graph):
        self.calls += 1
        return graph.model_copy(update={"claim_score": 0.8})


class Clash:
    def __init__(self):
        self.checked = 0
        self.resolved = 0
    def should_resolve(self, *args):
        self.checked += 1
        return True
    def resolve(self, **kwargs):
        self.resolved += 1
        return kwargs["arguments"]


def _bundle():
    return CaseBundle(case_id="case", dataset=DatasetInfo(dataset_name="unit"),
                      task=TaskInfo(task_type="multimedia_verification", media_type="image"), input=InputMetadata())


def _run(features):
    ranker, verifier, propagator, clash = Ranker(), Verifier(), Propagator(), Clash()
    result = _process_claim(
        claim=SubClaim(claim_id="c1", claim_type="what", statement="claim"),
        normalized_evidence=[EvidenceItem(evidence_id="e1", content="evidence")],
        evidence_graph=EvidenceGraph(), memory_by_claim={}, bundle=_bundle(),
        evidence_ranker=ranker, argument_generator=Generator(), argument_verifier=verifier,
        argument_scorer=Scorer(), graph_builder=QBAFGraphBuilder(), propagator=propagator,
        clash_resolver=clash, decision_mapper=DecisionMapper(),
        runtime_config=PipelineRuntimeConfig(evidence_top_k=1, features=features),
    )
    return result, ranker, verifier, propagator, clash


def test_qbaf_verifier_and_clash_can_be_disabled_on_actual_claim_path():
    result, ranker, verifier, propagator, clash = _run(RuntimeFeatures(
        use_memory=False, memory_types=(), use_qbaf=False,
        argument_verifier=False, clash_resolution=False, adaptive_revision=False,
    ))
    assert ranker.top_k == 1
    assert verifier.calls == 0
    assert propagator.calls == 0
    assert clash.checked == 0 and clash.resolved == 0
    assert result[1].metadata["aggregation"] == "non_qbaf_laplace_weighted_mean"


def test_qbaf_verifier_and_clash_execute_when_enabled():
    _, _, verifier, propagator, clash = _run(RuntimeFeatures())
    assert verifier.calls == 1
    assert propagator.calls == 2
    assert clash.checked == 1 and clash.resolved == 1



def test_pipeline_injects_memory_top_k_and_exact_types_and_disables_cleanly(tmp_path):
    from src.main import run_case_bundle
    from src.schemas.case_bundle_schema import multimedia_case_to_case_bundle
    from src.schemas.case_schema import MultimediaCase
    from tests.conftest import FakeLLMClient

    class SpyMemory:
        frozen = True

        def __init__(self):
            self.calls = []
            self.usage_calls = 0

        def retrieve_for_claims(self, **kwargs):
            self.calls.append((kwargs["top_k"], tuple(kwargs["memory_types"])))
            return {}

        def retrieve(self, **kwargs):
            self.calls.append((kwargs["top_k"], tuple(kwargs["memory_types"])))
            return []

        def log_usage(self, **kwargs):
            self.usage_calls += 1

    bundle = multimedia_case_to_case_bundle(MultimediaCase(case_id="config_case", claim="claim"))
    enabled = SpyMemory()
    run_case_bundle(
        bundle, mode="test", llm_client=FakeLLMClient(), memory_service=enabled,
        runtime_config=PipelineRuntimeConfig(
            memory_top_k=2,
            features=RuntimeFeatures(memory_types=("failure",)),
        ),
        artifact_root=tmp_path / "enabled", save_case_trace=False,
    )
    assert enabled.calls and all(call == (2, ("failure",)) for call in enabled.calls)

    disabled = SpyMemory()
    run_case_bundle(
        bundle, mode="test", llm_client=FakeLLMClient(), memory_service=disabled,
        runtime_config=PipelineRuntimeConfig(features=RuntimeFeatures(use_memory=False, memory_types=())),
        artifact_root=tmp_path / "disabled", save_case_trace=False,
    )
    assert disabled.calls == []
    assert disabled.usage_calls == 0



def test_configured_output_path_changes_standalone_artifact_location(tmp_path):
    from src.main import run_case_bundle
    from src.schemas.case_bundle_schema import multimedia_case_to_case_bundle
    from src.schemas.case_schema import MultimediaCase
    from tests.conftest import FakeLLMClient

    config = tmp_path / "runtime.yaml"
    configured_output = tmp_path / "configured_outputs"
    config.write_text(
        f"pipeline:\n  evidence_top_k: 3\n  memory_top_k: 2\npaths:\n  outputs_dir: {configured_output}\n",
        encoding="utf-8",
    )
    bundle = multimedia_case_to_case_bundle(MultimediaCase(case_id="path_case", claim="claim"))
    bundle = bundle.model_copy(update={
        "run_config": bundle.run_config.model_copy(update={"allow_memory_retrieval": False})
    })
    run_case_bundle(
        bundle, mode="test", llm_client=FakeLLMClient(), config_path=str(config),
        save_case_trace=False,
    )
    assert (configured_output / "cases" / "path_case" / "report.json").exists()
