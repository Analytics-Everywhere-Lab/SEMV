from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.aggregation.final_decision_aggregator import FinalDecisionAggregator
from src.argumentation.argument_generator import ArgumentGenerator
from src.argumentation.argument_scorer import ArgumentScorer
from src.argumentation.argument_verifier import ArgumentVerifier
from src.argumentation.clash_resolver import ClashResolver
from src.evidence.evidence_graph import EvidenceGraphBuilder
from src.evidence.evidence_normalizer import EvidenceNormalizer
from src.ingestion.gold_leakage_guard import assert_no_gold_leakage
from src.memory.memory_retriever import MemoryRetriever
from src.planning.claim_decomposer import ClaimDecomposer
from src.planning.research_planner import ResearchPlanner
from src.processing.media_loader import RawMediaProcessor
from src.qbaf.decision_mapper import DecisionMapper
from src.qbaf.graph_builder import QBAFGraphBuilder
from src.qbaf.propagator import QBAFPropagator
from src.reflection.reflex_agent import ReflectionAgent
from src.reporting.markdown_renderer import MarkdownRenderer
from src.reporting.report_generator import ReportGenerator
from src.retrieval.deep_researcher import DeepResearcher
from src.retrieval.evidence_ranker import EvidenceRanker
from src.schemas.argument_schema import Argument
from src.schemas.case_bundle_schema import (
    CaseBundle,
    case_bundle_to_multimedia_case,
    multimedia_case_to_case_bundle,
)
from src.schemas.case_schema import MultimediaCase
from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceItem
from src.schemas.memory_schema import MemoryRecord
from src.schemas.report_schema import SubClaimReport, VerificationReport
from src.utils.io import project_root, write_json
from src.utils.llm_client import LLMClient, OllamaLLMClient


RunMode = Literal["inference_only", "self_evolving", "test", "bootstrap_memory"]
LegacyRunMode = Literal["inference_only", "self_evolving"]


def run_case_bundle(
    bundle: CaseBundle,
    mode: RunMode = "inference_only",
    config_path: str = "configs/default.yaml",
    llm_client: LLMClient | None = None,
    case_path: Path | None = None,
) -> VerificationReport:
    del config_path
    if mode not in {"inference_only", "self_evolving", "test", "bootstrap_memory"}:
        raise ValueError("mode must be inference_only, self_evolving, test, or bootstrap_memory")

    assert_no_gold_leakage(bundle, mode)
    shared_llm_client = llm_client or OllamaLLMClient()
    legacy_case = case_bundle_to_multimedia_case(bundle)

    raw_evidence = RawMediaProcessor().process(legacy_case, case_path=case_path)
    raw_evidence.extend(legacy_case.provided_evidence)

    claims = _claims_from_bundle(bundle)
    if not claims:
        claims = ClaimDecomposer(llm_client=shared_llm_client).decompose(
            case=legacy_case,
            evidence=raw_evidence,
        )

    memory_retriever = MemoryRetriever()
    if bundle.run_config.allow_memory_retrieval:
        memory_by_claim = memory_retriever.retrieve_for_claims(
            bundle=bundle,
            claims=bundle.claims or [],
            evidence=raw_evidence,
            source_clusters=bundle.source_clusters,
            top_k=5,
        )
        if not memory_by_claim:
            memory_by_claim = {
                claim.claim_id: memory_retriever.retrieve(
                    case=legacy_case,
                    claim=claim,
                    evidence=raw_evidence,
                    top_k=5,
                )
                for claim in claims
            }
    else:
        memory_by_claim = {claim.claim_id: [] for claim in claims}

    research_plans = ResearchPlanner(llm_client=shared_llm_client).plan(
        case=legacy_case,
        subclaims=claims,
        evidence=raw_evidence,
        memory_by_claim=memory_by_claim,
    )

    all_evidence = list(raw_evidence)
    if bundle.run_config.allow_web_search or bundle.run_config.allow_reverse_search:
        researcher = DeepResearcher(llm_client=shared_llm_client)
        for claim in claims:
            all_evidence.extend(
                researcher.research(
                    claim=claim,
                    plan=research_plans[claim.claim_id],
                    existing_evidence=all_evidence,
                )
            )

    normalized_evidence = EvidenceNormalizer().normalize(all_evidence)
    evidence_graph = EvidenceGraphBuilder().build(normalized_evidence, claims)

    subclaim_reports: list[SubClaimReport] = []
    all_arguments: list[Argument] = []
    qbaf_graphs = []
    evidence_ranker = EvidenceRanker()
    argument_generator = ArgumentGenerator(llm_client=shared_llm_client)
    argument_verifier = ArgumentVerifier(llm_client=shared_llm_client)
    argument_scorer = ArgumentScorer(llm_client=shared_llm_client)
    graph_builder = QBAFGraphBuilder()
    propagator = QBAFPropagator()
    clash_resolver = ClashResolver(llm_client=shared_llm_client)
    decision_mapper = DecisionMapper()

    for claim in claims:
        claim_evidence = evidence_ranker.select_for_claim(
            claim=claim,
            evidence=normalized_evidence,
            evidence_graph=evidence_graph,
            top_k=10,
        )
        arguments = argument_generator.generate(
            claim=claim,
            evidence=claim_evidence,
            evidence_graph=evidence_graph,
            memory_items=memory_by_claim.get(claim.claim_id, []),
        )
        verified_arguments = argument_verifier.verify_all(
            claim=claim,
            arguments=arguments,
            evidence=claim_evidence,
            bundle=bundle,
        )
        scored_arguments = argument_scorer.score_all(
            claim=claim,
            arguments=verified_arguments,
            evidence=claim_evidence,
            evidence_graph=evidence_graph,
            bundle=bundle,
        )
        graph = propagator.propagate(graph_builder.build(claim=claim, arguments=scored_arguments))
        if clash_resolver.should_resolve(graph, scored_arguments):
            scored_arguments = clash_resolver.resolve(
                claim=claim,
                graph=graph,
                arguments=scored_arguments,
            )
            scored_arguments = argument_scorer.score_all(
                claim=claim,
                arguments=scored_arguments,
                evidence=claim_evidence,
                evidence_graph=evidence_graph,
                bundle=bundle,
            )
            graph = propagator.propagate(
                graph_builder.build(claim=claim, arguments=scored_arguments)
            )
        all_arguments.extend(scored_arguments)
        qbaf_graphs.append(graph)
        subclaim_reports.append(
            decision_mapper.to_subclaim_report(claim, graph, scored_arguments)
        )

    final_status, final_confidence = FinalDecisionAggregator().aggregate(
        subclaim_reports,
        bundle=bundle,
    )
    memory_used = _flatten_memory(memory_by_claim.values())
    report = ReportGenerator(llm_client=shared_llm_client).generate(
        case=legacy_case,
        final_status=final_status,
        final_confidence=final_confidence,
        subclaim_reports=subclaim_reports,
        evidence=normalized_evidence,
        evidence_graph=evidence_graph,
        memory_used=memory_used,
    )
    report = report.model_copy(
        update={
            "metadata": {
                **report.metadata,
                "dataset": bundle.dataset.model_dump(mode="json"),
                "task": bundle.task.model_dump(mode="json"),
                "input": bundle.input.model_dump(mode="json"),
                "gold_visibility": bundle.gold.gold_visibility,
            }
        }
    )

    if mode in {"self_evolving", "bootstrap_memory"} and (
        bundle.gold.gold_final_label or bundle.gold.gold_report_available
    ):
        reflection, candidates = ReflectionAgent(llm_client=shared_llm_client).reflect(
            report=report,
            ground_truth_label=bundle.gold.gold_final_label,
            human_feedback=None,
            update_memory=bundle.run_config.allow_memory_update,
        )
        report.reflection_logs = [reflection]
        report.memory_update_candidates = candidates

    _save_case_outputs(
        bundle=bundle,
        raw_evidence=raw_evidence,
        normalized_evidence=normalized_evidence,
        evidence_graph=evidence_graph,
        claims=claims,
        arguments=all_arguments,
        qbaf_graphs=qbaf_graphs,
        memory_by_claim=memory_by_claim,
        report=report,
    )
    return report


def run_case(
    case: MultimediaCase,
    mode: LegacyRunMode,
    ground_truth_label: str | None = None,
    subclaim_labels: dict[str, str] | None = None,
    human_feedback: dict | None = None,
    llm_client: LLMClient | None = None,
    case_path: Path | None = None,
) -> VerificationReport:
    bundle = multimedia_case_to_case_bundle(case)
    if ground_truth_label or subclaim_labels:
        bundle = bundle.model_copy(
            update={
                "gold": bundle.gold.model_copy(
                    update={
                        "gold_final_label": ground_truth_label or bundle.gold.gold_final_label,
                        "gold_subclaim_labels": subclaim_labels or bundle.gold.gold_subclaim_labels,
                    }
                )
            }
        )
    del human_feedback
    return run_case_bundle(
        bundle=bundle,
        mode=mode,
        llm_client=llm_client,
        case_path=case_path,
    )


def _claims_from_bundle(bundle: CaseBundle) -> list[SubClaim]:
    subclaims: list[SubClaim] = []
    for claim in bundle.claims:
        subclaims.append(
            SubClaim(
                claim_id=claim.claim_id,
                claim_type=claim.claim_type,
                statement=claim.statement,
                priority=max(1, int(round(claim.priority))),
                search_queries=[claim.statement, bundle.primary_claim_text()],
                metadata={
                    "scope_type": claim.scope_type,
                    "media_ids": claim.media_ids,
                    "segment_ids": claim.segment_ids,
                    "source_cluster_id": claim.source_cluster_id,
                    "expected_evidence_types": claim.expected_evidence_types,
                },
            )
        )
    return subclaims


def _save_case_outputs(
    bundle: CaseBundle,
    raw_evidence: list[EvidenceItem],
    normalized_evidence: list[EvidenceItem],
    evidence_graph,
    claims: list[SubClaim],
    arguments: list[Argument],
    qbaf_graphs,
    memory_by_claim: dict[str, list[MemoryRecord]],
    report: VerificationReport,
) -> None:
    output_dir = project_root() / "data" / "outputs" / "cases" / bundle.case_id
    write_json(output_dir / "input_case_bundle.json", bundle)
    write_json(output_dir / "raw_evidence.json", raw_evidence)
    write_json(output_dir / "normalized_evidence.json", normalized_evidence)
    write_json(output_dir / "evidence_graph.json", evidence_graph)
    write_json(output_dir / "subclaims.json", claims)
    write_json(output_dir / "arguments.json", arguments)
    write_json(output_dir / "qbaf_graphs.json", qbaf_graphs)
    write_json(output_dir / "retrieved_memory.json", memory_by_claim)
    write_json(output_dir / "report.json", report)
    MarkdownRenderer().render_to_file(report, output_dir / "report.md")
    write_json(output_dir / "reflection_candidates.json", report.memory_update_candidates)
    write_json(output_dir / "verified_memory_updates.json", report.memory_updates_applied)
    (output_dir / "run_log.txt").write_text(
        "gold_read_before_prediction=false\n"
        f"final_status={report.final_status}\n"
        f"final_confidence={report.final_confidence}\n",
        encoding="utf-8",
    )


def _top_arguments(arguments: list[Argument], stance: str, limit: int = 3) -> list[Argument]:
    return sorted(
        [argument for argument in arguments if argument.stance == stance],
        key=lambda argument: argument.score,
        reverse=True,
    )[:limit]


def _uncertainty_reason(arguments: list[Argument], graph_flags: list[str]) -> str | None:
    flags = set(graph_flags)
    for argument in arguments:
        flags.update(argument.uncertainty_flags)
    if not flags:
        return None
    return ", ".join(sorted(flags))


def _flatten_memory(groups) -> list[MemoryRecord]:
    flattened: dict[str, MemoryRecord] = {}
    for group in groups:
        for item in group:
            flattened[item.memory_id] = item
    return list(flattened.values())
