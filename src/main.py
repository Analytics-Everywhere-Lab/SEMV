from __future__ import annotations

import argparse
import logging

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from src.aggregation.final_decision_aggregator import FinalDecisionAggregator
from src.contestation.adaptive_revision_executor import execute_adaptive_revision
from src.contestation.contestation_applier import apply_human_contestations, contestation_summary
from src.contestation.revision_router import route_revision
from src.argumentation.argument_generator import ArgumentGenerator
from src.argumentation.uncertainty_escalator import UncertaintyEscalator
from src.argumentation.argument_scorer import ArgumentScorer
from src.argumentation.argument_verifier import ArgumentVerifier
from src.argumentation.clash_resolver import ClashResolver
from src.evidence.evidence_graph import EvidenceGraphBuilder
from src.evidence.evidence_normalizer import EvidenceNormalizer
from src.ingestion.gold_leakage_guard import assert_no_gold_leakage
from src.memory.memory_service import MemoryService
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
from src.schemas.case_trace_schema import CaseTrace
from src.schemas.case_bundle_schema import (
    CaseBundle,
    case_bundle_to_multimedia_case,
    multimedia_case_to_case_bundle,
)
from src.schemas.case_schema import MultimediaCase
from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.contestation_schema import HumanReviewBatch, RevisionTarget
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem
from src.schemas.memory_schema import MemoryRecord
from src.schemas.qbaf_schema import QBAFGraph
from src.schemas.report_schema import SubClaimReport, VerificationReport
from src.utils.env_loader import get_bool_env, get_int_env
from src.utils.io import project_root, read_json, write_json
from src.utils.llm_client import LLMClient, build_llm_client
from src.utils.tool_config import load_tools_config


logger = logging.getLogger("run_case")

RunMode = Literal["inference_only", "self_evolving", "test", "bootstrap_memory"]
LegacyRunMode = Literal["inference_only", "self_evolving"]


def run_case_bundle(
    bundle: CaseBundle,
    mode: RunMode = "inference_only",
    config_path: str = "configs/default.yaml",
    llm_client: LLMClient | None = None,
    case_path: Path | None = None,
    human_review_path: str | Path | None = None,
    human_review_batch: HumanReviewBatch | None = None,
    enable_adaptive_revision: bool | None = None,
    save_case_trace: bool = True,
    exclude_rejected_arguments: bool = True,
    memory_service: MemoryService | None = None,
) -> VerificationReport:
    if mode not in {"inference_only", "self_evolving", "test", "bootstrap_memory"}:
        raise ValueError("mode must be inference_only, self_evolving, test, or bootstrap_memory")

    if (human_review_path is not None or human_review_batch is not None) and not bundle.run_config.allow_human_contestation:
        raise ValueError(
            "Human contestation review was provided, but "
            "bundle.run_config.allow_human_contestation=false."
        )

    assert_no_gold_leakage(bundle, mode)
    shared_llm_client = llm_client or build_llm_client()
    legacy_case = case_bundle_to_multimedia_case(bundle)
    tools_config = load_tools_config(config_path)

    logger.info("[1/8] Raw media processing")
    raw_evidence = RawMediaProcessor(llm_client=shared_llm_client, config=tools_config).process(legacy_case, case_path=case_path)
    raw_evidence.extend(legacy_case.provided_evidence)

    logger.info("[3/8] Claim decomposition")
    claims = _claims_from_bundle(bundle)
    if not claims:
        claims = ClaimDecomposer(llm_client=shared_llm_client).decompose(
            case=legacy_case,
            evidence=raw_evidence,
        )

    shared_memory_service = memory_service or MemoryService(llm_client=shared_llm_client)
    if bundle.run_config.allow_memory_retrieval:
        memory_by_claim = shared_memory_service.retrieve_for_claims(
            bundle=bundle,
            claims=bundle.claims or [],
            evidence=raw_evidence,
            source_clusters=bundle.source_clusters,
            include_short_term=mode in {"bootstrap_memory", "self_evolving"},
        )
        if not memory_by_claim:
            memory_by_claim = {
                claim.claim_id: shared_memory_service.retrieve(
                    case=legacy_case,
                    claim=claim,
                    evidence=raw_evidence,
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
        logger.info("[4/8] Deep research")
        researcher = DeepResearcher(llm_client=shared_llm_client)
        existing_evidence_snapshot = list(all_evidence)
        research_results = _research_claims_parallel(
            researcher=researcher,
            claims=claims,
            research_plans=research_plans,
            existing_evidence=existing_evidence_snapshot,
            allow_reverse_search=bundle.run_config.allow_reverse_search,
        )
        all_evidence = _merge_dedup_evidence(all_evidence, research_results)

    logger.info("[2/8] Evidence normalization")
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

    logger.info("[5/8] Argument generation")
    logger.info("[6/8] A-QBAF reasoning")
    claim_results = _process_claims_parallel(
        claims=claims,
        normalized_evidence=normalized_evidence,
        evidence_graph=evidence_graph,
        memory_by_claim=memory_by_claim,
        bundle=bundle,
        evidence_ranker=evidence_ranker,
        argument_generator=argument_generator,
        argument_verifier=argument_verifier,
        argument_scorer=argument_scorer,
        graph_builder=graph_builder,
        propagator=propagator,
        clash_resolver=clash_resolver,
        decision_mapper=decision_mapper,
        exclude_rejected_arguments=exclude_rejected_arguments,
    )
    for scored_arguments, graph, subclaim_report in claim_results:
        all_arguments.extend(scored_arguments)
        qbaf_graphs.append(graph)
        subclaim_reports.append(subclaim_report)

    final_status, final_confidence = FinalDecisionAggregator().aggregate(
        subclaim_reports,
        bundle=bundle,
    )
    memory_retrieved = _flatten_memory(memory_by_claim.values())
    used_memory_ids = _collect_used_memory_ids(research_plans, all_arguments)
    memory_used = [record for record in memory_retrieved if record.memory_id in used_memory_ids]
    _log_memory_usage(
        service=shared_memory_service,
        bundle=bundle,
        memory_by_claim=memory_by_claim,
        research_plans=research_plans,
        arguments=all_arguments,
    )
    logger.info("[7/8] Uncertainty escalation")
    logger.info("[8/8] Report rendering")
    report = ReportGenerator(llm_client=shared_llm_client).generate(
        case=legacy_case,
        final_status=final_status,
        final_confidence=final_confidence,
        subclaim_reports=subclaim_reports,
        evidence=normalized_evidence,
        evidence_graph=evidence_graph,
        memory_used=memory_used,
    )
    report = _attach_escalation(report, subclaim_reports, all_arguments, normalized_evidence)
    report = report.model_copy(
        update={
            "memory_retrieved": memory_retrieved,
            "metadata": {
                **report.metadata,
                "dataset": bundle.dataset.model_dump(mode="json"),
                "task": bundle.task.model_dump(mode="json"),
                "input": bundle.input.model_dump(mode="json"),
                "gold_visibility": bundle.gold.gold_visibility,
                "used_memory_ids": sorted(used_memory_ids),
            }
        }
    )

    review_batch = _load_human_review_batch(human_review_path, human_review_batch)
    report_before_contestation: VerificationReport | None = None
    contestation_diff: dict[str, Any] | None = None
    revision_plan = None
    if review_batch is not None:
        report_before_contestation = report
        original_arguments = list(all_arguments)
        pre_trace = _build_case_trace(
            bundle=bundle,
            claims=claims,
            evidence_items=all_evidence,
            validated_evidence_items=normalized_evidence,
            arguments=all_arguments,
            qbaf_graphs=qbaf_graphs,
            report=report,
            raw_evidence=raw_evidence,
        )
        revision_plan = route_revision(
            review_batch=review_batch,
            current_arguments=all_arguments,
            case_trace=pre_trace,
        )
        use_adaptive_revision = True if enable_adaptive_revision is None else enable_adaptive_revision
        if not use_adaptive_revision:
            reviewed_arguments = apply_human_contestations(all_arguments, review_batch)
            report, qbaf_graphs, all_arguments = _rerun_qbaf_and_report(
                legacy_case=legacy_case,
                bundle=bundle,
                claims=claims,
                arguments=reviewed_arguments,
                evidence=normalized_evidence,
                evidence_graph=evidence_graph,
                memory_used=memory_used,
                llm_client=shared_llm_client,
                exclude_rejected_arguments=exclude_rejected_arguments,
            )
        else:
            report, qbaf_graphs, all_arguments, normalized_evidence, evidence_graph = execute_adaptive_revision(
                bundle=bundle,
                legacy_case=legacy_case,
                claims=claims,
                raw_evidence=raw_evidence,
                all_evidence=all_evidence,
                normalized_evidence=normalized_evidence,
                evidence_graph=evidence_graph,
                memory_by_claim=memory_by_claim,
                original_arguments=all_arguments,
                review_batch=review_batch,
                revision_plan=revision_plan,
                research_plans=research_plans,
                llm_client=shared_llm_client,
                exclude_rejected_arguments=exclude_rejected_arguments,
                memory_retriever=shared_memory_service.retriever,
            )
        contestation_diff = _contestation_diff(
            original_report=report_before_contestation,
            revised_report=report,
            original_arguments=original_arguments,
            revised_arguments=all_arguments,
            revision_plan=revision_plan,
        )
        summary = contestation_summary(original_arguments, all_arguments, review_batch)
        summary.update(contestation_diff)
        report = report.model_copy(
            update={
                "human_review_applied": True,
                "human_review_batch": review_batch,
                "revision_plan": revision_plan,
                "contestation_summary": summary,
                "metadata": {
                    **report.metadata,
                    "dataset": bundle.dataset.model_dump(mode="json"),
                    "task": bundle.task.model_dump(mode="json"),
                    "input": bundle.input.model_dump(mode="json"),
                    "gold_visibility": bundle.gold.gold_visibility,
                    "enable_adaptive_revision": use_adaptive_revision,
                    "adaptive_revision_executed": use_adaptive_revision,
                    "rerun_from_step": revision_plan.rerun_from_step,
                    "affected_subclaim_ids": revision_plan.affected_subclaim_ids,
                    "affected_evidence_ids": revision_plan.affected_evidence_ids,
                    "affected_argument_ids": revision_plan.affected_argument_ids,
                    "human_contestation_changed_final_decision": contestation_diff[
                        "human_contestation_changed_final_decision"
                    ],
                },
            }
        )

    if mode in {"self_evolving", "bootstrap_memory"} and (
        bundle.gold.gold_final_label or bundle.gold.gold_report_available
    ):
        # Gold is only revealed here, after the report (prediction) exists.
        # Only training/bootstrap runs may stage short-term memory; a frozen
        # memory service (validation/test snapshots) never updates.
        allow_update = (
            bundle.run_config.allow_memory_update and not shared_memory_service.frozen
        )
        reflection, candidates = ReflectionAgent(
            llm_client=shared_llm_client,
            memory_service=shared_memory_service,
        ).reflect(
            report=report,
            ground_truth_label=bundle.gold.gold_final_label,
            human_feedback=(
                {
                    "human_review_batch": review_batch.model_dump(mode="json"),
                    "revision_plan": revision_plan.model_dump(mode="json") if revision_plan else None,
                    "original_report": report_before_contestation.model_dump(mode="json") if report_before_contestation else None,
                    "revised_report": report.model_dump(mode="json"),
                    "contestation_diff": contestation_diff,
                }
                if review_batch is not None
                else None
            ),
            update_memory=allow_update,
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
        save_case_trace=save_case_trace,
        review_batch=review_batch,
        revision_plan=revision_plan,
        report_before_contestation=report_before_contestation,
        contestation_diff=contestation_diff,
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
    human_review_path: str | Path | None = None,
    enable_adaptive_revision: bool | None = None,
    exclude_rejected_arguments: bool = True,
    config_path: str = "configs/default.yaml",
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
        config_path=config_path,
        llm_client=llm_client,
        case_path=case_path,
        human_review_path=human_review_path,
        enable_adaptive_revision=enable_adaptive_revision,
        exclude_rejected_arguments=exclude_rejected_arguments,
    )



def run_from_step(
    bundle: CaseBundle,
    step_name: RevisionTarget,
    previous_state: CaseTrace | None = None,
    human_review_batch: HumanReviewBatch | None = None,
    llm_client: LLMClient | None = None,
    case_path: Path | None = None,
    exclude_rejected_arguments: bool = True,
) -> VerificationReport:
    if human_review_batch is not None and not bundle.run_config.allow_human_contestation:
        raise ValueError(
            "Human contestation review was provided, but "
            "bundle.run_config.allow_human_contestation=false."
        )

    if previous_state is None or not previous_state.arguments:
        return run_case_bundle(
            bundle=bundle,
            mode="inference_only",
            llm_client=llm_client,
            case_path=case_path,
            human_review_batch=human_review_batch,
            enable_adaptive_revision=human_review_batch is not None,
            exclude_rejected_arguments=exclude_rejected_arguments,
        )

    claims = _coerce_subclaims(previous_state.subclaims)
    evidence = _coerce_evidence(
        previous_state.validated_evidence_items or previous_state.evidence_items
    )
    raw_evidence = _coerce_evidence(previous_state.metadata.get("raw_evidence", []) or evidence)
    original_arguments = _coerce_arguments(previous_state.arguments)
    revision_plan = None
    arguments = original_arguments
    evidence_graph = EvidenceGraphBuilder().build(evidence, claims)
    legacy_case = case_bundle_to_multimedia_case(bundle)
    active_llm_client = llm_client or build_llm_client()
    memory_used: list[MemoryRecord] = []
    normalized_evidence = evidence

    if human_review_batch is not None:
        revision_plan = route_revision(
            review_batch=human_review_batch,
            current_arguments=original_arguments,
            case_trace=previous_state,
        )
        memory_by_claim = _memory_by_claim_from_trace(previous_state, claims)
        research_plans = _research_plans_from_trace(previous_state)
        report, _, reviewed_arguments, normalized_evidence, evidence_graph = execute_adaptive_revision(
            bundle=bundle,
            legacy_case=legacy_case,
            claims=claims,
            raw_evidence=raw_evidence,
            all_evidence=evidence,
            normalized_evidence=evidence,
            evidence_graph=evidence_graph,
            memory_by_claim=memory_by_claim,
            original_arguments=original_arguments,
            review_batch=human_review_batch,
            revision_plan=revision_plan,
            research_plans=research_plans,
            llm_client=active_llm_client,
            exclude_rejected_arguments=exclude_rejected_arguments,
        )
    else:
        report, _, reviewed_arguments = _rerun_qbaf_and_report(
            legacy_case=legacy_case,
            bundle=bundle,
            claims=claims,
            arguments=arguments,
            evidence=evidence,
            evidence_graph=evidence_graph,
            memory_used=memory_used,
            llm_client=active_llm_client,
            exclude_rejected_arguments=exclude_rejected_arguments,
        )

    trace = _build_case_trace(
        bundle=bundle,
        claims=claims,
        evidence_items=normalized_evidence,
        validated_evidence_items=normalized_evidence,
        arguments=reviewed_arguments,
        qbaf_graphs=[],
        report=report,
        raw_evidence=raw_evidence,
    )
    trace.metadata["rerun_from_step"] = step_name
    report.metadata["case_trace"] = trace.model_dump(mode="json")
    if human_review_batch is not None:
        diff = _contestation_diff(
            original_report=VerificationReport(
                case_id=bundle.case_id,
                final_status=previous_state.final_decision.get("final_status", "unknown"),
                final_confidence=previous_state.final_decision.get("final_confidence", 0.0),
            ),
            revised_report=report,
            original_arguments=original_arguments,
            revised_arguments=reviewed_arguments,
            revision_plan=revision_plan,
        )
        summary = contestation_summary(original_arguments, reviewed_arguments, human_review_batch)
        summary.update(diff)
        report = report.model_copy(
            update={
                "human_review_applied": True,
                "human_review_batch": human_review_batch,
                "revision_plan": revision_plan,
                "contestation_summary": summary,
                "metadata": {
                    **report.metadata,
                    "case_trace": trace.model_dump(mode="json"),
                    "adaptive_revision_executed": True,
                    "rerun_from_step": revision_plan.rerun_from_step,
                    "affected_subclaim_ids": revision_plan.affected_subclaim_ids,
                    "affected_evidence_ids": revision_plan.affected_evidence_ids,
                    "affected_argument_ids": revision_plan.affected_argument_ids,
                },
            }
        )
    return report


def _memory_by_claim_from_trace(
    previous_state: CaseTrace,
    claims: list[SubClaim],
) -> dict[str, list[MemoryRecord]]:
    stored = previous_state.metadata.get("memory_by_claim")
    if stored:
        return {
            claim_id: [
                item if isinstance(item, MemoryRecord) else MemoryRecord.model_validate(item)
                for item in items
            ]
            for claim_id, items in stored.items()
        }
    return {claim.claim_id: [] for claim in claims}


def _research_plans_from_trace(previous_state: CaseTrace) -> dict[str, ResearchPlan]:
    stored = previous_state.metadata.get("research_plans")
    if not stored:
        return {}
    return {
        claim_id: item if isinstance(item, ResearchPlan) else ResearchPlan.model_validate(item)
        for claim_id, item in stored.items()
    }


def _load_human_review_batch(
    human_review_path: str | Path | None,
    human_review_batch: HumanReviewBatch | None,
) -> HumanReviewBatch | None:
    if human_review_batch is not None:
        return human_review_batch
    if human_review_path is None:
        return None
    return HumanReviewBatch.model_validate(read_json(human_review_path))


def _build_case_trace(
    bundle: CaseBundle,
    claims: list[SubClaim],
    evidence_items: list[EvidenceItem],
    validated_evidence_items: list[EvidenceItem],
    arguments: list[Argument],
    qbaf_graphs: list[QBAFGraph],
    report: VerificationReport,
    raw_evidence: list[EvidenceItem] | None = None,
) -> CaseTrace:
    retrieval_queries = [
        {"claim_id": claim.claim_id, "query": query}
        for claim in claims
        for query in claim.search_queries
    ]
    return CaseTrace(
        case_id=bundle.case_id,
        input_bundle=bundle.model_dump(mode="json"),
        subclaims=claims,
        retrieval_queries=retrieval_queries,
        evidence_items=evidence_items,
        validated_evidence_items=validated_evidence_items,
        arguments=arguments,
        qbaf_state={"graphs": qbaf_graphs},
        final_decision={
            "final_status": report.final_status,
            "final_confidence": report.final_confidence,
        },
        report=report.model_dump(mode="json"),
        metadata={"raw_evidence": raw_evidence or []},
    )


def _rerun_qbaf_and_report(
    legacy_case: MultimediaCase,
    bundle: CaseBundle,
    claims: list[SubClaim],
    arguments: list[Argument],
    evidence: list[EvidenceItem],
    evidence_graph: EvidenceGraph,
    memory_used: list[MemoryRecord],
    llm_client: LLMClient,
    exclude_rejected_arguments: bool,
) -> tuple[VerificationReport, list[QBAFGraph], list[Argument]]:
    graph_builder = QBAFGraphBuilder()
    propagator = QBAFPropagator()
    decision_mapper = DecisionMapper()
    qbaf_graphs: list[QBAFGraph] = []
    subclaim_reports: list[SubClaimReport] = []

    arguments_by_claim: dict[str, list[Argument]] = {}
    for argument in arguments:
        arguments_by_claim.setdefault(argument.claim_id, []).append(argument)

    for claim in claims:
        claim_arguments = arguments_by_claim.get(claim.claim_id, [])
        graph = propagator.propagate(
            graph_builder.build(
                claim=claim,
                arguments=claim_arguments,
                exclude_rejected_arguments=exclude_rejected_arguments,
            )
        )
        qbaf_graphs.append(graph)
        included_arguments = [
            argument
            for argument in claim_arguments
            if argument.argument_id in graph.nodes and argument.argument_id != claim.claim_id
        ]
        subclaim_reports.append(
            decision_mapper.to_subclaim_report(claim, graph, included_arguments)
        )

    final_status, final_confidence = FinalDecisionAggregator().aggregate(
        subclaim_reports,
        bundle=bundle,
    )
    report = ReportGenerator(llm_client=llm_client).generate(
        case=legacy_case,
        final_status=final_status,
        final_confidence=final_confidence,
        subclaim_reports=subclaim_reports,
        evidence=evidence,
        evidence_graph=evidence_graph,
        memory_used=memory_used,
    )
    report = _attach_escalation(report, subclaim_reports, arguments, evidence)
    return report, qbaf_graphs, arguments


def _attach_escalation(
    report: VerificationReport,
    subclaim_reports: list[SubClaimReport],
    arguments: list[Argument],
    evidence: list[EvidenceItem],
) -> VerificationReport:
    claim_scores = {subclaim.claim_id: subclaim.score for subclaim in subclaim_reports}
    decisions = UncertaintyEscalator().evaluate(claim_scores, arguments, evidence)
    escalation = [decision.model_dump(mode="json") for decision in decisions]
    return report.model_copy(
        update={
            "escalation": escalation,
            "metadata": {**report.metadata, "escalation": escalation},
        }
    )


def _contestation_diff(
    original_report: VerificationReport,
    revised_report: VerificationReport,
    original_arguments: list[Argument],
    revised_arguments: list[Argument],
    revision_plan,
) -> dict[str, Any]:
    original_ids = {argument.argument_id for argument in original_arguments}
    revised_ids = {argument.argument_id for argument in revised_arguments}
    return {
        "original_final_status": original_report.final_status,
        "revised_final_status": revised_report.final_status,
        "original_confidence": original_report.final_confidence,
        "revised_confidence": revised_report.final_confidence,
        "human_contestation_changed_final_decision": (
            original_report.final_status != revised_report.final_status
            or original_report.final_confidence != revised_report.final_confidence
        ),
        "changed_arguments": sorted(
            argument.argument_id
            for argument in revised_arguments
            if argument.human_status in {"accepted", "rejected", "edited", "added"}
        ),
        "removed_arguments": sorted(
            argument.argument_id
            for argument in revised_arguments
            if argument.human_status == "rejected"
        ),
        "added_arguments": sorted(revised_ids - original_ids),
        "edited_arguments": sorted(
            argument.argument_id
            for argument in revised_arguments
            if argument.human_status == "edited"
        ),
        "affected_subclaims": revision_plan.affected_subclaim_ids if revision_plan else [],
        "affected_evidence": revision_plan.affected_evidence_ids if revision_plan else [],
        "rerun_from_step": revision_plan.rerun_from_step if revision_plan else None,
    }


def _coerce_subclaims(items: list[Any]) -> list[SubClaim]:
    return [item if isinstance(item, SubClaim) else SubClaim.model_validate(item) for item in items]


def _coerce_evidence(items: list[Any]) -> list[EvidenceItem]:
    return [
        item if isinstance(item, EvidenceItem) else EvidenceItem.model_validate(item)
        for item in items
    ]


def _coerce_arguments(items: list[Any]) -> list[Argument]:
    return [item if isinstance(item, Argument) else Argument.model_validate(item) for item in items]

def _process_claims_parallel(
    claims: list[SubClaim],
    normalized_evidence: list[EvidenceItem],
    evidence_graph: EvidenceGraph,
    memory_by_claim: dict[str, list[MemoryRecord]],
    bundle: CaseBundle,
    evidence_ranker: EvidenceRanker,
    argument_generator: ArgumentGenerator,
    argument_verifier: ArgumentVerifier,
    argument_scorer: ArgumentScorer,
    graph_builder: QBAFGraphBuilder,
    propagator: QBAFPropagator,
    clash_resolver: ClashResolver,
    decision_mapper: DecisionMapper,
    exclude_rejected_arguments: bool = True,
) -> list[tuple[list[Argument], QBAFGraph, SubClaimReport]]:
    max_workers = _max_parallel_claim_workers(len(claims))
    if max_workers <= 1:
        return [
            _process_claim(
                claim=claim,
                normalized_evidence=normalized_evidence,
                evidence_graph=evidence_graph,
                memory_by_claim=memory_by_claim,
                bundle=bundle,
                evidence_ranker=evidence_ranker,
                argument_generator=argument_generator,
                argument_verifier=argument_verifier,
                argument_scorer=argument_scorer,
                graph_builder=graph_builder,
                propagator=propagator,
                clash_resolver=clash_resolver,
                decision_mapper=decision_mapper,
                exclude_rejected_arguments=exclude_rejected_arguments,
            )
            for claim in claims
        ]

    results_by_claim: dict[str, tuple[list[Argument], QBAFGraph, SubClaimReport]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_claim,
                claim=claim,
                normalized_evidence=normalized_evidence,
                evidence_graph=evidence_graph,
                memory_by_claim=memory_by_claim,
                bundle=bundle,
                evidence_ranker=evidence_ranker,
                argument_generator=argument_generator,
                argument_verifier=argument_verifier,
                argument_scorer=argument_scorer,
                graph_builder=graph_builder,
                propagator=propagator,
                clash_resolver=clash_resolver,
                decision_mapper=decision_mapper,
                exclude_rejected_arguments=exclude_rejected_arguments,
            ): claim
            for claim in claims
        }
        for future in as_completed(futures):
            claim = futures[future]
            results_by_claim[claim.claim_id] = future.result()

    return [results_by_claim[claim.claim_id] for claim in claims]


def _process_claim(
    claim: SubClaim,
    normalized_evidence: list[EvidenceItem],
    evidence_graph: EvidenceGraph,
    memory_by_claim: dict[str, list[MemoryRecord]],
    bundle: CaseBundle,
    evidence_ranker: EvidenceRanker,
    argument_generator: ArgumentGenerator,
    argument_verifier: ArgumentVerifier,
    argument_scorer: ArgumentScorer,
    graph_builder: QBAFGraphBuilder,
    propagator: QBAFPropagator,
    clash_resolver: ClashResolver,
    decision_mapper: DecisionMapper,
    exclude_rejected_arguments: bool = True,
) -> tuple[list[Argument], QBAFGraph, SubClaimReport]:
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
    graph = propagator.propagate(graph_builder.build(claim=claim, arguments=scored_arguments, exclude_rejected_arguments=exclude_rejected_arguments))
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
            graph_builder.build(claim=claim, arguments=scored_arguments, exclude_rejected_arguments=exclude_rejected_arguments)
        )

    subclaim_report = decision_mapper.to_subclaim_report(claim, graph, scored_arguments)
    return scored_arguments, graph, subclaim_report


def _max_parallel_claim_workers(claim_count: int) -> int:
    if claim_count <= 1 or not get_bool_env("SEMV_PARALLEL_CLAIMS", True):
        return 1
    max_workers = get_int_env("SEMV_MAX_WORKERS", 2)
    return max(1, min(claim_count, max_workers))


def _research_claims_parallel(
    researcher: DeepResearcher,
    claims: list[SubClaim],
    research_plans: dict[str, ResearchPlan],
    existing_evidence: list[EvidenceItem],
    allow_reverse_search: bool = True,
) -> list[EvidenceItem]:
    max_workers = _max_parallel_research_workers(len(claims))
    if max_workers <= 1:
        results: list[EvidenceItem] = []
        for claim in claims:
            results.extend(
                researcher.research(
                    claim=claim,
                    plan=research_plans[claim.claim_id],
                    existing_evidence=existing_evidence,
                    allow_reverse_search=allow_reverse_search,
                )
            )
        return results

    results_by_claim: dict[str, list[EvidenceItem]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                researcher.research,
                claim=claim,
                plan=research_plans[claim.claim_id],
                existing_evidence=existing_evidence,
                allow_reverse_search=allow_reverse_search,
            ): claim
            for claim in claims
        }
        for future in as_completed(futures):
            claim = futures[future]
            results_by_claim[claim.claim_id] = future.result()

    results = []
    for claim in claims:
        results.extend(results_by_claim.get(claim.claim_id, []))
    return results


def _max_parallel_research_workers(claim_count: int) -> int:
    if claim_count <= 1 or not get_bool_env("SEMV_PARALLEL_DEEP_RESEARCH", True):
        return 1
    max_workers = get_int_env("SEMV_MAX_WORKERS", 2)
    return max(1, min(claim_count, max_workers))


def _merge_dedup_evidence(
    existing_evidence: list[EvidenceItem],
    research_results: list[EvidenceItem],
) -> list[EvidenceItem]:
    deduped: dict[str, EvidenceItem] = {}
    for item in [*existing_evidence, *research_results]:
        if item.evidence_id not in deduped:
            deduped[item.evidence_id] = item
            continue

        existing = deduped[item.evidence_id]
        deduped[item.evidence_id] = existing.model_copy(
            update={
                "reliability": max(existing.reliability, item.reliability),
                "relevance": max(existing.relevance, item.relevance),
                "uncertainty_flags": sorted(
                    set(existing.uncertainty_flags + item.uncertainty_flags)
                ),
            }
        )
    return list(deduped.values())


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
    save_case_trace: bool = True,
    review_batch: HumanReviewBatch | None = None,
    revision_plan=None,
    report_before_contestation: VerificationReport | None = None,
    contestation_diff: dict[str, Any] | None = None,
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
    if save_case_trace:
        trace = _build_case_trace(
            bundle=bundle,
            claims=claims,
            evidence_items=normalized_evidence,
            validated_evidence_items=normalized_evidence,
            arguments=arguments,
            qbaf_graphs=qbaf_graphs,
            report=report,
            raw_evidence=raw_evidence,
        )
        write_json(output_dir / "case_trace.json", trace)
    write_json(output_dir / "report.json", report)
    MarkdownRenderer().render_to_file(report, output_dir / "report.md")
    if review_batch is not None:
        write_json(output_dir / "human_review_batch.json", review_batch)
    if revision_plan is not None:
        write_json(output_dir / "revision_plan.json", revision_plan)
    if report_before_contestation is not None:
        write_json(output_dir / "report_before_contestation.json", report_before_contestation)
        write_json(output_dir / "report_after_contestation.json", report)
    if contestation_diff is not None:
        write_json(output_dir / "contestation_diff.json", contestation_diff)
    write_json(output_dir / "reflection_candidates.json", report.memory_update_candidates)
    write_json(output_dir / "verified_memory_updates.json", report.memory_updates_applied)
    write_json(output_dir / "staged_memory_updates.json", report.memory_updates_staged)
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


def _collect_used_memory_ids(
    research_plans: dict[str, ResearchPlan],
    arguments: list[Argument],
) -> set[str]:
    """Only memory ids explicitly cited by the planner or an argument count as used."""
    used: set[str] = set()
    for plan in research_plans.values():
        used.update(plan.used_memory_ids)
    for argument in arguments:
        used.update(argument.metadata.get("used_memory_ids", []) or [])
    return used


def _log_memory_usage(
    service: MemoryService,
    bundle: CaseBundle,
    memory_by_claim: dict[str, list[MemoryRecord]],
    research_plans: dict[str, ResearchPlan],
    arguments: list[Argument],
) -> None:
    dataset_name = bundle.dataset.dataset_name
    dataset_split = bundle.dataset.dataset_split
    try:
        seen: set[tuple[str, str, str]] = set()
        for claim_id, records in memory_by_claim.items():
            for record in records:
                key = (record.memory_id, "retrieved", claim_id)
                if key in seen:
                    continue
                seen.add(key)
                service.log_usage(
                    case_id=bundle.case_id,
                    memory_id=record.memory_id,
                    stage="retrieved",
                    claim_id=claim_id,
                    dataset_name=dataset_name,
                    dataset_split=dataset_split,
                )
        for claim_id, plan in research_plans.items():
            for memory_id in plan.used_memory_ids:
                service.log_usage(
                    case_id=bundle.case_id,
                    memory_id=memory_id,
                    stage="planner_cited",
                    claim_id=claim_id,
                    dataset_name=dataset_name,
                    dataset_split=dataset_split,
                )
        for argument in arguments:
            for memory_id in argument.metadata.get("used_memory_ids", []) or []:
                service.log_usage(
                    case_id=bundle.case_id,
                    memory_id=memory_id,
                    stage="argument_cited",
                    claim_id=argument.claim_id,
                    argument_id=argument.argument_id,
                    dataset_name=dataset_name,
                    dataset_split=dataset_split,
                )
    except Exception:
        logger.warning("Memory usage logging failed for case %s", bundle.case_id, exc_info=True)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _load_bundle_or_legacy_case(path: Path) -> tuple[CaseBundle, MultimediaCase | None]:
    data = read_json(path)
    try:
        return CaseBundle.model_validate(data), None
    except Exception:
        legacy_case = MultimediaCase.model_validate(data)
        return multimedia_case_to_case_bundle(legacy_case), legacy_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SEMV on one case.")
    parser.add_argument("--case_path", required=True, help="CaseBundle or legacy MultimediaCase JSON path.")
    parser.add_argument("--mode", choices=["inference_only", "self_evolving", "test", "bootstrap_memory"], default="inference_only")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--human_review_path", default=None)
    parser.add_argument("--enable_adaptive_revision", default=None)
    parser.add_argument("--save_case_trace", default="true")
    parser.add_argument("--exclude_rejected_arguments", default="true")
    args = parser.parse_args()

    case_path = Path(args.case_path)
    if not case_path.is_absolute():
        case_path = project_root() / case_path
    bundle, legacy_case = _load_bundle_or_legacy_case(case_path)
    report = run_case_bundle(
        bundle=bundle,
        mode=args.mode,
        config_path=args.config,
        case_path=case_path,
        human_review_path=args.human_review_path,
        enable_adaptive_revision=(
            None if args.enable_adaptive_revision is None else _parse_bool(args.enable_adaptive_revision)
        ),
        save_case_trace=_parse_bool(args.save_case_trace),
        exclude_rejected_arguments=_parse_bool(args.exclude_rejected_arguments),
    )
    del legacy_case
    output_dir = project_root() / "data" / "outputs" / "cases" / report.case_id
    print(f"Wrote {output_dir / 'report.json'}")
    print(f"Wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
