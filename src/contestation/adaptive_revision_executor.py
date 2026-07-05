from __future__ import annotations

import logging

from src.aggregation.final_decision_aggregator import FinalDecisionAggregator
from src.argumentation.argument_generator import ArgumentGenerator
from src.argumentation.argument_scorer import ArgumentScorer
from src.argumentation.argument_verifier import ArgumentVerifier
from src.argumentation.clash_resolver import ClashResolver
from src.contestation.contestation_applier import apply_human_contestations
from src.evidence.evidence_graph import EvidenceGraphBuilder
from src.evidence.evidence_normalizer import EvidenceNormalizer
from src.qbaf.decision_mapper import DecisionMapper
from src.qbaf.graph_builder import QBAFGraphBuilder
from src.qbaf.propagator import QBAFPropagator
from src.reporting.report_generator import ReportGenerator
from src.retrieval.deep_researcher import DeepResearcher
from src.retrieval.evidence_ranker import EvidenceRanker
from src.schemas.argument_schema import Argument
from src.schemas.case_bundle_schema import CaseBundle
from src.schemas.case_schema import MultimediaCase
from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.contestation_schema import ArgumentProvenance, HumanReviewBatch, RevisionPlan
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem
from src.schemas.memory_schema import MemoryRecord
from src.schemas.qbaf_schema import QBAFGraph
from src.schemas.report_schema import SubClaimReport, VerificationReport
from src.utils.llm_client import LLMClient


logger = logging.getLogger("run_case")


def affected_claim_ids_from_plan(
    revision_plan: RevisionPlan,
    arguments: list[Argument],
    all_claim_ids: list[str] | None = None,
) -> set[str]:
    affected: set[str] = set(revision_plan.affected_subclaim_ids)

    arguments_by_id = {argument.argument_id: argument for argument in arguments}
    for argument_id in revision_plan.affected_argument_ids:
        argument = arguments_by_id.get(argument_id)
        if argument:
            affected.add(argument.claim_id)

    if revision_plan.affected_evidence_ids:
        affected_evidence_ids = set(revision_plan.affected_evidence_ids)
        for argument in arguments:
            if affected_evidence_ids.intersection(argument.evidence_ids):
                affected.add(argument.claim_id)

    if not affected and all_claim_ids:
        affected = set(all_claim_ids)

    return affected


def merge_arguments_after_rerun(
    *,
    old_arguments: list[Argument],
    reviewed_arguments: list[Argument],
    regenerated_arguments: list[Argument],
    affected_claim_ids: set[str],
) -> list[Argument]:
    del old_arguments

    merged: list[Argument] = []
    existing_ids: set[str] = set()

    for argument in reviewed_arguments:
        if argument.claim_id not in affected_claim_ids:
            merged.append(argument)
            existing_ids.add(argument.argument_id)
            continue

        # For affected claims, keep human-reviewed arguments (accepted / edited /
        # added / rejected) for traceability. Machine-generated, unreviewed
        # arguments for affected claims are dropped in favor of regenerated ones.
        if argument.human_status != "unreviewed":
            merged.append(argument)
            existing_ids.add(argument.argument_id)

    for index, argument in enumerate(regenerated_arguments, start=1):
        argument_id = argument.argument_id
        if argument_id in existing_ids:
            suffix = 1
            candidate = f"{argument_id}_rerun_{suffix}"
            while candidate in existing_ids:
                suffix += 1
                candidate = f"{argument_id}_rerun_{suffix}"
            argument = argument.model_copy(update={"argument_id": candidate})
            argument_id = candidate
        merged.append(argument)
        existing_ids.add(argument_id)

    return merged


def rerun_argument_stage_for_claims(
    *,
    affected_claims: list[SubClaim],
    normalized_evidence: list[EvidenceItem],
    evidence_graph: EvidenceGraph,
    memory_by_claim: dict[str, list[MemoryRecord]],
    bundle: CaseBundle,
    llm_client: LLMClient,
    exclude_rejected_arguments: bool,
    revision_plan: RevisionPlan | None = None,
) -> list[Argument]:
    from src.main import _process_claim

    evidence_ranker = EvidenceRanker()
    argument_generator = ArgumentGenerator(llm_client=llm_client)
    argument_verifier = ArgumentVerifier(llm_client=llm_client)
    argument_scorer = ArgumentScorer(llm_client=llm_client)
    graph_builder = QBAFGraphBuilder()
    propagator = QBAFPropagator()
    clash_resolver = ClashResolver(llm_client=llm_client)
    decision_mapper = DecisionMapper()

    regenerated: list[Argument] = []
    for claim in affected_claims:
        scored_arguments, _graph, _subclaim_report = _process_claim(
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
        for argument in scored_arguments:
            argument = _attach_argument_provenance(argument, claim, revision_plan)
            regenerated.append(argument)
    return regenerated


def _attach_argument_provenance(
    argument: Argument,
    claim: SubClaim,
    revision_plan: RevisionPlan | None,
) -> Argument:
    if argument.provenance is not None:
        return argument
    provenance = ArgumentProvenance(
        source_step="argument_construction",
        subclaim_id=claim.claim_id,
        evidence_ids=argument.evidence_ids,
        upstream_steps=["evidence_retrieval", "evidence_validation"],
        metadata={
            "adaptive_revision": revision_plan is not None,
            "rerun_from_step": revision_plan.rerun_from_step if revision_plan else None,
        },
    )
    return argument.model_copy(update={"provenance": provenance})


def _tag_evidence_for_retrieval_rerun(
    evidence: list[EvidenceItem],
    claim_id: str,
    revision_plan: RevisionPlan,
) -> list[EvidenceItem]:
    tagged = []
    for item in evidence:
        tagged.append(
            item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "claim_id": claim_id,
                        "retrieval_rerun": True,
                        "revision_plan_rationale": revision_plan.rationale,
                    }
                }
            )
        )
    return tagged


def _rerun_qbaf_and_report_all_claims(
    *,
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
    from src.main import _rerun_qbaf_and_report

    return _rerun_qbaf_and_report(
        legacy_case=legacy_case,
        bundle=bundle,
        claims=claims,
        arguments=arguments,
        evidence=evidence,
        evidence_graph=evidence_graph,
        memory_used=memory_used,
        llm_client=llm_client,
        exclude_rejected_arguments=exclude_rejected_arguments,
    )


def execute_adaptive_revision(
    *,
    bundle: CaseBundle,
    legacy_case: MultimediaCase,
    claims: list[SubClaim],
    raw_evidence: list[EvidenceItem],
    all_evidence: list[EvidenceItem],
    normalized_evidence: list[EvidenceItem],
    evidence_graph: EvidenceGraph,
    memory_by_claim: dict[str, list[MemoryRecord]],
    original_arguments: list[Argument],
    review_batch: HumanReviewBatch,
    revision_plan: RevisionPlan,
    research_plans: dict[str, ResearchPlan],
    llm_client: LLMClient,
    exclude_rejected_arguments: bool = True,
) -> tuple[VerificationReport, list[QBAFGraph], list[Argument], list[EvidenceItem], EvidenceGraph]:
    from src.main import _research_claims_parallel, _merge_dedup_evidence

    memory_used = _flatten_memory(memory_by_claim.values())
    claims_by_id = {claim.claim_id: claim for claim in claims}
    all_claim_ids = [claim.claim_id for claim in claims]

    reviewed_arguments = apply_human_contestations(original_arguments, review_batch)

    step = revision_plan.rerun_from_step

    if step == "qbaf_reasoning":
        logger.info("Human contestation routed to qbaf_reasoning")
        report, qbaf_graphs, final_arguments = _rerun_qbaf_and_report_all_claims(
            legacy_case=legacy_case,
            bundle=bundle,
            claims=claims,
            arguments=reviewed_arguments,
            evidence=normalized_evidence,
            evidence_graph=evidence_graph,
            memory_used=memory_used,
            llm_client=llm_client,
            exclude_rejected_arguments=exclude_rejected_arguments,
        )
        return report, qbaf_graphs, final_arguments, normalized_evidence, evidence_graph

    if step == "argument_construction":
        logger.info("Human contestation routed to argument_construction")
        affected_claim_ids = affected_claim_ids_from_plan(revision_plan, reviewed_arguments, all_claim_ids)
        affected_claims = [claims_by_id[claim_id] for claim_id in affected_claim_ids if claim_id in claims_by_id]

        logger.info("Regenerating arguments for affected subclaims only")
        regenerated = rerun_argument_stage_for_claims(
            affected_claims=affected_claims,
            normalized_evidence=normalized_evidence,
            evidence_graph=evidence_graph,
            memory_by_claim=memory_by_claim,
            bundle=bundle,
            llm_client=llm_client,
            exclude_rejected_arguments=exclude_rejected_arguments,
            revision_plan=revision_plan,
        )
        merged_arguments = merge_arguments_after_rerun(
            old_arguments=original_arguments,
            reviewed_arguments=reviewed_arguments,
            regenerated_arguments=regenerated,
            affected_claim_ids=affected_claim_ids,
        )

        logger.info("Rerunning QBAF and final report")
        report, qbaf_graphs, final_arguments = _rerun_qbaf_and_report_all_claims(
            legacy_case=legacy_case,
            bundle=bundle,
            claims=claims,
            arguments=merged_arguments,
            evidence=normalized_evidence,
            evidence_graph=evidence_graph,
            memory_used=memory_used,
            llm_client=llm_client,
            exclude_rejected_arguments=exclude_rejected_arguments,
        )
        return report, qbaf_graphs, final_arguments, normalized_evidence, evidence_graph

    if step == "evidence_validation":
        logger.info("Human contestation routed to evidence_validation")
        affected_claim_ids = affected_claim_ids_from_plan(revision_plan, reviewed_arguments, all_claim_ids)
        affected_claims = [claims_by_id[claim_id] for claim_id in affected_claim_ids if claim_id in claims_by_id]

        re_normalized_evidence = EvidenceNormalizer().normalize(all_evidence)
        re_evidence_graph = EvidenceGraphBuilder().build(re_normalized_evidence, claims)

        logger.info("Regenerating arguments for affected subclaims only")
        regenerated = rerun_argument_stage_for_claims(
            affected_claims=affected_claims,
            normalized_evidence=re_normalized_evidence,
            evidence_graph=re_evidence_graph,
            memory_by_claim=memory_by_claim,
            bundle=bundle,
            llm_client=llm_client,
            exclude_rejected_arguments=exclude_rejected_arguments,
            revision_plan=revision_plan,
        )
        merged_arguments = merge_arguments_after_rerun(
            old_arguments=original_arguments,
            reviewed_arguments=reviewed_arguments,
            regenerated_arguments=regenerated,
            affected_claim_ids=affected_claim_ids,
        )

        logger.info("Rerunning QBAF and final report")
        report, qbaf_graphs, final_arguments = _rerun_qbaf_and_report_all_claims(
            legacy_case=legacy_case,
            bundle=bundle,
            claims=claims,
            arguments=merged_arguments,
            evidence=re_normalized_evidence,
            evidence_graph=re_evidence_graph,
            memory_used=memory_used,
            llm_client=llm_client,
            exclude_rejected_arguments=exclude_rejected_arguments,
        )
        return report, qbaf_graphs, final_arguments, re_normalized_evidence, re_evidence_graph

    if step == "evidence_retrieval":
        logger.info("Human contestation routed to evidence_retrieval")
        affected_claim_ids = affected_claim_ids_from_plan(revision_plan, reviewed_arguments, all_claim_ids)
        affected_claims = [claims_by_id[claim_id] for claim_id in affected_claim_ids if claim_id in claims_by_id]

        logger.info("Rerunning retrieval for affected subclaims only")
        researcher = DeepResearcher(llm_client=llm_client)
        safe_research_plans = _ensure_research_plans(research_plans, affected_claims, legacy_case, llm_client)
        new_evidence = _research_claims_parallel(
            researcher=researcher,
            claims=affected_claims,
            research_plans=safe_research_plans,
            existing_evidence=all_evidence,
        )
        new_evidence = [
            item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "retrieval_rerun": True,
                        "revision_plan_rationale": revision_plan.rationale,
                    }
                }
            )
            for item in new_evidence
        ]
        merged_evidence = _merge_dedup_evidence(all_evidence, new_evidence)

        re_normalized_evidence = EvidenceNormalizer().normalize(merged_evidence)
        re_evidence_graph = EvidenceGraphBuilder().build(re_normalized_evidence, claims)

        logger.info("Regenerating arguments for affected subclaims only")
        regenerated = rerun_argument_stage_for_claims(
            affected_claims=affected_claims,
            normalized_evidence=re_normalized_evidence,
            evidence_graph=re_evidence_graph,
            memory_by_claim=memory_by_claim,
            bundle=bundle,
            llm_client=llm_client,
            exclude_rejected_arguments=exclude_rejected_arguments,
            revision_plan=revision_plan,
        )
        merged_arguments = merge_arguments_after_rerun(
            old_arguments=original_arguments,
            reviewed_arguments=reviewed_arguments,
            regenerated_arguments=regenerated,
            affected_claim_ids=affected_claim_ids,
        )

        logger.info("Rerunning QBAF and final report")
        report, qbaf_graphs, final_arguments = _rerun_qbaf_and_report_all_claims(
            legacy_case=legacy_case,
            bundle=bundle,
            claims=claims,
            arguments=merged_arguments,
            evidence=re_normalized_evidence,
            evidence_graph=re_evidence_graph,
            memory_used=memory_used,
            llm_client=llm_client,
            exclude_rejected_arguments=exclude_rejected_arguments,
        )
        return report, qbaf_graphs, final_arguments, re_normalized_evidence, re_evidence_graph

    # Fallback: unknown / downstream-only step (final_aggregation, report_generation,
    # claim_decomposition) -- treat like qbaf_reasoning since arguments are unaffected.
    logger.info("Human contestation routed to %s; falling back to qbaf_reasoning rerun", step)
    report, qbaf_graphs, final_arguments = _rerun_qbaf_and_report_all_claims(
        legacy_case=legacy_case,
        bundle=bundle,
        claims=claims,
        arguments=reviewed_arguments,
        evidence=normalized_evidence,
        evidence_graph=evidence_graph,
        memory_used=memory_used,
        llm_client=llm_client,
        exclude_rejected_arguments=exclude_rejected_arguments,
    )
    return report, qbaf_graphs, final_arguments, normalized_evidence, evidence_graph


def _ensure_research_plans(
    research_plans: dict[str, ResearchPlan],
    affected_claims: list[SubClaim],
    legacy_case: MultimediaCase,
    llm_client: LLMClient,
) -> dict[str, ResearchPlan]:
    missing = [claim for claim in affected_claims if claim.claim_id not in research_plans]
    if not missing:
        return research_plans

    from src.planning.research_planner import ResearchPlanner

    rebuilt = ResearchPlanner(llm_client=llm_client).plan(
        case=legacy_case,
        subclaims=missing,
        evidence=[],
        memory_by_claim={claim.claim_id: [] for claim in missing},
    )
    merged = dict(research_plans)
    merged.update(rebuilt)
    return merged


def _flatten_memory(groups) -> list[MemoryRecord]:
    flattened: dict[str, MemoryRecord] = {}
    for group in groups:
        for item in group:
            flattened[item.memory_id] = item
    return list(flattened.values())
