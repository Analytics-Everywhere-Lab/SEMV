from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.schemas.argument_schema import Argument
from src.schemas.evidence_schema import EvidenceItem


class EscalationConfig(BaseModel):
    neutral_low: float = 0.45
    neutral_high: float = 0.55
    low_reliability_threshold: float = 0.35
    support_attack_tie_margin: float = 0.10


class EscalationDecision(BaseModel):
    claim_id: str
    should_escalate: bool
    reason_codes: list[str] = Field(default_factory=list)
    defer_to_human: bool = False
    stronger_verifier_recommended: bool = False
    affected_pipeline_stages: list[str] = Field(default_factory=list)
    notes: str = ""


ArgumentCard = Argument


class UncertaintyEscalator:
    def evaluate(
        self,
        claim_scores: dict[str, float],
        arguments: list[ArgumentCard],
        evidence: list[EvidenceItem],
        config: EscalationConfig | None = None,
    ) -> list[EscalationDecision]:
        cfg = config or EscalationConfig()
        decisions = []
        evidence_by_id = {item.evidence_id: item for item in evidence}
        arguments_by_claim: dict[str, list[Argument]] = {}
        for argument in arguments:
            arguments_by_claim.setdefault(argument.claim_id, []).append(argument)
        claim_ids = sorted(set(claim_scores) | set(arguments_by_claim))
        for claim_id in claim_ids:
            claim_arguments = arguments_by_claim.get(claim_id, [])
            claim_evidence = _linked_evidence(claim_arguments, evidence_by_id)
            reason_codes: set[str] = set()
            stages: set[str] = set()
            score = claim_scores.get(claim_id)
            if score is not None and cfg.neutral_low <= score <= cfg.neutral_high:
                reason_codes.add("neutral_score_band")
                stages.add("qbaf_reasoning")
            if claim_evidence and all(item.source_type == "synthetic_uncertainty" for item in claim_evidence):
                reason_codes.add("all_synthetic_uncertainty")
                stages.update({"raw_media_processing", "deep_research"})
            if not claim_evidence:
                reason_codes.add("missing_relevant_evidence")
                stages.update({"deep_research", "argument_generation"})
            elif all(item.reliability < cfg.low_reliability_threshold for item in claim_evidence):
                reason_codes.add("low_reliability_evidence")
                stages.add("deep_research")
            if _ocr_asr_conflict(claim_evidence):
                reason_codes.add("cross_modal_conflict")
                stages.update({"ocr", "asr", "human_contestation"})
            if _metadata_claim_conflict(claim_evidence):
                reason_codes.add("metadata_claim_conflict")
                stages.update({"metadata", "human_contestation"})
            if _reverse_context_conflict(claim_evidence):
                reason_codes.add("reverse_search_context_conflict")
                stages.update({"reverse_search", "deep_research"})
            if _forensic_source_conflict(claim_evidence):
                reason_codes.add("forensic_source_conflict")
                stages.update({"vlm_analysis", "metadata", "human_contestation"})
            if _support_attack_tie(claim_arguments, cfg.support_attack_tie_margin):
                reason_codes.add("support_attack_tie")
                stages.update({"argument_generation", "qbaf_reasoning"})
            decisions.append(
                EscalationDecision(
                    claim_id=claim_id,
                    should_escalate=bool(reason_codes),
                    reason_codes=sorted(reason_codes),
                    defer_to_human=bool(reason_codes),
                    stronger_verifier_recommended=bool(
                        reason_codes
                        & {
                            "cross_modal_conflict",
                            "forensic_source_conflict",
                            "reverse_search_context_conflict",
                        }
                    ),
                    affected_pipeline_stages=sorted(stages or {"human_contestation"} if reason_codes else []),
                    notes="; ".join(sorted(reason_codes)),
                )
            )
        return decisions


def _linked_evidence(arguments: list[Argument], evidence_by_id: dict[str, EvidenceItem]) -> list[EvidenceItem]:
    ids = {evidence_id for argument in arguments for evidence_id in argument.evidence_ids}
    return [evidence_by_id[evidence_id] for evidence_id in ids if evidence_id in evidence_by_id]


def _ocr_asr_conflict(evidence: list[EvidenceItem]) -> bool:
    ocr = " ".join(item.content.lower() for item in evidence if item.source_type == "ocr")
    asr = " ".join(item.content.lower() for item in evidence if item.source_type == "asr")
    if not ocr or not asr:
        return False
    return any(token in ocr for token in ("today", "live", "now")) and any(
        token in asr for token in ("yesterday", "old", "last year")
    )


def _metadata_claim_conflict(evidence: list[EvidenceItem]) -> bool:
    text = " ".join(f"{item.content} {item.metadata}".lower() for item in evidence)
    return (
        "creation" in text
        and "claimed" in text
        and any(token in text for token in ("contradict", "different date", "older"))
    )


def _reverse_context_conflict(evidence: list[EvidenceItem]) -> bool:
    text = " ".join(
        item.content.lower()
        for item in evidence
        if item.source_type in {"reverse_image_local", "reverse_image_web_candidate"}
    )
    return any(token in text for token in ("old", "previous", "earlier", "first appeared", "two years"))


def _forensic_source_conflict(evidence: list[EvidenceItem]) -> bool:
    forensic_attack = any(
        item.source_type == "forensic_analysis"
        and (
            item.uncertainty_flags
            or any(token in item.content.lower() for token in ("manipulation", "suspicious", "caution"))
        )
        for item in evidence
    )
    source_support = any(
        item.source_type in {"web_article", "news_article", "factcheck_article", "case_provided"}
        and item.metadata.get("stance") == "support"
        for item in evidence
    )
    return forensic_attack and source_support


def _support_attack_tie(arguments: list[Argument], margin: float) -> bool:
    support = sum(argument.score or argument.intrinsic_score for argument in arguments if argument.stance == "support")
    attack = sum(argument.score or argument.intrinsic_score for argument in arguments if argument.stance == "attack")
    if support == 0 and attack == 0:
        return False
    return abs(support - attack) <= margin
