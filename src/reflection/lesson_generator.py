from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.argumentation.contribution_analyzer import ContributionAnalyzer
from src.memory.memory_similarity import canonical_key, normalize_text, semantic_signature
from src.schemas.memory_schema import MemoryUpdateCandidate, SupervisionSource
from src.schemas.report_schema import ReflectionLog, VerificationReport
from src.utils.hashing import stable_hash_text
from src.utils.llm_client import LLMClient


logger = logging.getLogger("run_case")


class _EpisodicLesson(BaseModel):
    observation: str
    what_worked_or_failed: str | None = None
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    argument_ids: list[str] = Field(default_factory=list)


class _FailureLesson(BaseModel):
    failure_type: str
    trigger_pattern: str
    lesson: str
    recommended_action: str
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    argument_ids: list[str] = Field(default_factory=list)


class _SemanticCandidate(BaseModel):
    generalizable_pattern: str
    lesson: str
    trigger_pattern: str
    evidence_pattern: str | None = None
    argument_pattern: str | None = None
    recommended_action: str
    applicability_scope: str | None = None
    exceptions: list[str] = Field(default_factory=list)
    claim_type: str | None = None
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    argument_ids: list[str] = Field(default_factory=list)


class _LessonResponse(BaseModel):
    episodic: _EpisodicLesson | None = None
    failures: list[_FailureLesson] = Field(default_factory=list)
    semantic: _SemanticCandidate | None = None


class LessonGenerator:
    """Generates case-specific, grounded memory candidates from one structured
    LLM reflection call (never one call per candidate).

    - Episodic observation: what specifically happened in this case.
    - Failure lessons: what failed, its trigger, and a corrective action.
    - Semantic candidate: a possible general principle, produced only when the
      case contains a generalizable pattern; it is not yet a trusted rule.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self.contribution_analyzer = ContributionAnalyzer()

    def generate(
        self,
        report: VerificationReport,
        reflection: ReflectionLog,
    ) -> list[MemoryUpdateCandidate]:
        contributions = self.contribution_analyzer.analyze(report.subclaim_reports)
        context = self._case_context(report, reflection, contributions)
        try:
            data = self.llm_client.generate_json(self._prompt(report, reflection, context))
            response = _LessonResponse.model_validate(data)
        except (Exception, ValidationError) as exc:
            logger.warning("Structured lesson generation failed for %s: %s", report.case_id, exc)
            return [self._fallback_episodic(report, reflection, context, contributions)]

        candidates: list[MemoryUpdateCandidate] = []
        episodic = response.episodic or _EpisodicLesson(
            observation=self._default_observation(report, reflection),
        )
        candidates.append(
            self._episodic_candidate(report, reflection, context, contributions, episodic)
        )
        for index, failure in enumerate(response.failures):
            candidates.append(self._failure_candidate(report, reflection, context, failure, index))
        if response.semantic is not None and self._is_generalizable(response.semantic, reflection):
            candidates.append(self._semantic_candidate(report, reflection, context, response.semantic))
        return candidates

    # ----------------------------------------------------------------- prompt

    def _prompt(
        self,
        report: VerificationReport,
        reflection: ReflectionLog,
        context: dict[str, Any],
    ) -> str:
        return (
            "You are the reflection module of a multimedia verification system. "
            "From this finished case, produce structured lessons as JSON only:\n"
            "{\n"
            '  "episodic": {"observation": "what specifically happened in this case",'
            ' "what_worked_or_failed": "...", "confidence": 0.0-1.0,'
            ' "evidence_ids": [...], "argument_ids": [...]},\n'
            '  "failures": [{"failure_type": "...", "trigger_pattern": "...",'
            ' "lesson": "...", "recommended_action": "...", "confidence": 0.0-1.0,'
            ' "evidence_ids": [...], "argument_ids": [...]}],\n'
            '  "semantic": null or {"generalizable_pattern": "why this generalizes beyond this case",'
            ' "lesson": "...", "trigger_pattern": "...", "evidence_pattern": "...",'
            ' "argument_pattern": "...", "recommended_action": "...",'
            ' "applicability_scope": "...", "exceptions": [...], "claim_type": "...",'
            ' "confidence": 0.0-1.0, "evidence_ids": [...], "argument_ids": [...]}\n'
            "}\n"
            "Rules: use only evidence_ids and argument_ids listed below; set semantic to null "
            "unless the case exhibits a pattern that clearly generalizes to other cases; never "
            "restate case-specific names or labels as a universal rule; only add a failures entry "
            "for a failure mode that actually occurred.\n"
            f"Case id: {report.case_id}\n"
            f"Predicted label: {report.final_status} (confidence {report.final_confidence:.2f})\n"
            f"Gold label: {reflection.ground_truth_label or 'unavailable'}\n"
            f"Failure modes detected: {reflection.failure_modes or ['none']}\n"
            f"Subclaim decisions: {context['subclaim_summaries']}\n"
            f"Top argument contributions: {context['top_contributions']}\n"
            f"Evidence provenance: {context['evidence_summaries']}\n"
            f"Uncertainty flags: {context['uncertainty_flags']}\n"
            f"Human contestation: {context['human_summary']}\n"
            f"Report revision diff: {context['revision_summary']}\n"
            f"Valid evidence_ids: {context['valid_evidence_ids']}\n"
            f"Valid argument_ids: {context['valid_argument_ids']}\n"
        )

    def _case_context(
        self,
        report: VerificationReport,
        reflection: ReflectionLog,
        contributions: dict[str, float],
    ) -> dict[str, Any]:
        valid_evidence_ids = [item.evidence_id for item in report.evidence]
        arguments = [
            argument
            for sub in report.subclaim_reports
            for argument in sub.top_support_arguments + sub.top_attack_arguments
        ]
        valid_argument_ids = sorted({argument.argument_id for argument in arguments})
        ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        top_contributions = [
            {"argument_id": argument_id, "contribution": round(value, 3)}
            for argument_id, value in ranked[:6]
        ]
        evidence_summaries = [
            {
                "evidence_id": item.evidence_id,
                "source_type": item.source_type,
                "reliability": round(item.reliability, 2),
                "has_provenance": bool(item.provenance),
                "uncertainty_flags": item.uncertainty_flags[:4],
            }
            for item in report.evidence[:12]
        ]
        subclaim_summaries = [
            {
                "claim_type": sub.claim_type,
                "decision": sub.decision,
                "confidence": round(sub.confidence, 2),
                "uncertainty_reason": sub.uncertainty_reason,
            }
            for sub in report.subclaim_reports
        ]
        human = reflection.human_feedback or {}
        human_summary = "none"
        if human:
            diff = human.get("contestation_diff") or {}
            human_summary = {
                "changed_final_decision": diff.get("human_contestation_changed_final_decision"),
                "removed_arguments": diff.get("removed_arguments", []),
                "edited_arguments": diff.get("edited_arguments", []),
            }
        revision_summary = "none"
        diff = (human or {}).get("contestation_diff") or {}
        if diff:
            revision_summary = {
                "original_final_status": diff.get("original_final_status"),
                "revised_final_status": diff.get("revised_final_status"),
            }
        uncertainty_flags = sorted(
            {flag for item in report.evidence for flag in item.uncertainty_flags}
        )[:12]

        metadata = report.metadata or {}
        dataset = metadata.get("dataset") or {}
        task = metadata.get("task") or {}
        input_meta = metadata.get("input") or {}
        fingerprint_basis = " ".join(
            str(part)
            for part in [
                dataset.get("dataset_name") or "unknown",
                normalize_text(input_meta.get("title") or ""),
                normalize_text(input_meta.get("caption") or ""),
                normalize_text(input_meta.get("description") or ""),
            ]
        ).strip()
        if not any([input_meta.get("title"), input_meta.get("caption"), input_meta.get("description")]):
            fingerprint_basis = f"{dataset.get('dataset_name') or 'unknown'} {report.case_id}"
        return {
            "valid_evidence_ids": valid_evidence_ids,
            "valid_argument_ids": valid_argument_ids,
            "top_contributions": top_contributions,
            "evidence_summaries": evidence_summaries,
            "subclaim_summaries": subclaim_summaries,
            "human_summary": human_summary,
            "revision_summary": revision_summary,
            "uncertainty_flags": uncertainty_flags,
            "dataset_name": dataset.get("dataset_name"),
            "dataset_split": dataset.get("dataset_split") or dataset.get("split"),
            "task_type": task.get("task_type"),
            "source_fingerprint": f"fp_{stable_hash_text(fingerprint_basis)}",
        }

    # ------------------------------------------------------------- candidates

    def _base_fields(
        self,
        report: VerificationReport,
        reflection: ReflectionLog,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        supervision: SupervisionSource = "self_reflection"
        if reflection.human_feedback:
            supervision = "human_feedback"
        elif reflection.ground_truth_label:
            supervision = "gold_label"
        return {
            "source_case_id": report.case_id,
            "dataset_name": context["dataset_name"],
            "dataset_split": context["dataset_split"],
            "task_type": context["task_type"],
            "source_fingerprint": context["source_fingerprint"],
            "supervision_source": supervision,
        }

    def _grounding(
        self,
        context: dict[str, Any],
        evidence_ids: list[str],
        argument_ids: list[str],
        contributions: dict[str, float] | None = None,
    ) -> tuple[list[str], list[str], bool]:
        """Filter LLM-proposed grounding IDs to those present in the report.

        Returns (evidence_ids, argument_ids, had_invented_ids). If nothing valid
        remains, fall back to the top contributing arguments and their evidence."""
        valid_evidence = set(context["valid_evidence_ids"])
        valid_arguments = set(context["valid_argument_ids"])
        kept_evidence = [eid for eid in evidence_ids if eid in valid_evidence]
        kept_arguments = [aid for aid in argument_ids if aid in valid_arguments]
        invented = len(kept_evidence) != len(evidence_ids) or len(kept_arguments) != len(argument_ids)
        if not kept_evidence and not kept_arguments:
            kept_arguments = [row["argument_id"] for row in context["top_contributions"][:3]]
            kept_arguments = [aid for aid in kept_arguments if aid in valid_arguments]
            kept_evidence = context["valid_evidence_ids"][:3]
        del contributions
        return kept_evidence, kept_arguments, invented

    def _episodic_candidate(
        self,
        report: VerificationReport,
        reflection: ReflectionLog,
        context: dict[str, Any],
        contributions: dict[str, float],
        lesson: _EpisodicLesson,
    ) -> MemoryUpdateCandidate:
        evidence_ids, argument_ids, invented = self._grounding(
            context, lesson.evidence_ids, lesson.argument_ids
        )
        text = lesson.observation.strip() or self._default_observation(report, reflection)
        correct = (
            reflection.ground_truth_label is not None
            and reflection.ground_truth_label == report.final_status
        )
        confidence = min(1.0, max(lesson.confidence, report.final_confidence if correct else 0.0))
        return MemoryUpdateCandidate(
            candidate_id=f"cand_{stable_hash_text(report.case_id + 'episodic' + normalize_text(text))}",
            memory_type="episodic",
            text=text,
            claim_type="general",
            lesson=lesson.what_worked_or_failed,
            observed_failure_or_success=", ".join(reflection.failure_modes) or "no_failure_detected",
            normalized_text=normalize_text(text),
            canonical_key=canonical_key("episodic", "general", context["task_type"], text),
            semantic_signature=semantic_signature(
                "episodic", "general", context["task_type"], None, None, None
            ),
            polarity="positive" if correct else "negative",
            grounding_evidence_ids=evidence_ids,
            grounding_argument_ids=argument_ids,
            confidence=confidence,
            rationale="Case-level observation from post-prediction reflection.",
            verification_status="under_review" if invented else "pending",
            rejected_reason="candidate_referenced_unknown_ids" if invented else None,
            metadata={
                "argument_contributions": {
                    row["argument_id"]: row["contribution"] for row in context["top_contributions"]
                },
                "predicted_label": report.final_status,
                "gold_label": reflection.ground_truth_label,
            },
            **self._base_fields(report, reflection, context),
        )

    def _failure_candidate(
        self,
        report: VerificationReport,
        reflection: ReflectionLog,
        context: dict[str, Any],
        failure: _FailureLesson,
        index: int,
    ) -> MemoryUpdateCandidate:
        evidence_ids, argument_ids, invented = self._grounding(
            context, failure.evidence_ids, failure.argument_ids
        )
        text = f"{failure.lesson.strip()} Recommended action: {failure.recommended_action.strip()}"
        return MemoryUpdateCandidate(
            candidate_id=f"cand_{stable_hash_text(report.case_id + 'failure' + failure.failure_type + str(index))}",
            memory_type="failure",
            text=text,
            failure_type=failure.failure_type,
            trigger_pattern=failure.trigger_pattern,
            lesson=failure.lesson,
            recommended_action=failure.recommended_action,
            observed_failure_or_success=failure.failure_type,
            normalized_text=normalize_text(text),
            canonical_key=canonical_key("failure", failure.failure_type, context["task_type"], text),
            semantic_signature=semantic_signature(
                "failure", None, context["task_type"], failure.failure_type, "negative", None
            ),
            polarity="negative",
            grounding_evidence_ids=evidence_ids,
            grounding_argument_ids=argument_ids,
            confidence=failure.confidence,
            rationale="Failure lesson derived from observed failure modes.",
            verification_status="under_review" if invented else "pending",
            rejected_reason="candidate_referenced_unknown_ids" if invented else None,
            metadata={"detected_failure_modes": reflection.failure_modes},
            **self._base_fields(report, reflection, context),
        )

    def _semantic_candidate(
        self,
        report: VerificationReport,
        reflection: ReflectionLog,
        context: dict[str, Any],
        semantic: _SemanticCandidate,
    ) -> MemoryUpdateCandidate:
        evidence_ids, argument_ids, invented = self._grounding(
            context, semantic.evidence_ids, semantic.argument_ids
        )
        text = semantic.lesson.strip()
        return MemoryUpdateCandidate(
            candidate_id=f"cand_{stable_hash_text(report.case_id + 'semantic' + normalize_text(text))}",
            memory_type="semantic_rule",
            text=text,
            claim_type=semantic.claim_type or "general",
            trigger_pattern=semantic.trigger_pattern,
            lesson=semantic.lesson,
            evidence_pattern=semantic.evidence_pattern,
            argument_pattern=semantic.argument_pattern,
            recommended_action=semantic.recommended_action,
            applicability_scope=semantic.applicability_scope,
            exceptions=list(semantic.exceptions),
            normalized_text=normalize_text(text),
            canonical_key=canonical_key(
                "semantic_rule", semantic.claim_type or "general", context["task_type"], text
            ),
            semantic_signature=semantic_signature(
                "semantic_rule",
                semantic.claim_type or "general",
                context["task_type"],
                None,
                None,
                semantic.applicability_scope,
            ),
            polarity=None,
            grounding_evidence_ids=evidence_ids,
            grounding_argument_ids=argument_ids,
            confidence=semantic.confidence,
            rationale=f"Proposed general principle: {semantic.generalizable_pattern}",
            verification_status="under_review" if invented else "pending",
            rejected_reason="candidate_referenced_unknown_ids" if invented else None,
            metadata={"generalizable_pattern": semantic.generalizable_pattern},
            **self._base_fields(report, reflection, context),
        )

    def _fallback_episodic(
        self,
        report: VerificationReport,
        reflection: ReflectionLog,
        context: dict[str, Any],
        contributions: dict[str, float],
    ) -> MemoryUpdateCandidate:
        """Structured generation failed: emit at most one grounded episodic
        candidate held for review. No generic semantic fallback sentence."""
        text = self._default_observation(report, reflection)
        evidence_ids, argument_ids, _ = self._grounding(context, [], [])
        return MemoryUpdateCandidate(
            candidate_id=f"cand_{stable_hash_text(report.case_id + 'episodic_fallback')}",
            memory_type="episodic",
            text=text,
            claim_type="general",
            observed_failure_or_success=", ".join(reflection.failure_modes) or "no_failure_detected",
            normalized_text=normalize_text(text),
            canonical_key=canonical_key("episodic", "general", context["task_type"], text),
            semantic_signature=semantic_signature(
                "episodic", "general", context["task_type"], None, None, None
            ),
            grounding_evidence_ids=evidence_ids,
            grounding_argument_ids=argument_ids,
            confidence=min(0.6, report.final_confidence),
            rationale="Structured lesson generation failed; deterministic episodic fallback.",
            verification_status="under_review",
            rejected_reason="structured_lesson_generation_failed",
            metadata={
                "argument_contributions": {
                    row["argument_id"]: row["contribution"] for row in context["top_contributions"]
                }
            },
            **self._base_fields(report, reflection, context),
        )

    @staticmethod
    def _default_observation(report: VerificationReport, reflection: ReflectionLog) -> str:
        return (
            f"Case {report.case_id} ended as {report.final_status} with confidence "
            f"{report.final_confidence:.2f}; gold label: "
            f"{reflection.ground_truth_label or 'unavailable'}; observed modes: "
            f"{', '.join(reflection.failure_modes) or 'none'}."
        )

    @staticmethod
    def _is_generalizable(semantic: _SemanticCandidate, reflection: ReflectionLog) -> bool:
        """A semantic candidate needs an explicit generalizable pattern plus a
        trigger and action; otherwise the case only yields episodic/failure lessons."""
        if not semantic.generalizable_pattern.strip():
            return False
        if not semantic.trigger_pattern.strip() or not semantic.recommended_action.strip():
            return False
        del reflection
        return True
