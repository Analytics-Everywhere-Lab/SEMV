from __future__ import annotations

from src.evaluation.label_normalizer import normalize_mv2026_label
from src.memory.memory_service import MemoryService
from src.reflection.failure_classifier import FailureClassifier
from src.reflection.lesson_generator import LessonGenerator
from src.schemas.memory_schema import MemoryUpdateCandidate, ShortTermMemoryRecord
from src.schemas.report_schema import ReflectionLog, VerificationReport
from src.utils.llm_client import LLMClient


class ReflectionAgent:
    """Post-prediction reflection: classify failure modes, generate grounded
    memory candidates, verify them fail-closed, and stage the survivors into
    short-term memory. Nothing is written directly to long-term memory."""

    def __init__(
        self,
        llm_client: LLMClient,
        memory_service: MemoryService | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.memory_service = memory_service or MemoryService(llm_client=llm_client)
        self.failure_classifier = FailureClassifier()
        self.lesson_generator = LessonGenerator(llm_client)
        self.memory_verifier = self.memory_service.verifier(llm_client)

    def reflect(
        self,
        report: VerificationReport,
        ground_truth_label: str | None = None,
        human_feedback: dict | None = None,
        update_memory: bool = False,
    ) -> tuple[ReflectionLog, list[MemoryUpdateCandidate]]:
        normalized_gold = normalize_mv2026_label(ground_truth_label) if ground_truth_label else None
        normalized_predicted = normalize_mv2026_label(report.final_status)
        reflection_report = report.model_copy(update={"final_status": normalized_predicted})
        failure_modes = self.failure_classifier.classify(
            reflection_report, normalized_gold, human_feedback
        )
        reflection = ReflectionLog(
            case_id=report.case_id,
            predicted_label=normalized_predicted,
            ground_truth_label=normalized_gold,
            human_feedback=human_feedback or {},
            failure_modes=failure_modes,
            lessons=[],
        )
        candidates = self.lesson_generator.generate(reflection_report, reflection)
        valid_evidence_ids = {item.evidence_id for item in report.evidence}
        valid_argument_ids = {
            argument.argument_id
            for sub in report.subclaim_reports
            for argument in sub.top_support_arguments + sub.top_attack_arguments
        }
        verified_candidates = [
            self.memory_verifier.verify(
                candidate,
                valid_evidence_ids=valid_evidence_ids,
                valid_argument_ids=valid_argument_ids,
            )
            for candidate in candidates
        ]
        reflection.lessons = [
            candidate.text for candidate in verified_candidates if candidate.verified
        ]
        staged: list[ShortTermMemoryRecord] = []
        if update_memory and not self.memory_service.frozen:
            staged = self.memory_service.stage_candidates(verified_candidates)
            report.memory_updates_staged = staged
            consolidation = self.memory_service.register_case_processed()
            if consolidation is not None:
                report.memory_consolidation_events = [
                    event.model_dump(mode="json") for event in consolidation.events
                ]
                changed_ids = set(consolidation.changed_long_term_ids)
                report.memory_updates_applied = [
                    record
                    for record in self.memory_service.store.load_long_term()
                    if record.memory_id in changed_ids
                ]
        return reflection, verified_candidates
