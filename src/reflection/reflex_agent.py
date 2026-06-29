from __future__ import annotations

from src.memory.memory_consolidator import MemoryConsolidator
from src.memory.memory_verifier import MemoryVerifier
from src.reflection.failure_classifier import FailureClassifier
from src.reflection.lesson_generator import LessonGenerator
from src.schemas.memory_schema import MemoryUpdateCandidate
from src.schemas.report_schema import ReflectionLog, VerificationReport
from src.utils.llm_client import LLMClient


class ReflectionAgent:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self.failure_classifier = FailureClassifier()
        self.lesson_generator = LessonGenerator(llm_client)
        self.memory_verifier = MemoryVerifier(llm_client)
        self.memory_consolidator = MemoryConsolidator()

    def reflect(
        self,
        report: VerificationReport,
        ground_truth_label: str | None = None,
        human_feedback: dict | None = None,
        update_memory: bool = False,
    ) -> tuple[ReflectionLog, list[MemoryUpdateCandidate]]:
        failure_modes = self.failure_classifier.classify(
            report,
            ground_truth_label,
            human_feedback,
        )
        reflection = ReflectionLog(
            case_id=report.case_id,
            predicted_label=report.final_status,
            ground_truth_label=ground_truth_label,
            human_feedback=human_feedback or {},
            failure_modes=failure_modes,
            lessons=[],
        )
        candidates = self.lesson_generator.generate(report, reflection)
        verified_candidates = [self.memory_verifier.verify(candidate) for candidate in candidates]
        reflection.lessons = [
            candidate.text for candidate in verified_candidates if candidate.verified
        ]
        if update_memory:
            applied = self.memory_consolidator.apply(verified_candidates)
            report.memory_updates_applied = applied
        return reflection, verified_candidates
