from __future__ import annotations

from src.argumentation.contribution_analyzer import ContributionAnalyzer
from src.schemas.memory_schema import MemoryUpdateCandidate
from src.schemas.report_schema import ReflectionLog, VerificationReport
from src.utils.hashing import stable_hash_text
from src.utils.llm_client import LLMClient


class LessonGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self.contribution_analyzer = ContributionAnalyzer()

    def generate(
        self,
        report: VerificationReport,
        reflection: ReflectionLog,
    ) -> list[MemoryUpdateCandidate]:
        contributions = self.contribution_analyzer.analyze(report.subclaim_reports)
        candidates = [
            MemoryUpdateCandidate(
                candidate_id=f"cand_{stable_hash_text(report.case_id + 'episodic')}",
                memory_type="episodic",
                text=(
                    f"Case {report.case_id} ended as {report.final_status} with "
                    f"confidence {report.final_confidence:.2f}; main uncertainty: "
                    f"{', '.join(reflection.failure_modes) or 'none'}."
                ),
                source_case_id=report.case_id,
                confidence=max(0.6, report.final_confidence),
                rationale="Store successful or failed case-level reasoning trace.",
                metadata={"argument_contributions": contributions},
            )
        ]
        if reflection.failure_modes:
            candidates.append(
                MemoryUpdateCandidate(
                    candidate_id=f"cand_{stable_hash_text(report.case_id + 'failure')}",
                    memory_type="failure",
                    text=(
                        "When verification has adapter gaps, cache misses, or label mismatch, "
                        "preserve uncertainty instead of upgrading weak evidence to support."
                    ),
                    source_case_id=report.case_id,
                    confidence=0.75,
                    rationale="Derived from observed failure or uncertainty modes.",
                )
            )
        candidates.append(
            MemoryUpdateCandidate(
                candidate_id=f"cand_{stable_hash_text(report.case_id + 'semantic')}",
                memory_type="semantic_rule",
                text=(
                    "A multimedia claim should remain uncertain when provenance is weak and "
                    "supporting evidence does not directly address authenticity or context."
                ),
                source_case_id=report.case_id,
                claim_type="authenticity",
                confidence=0.7,
                rationale="General rule for cheapfake and out-of-context verification.",
            )
        )
        return candidates
