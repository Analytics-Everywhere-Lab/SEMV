from __future__ import annotations

from src.schemas.case_schema import MultimediaCase
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem
from src.schemas.memory_schema import MemoryRecord
from src.schemas.report_schema import SubClaimReport, VerificationReport
from src.utils.llm_client import LLMClient


class ReportGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        case: MultimediaCase,
        final_status: str,
        final_confidence: float,
        subclaim_reports: list[SubClaimReport],
        evidence: list[EvidenceItem],
        evidence_graph: EvidenceGraph,
        memory_used: list[MemoryRecord],
    ) -> VerificationReport:
        uncertainty_flags = sorted(
            {
                flag
                for item in evidence
                for flag in item.uncertainty_flags
            }
            | {
                "subclaim_uncertainty"
                for report in subclaim_reports
                if report.decision == "uncertain"
            }
        )
        return VerificationReport(
            case_id=case.case_id,
            final_status=final_status,
            final_confidence=final_confidence,
            subclaim_reports=subclaim_reports,
            evidence=evidence,
            evidence_graph=evidence_graph,
            memory_used=memory_used,
            uncertainty_flags=uncertainty_flags,
            metadata={"claim": case.claim, "context": case.context},
        )
