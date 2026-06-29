from __future__ import annotations

from src.schemas.report_schema import VerificationReport


class FailureClassifier:
    def classify(
        self,
        report: VerificationReport,
        ground_truth_label: str | None,
        human_feedback: dict | None,
    ) -> list[str]:
        modes: list[str] = []
        if ground_truth_label and ground_truth_label != report.final_status:
            modes.append("final_label_mismatch")
        if any(item.uncertainty_flags for item in report.evidence):
            modes.append("tool_or_adapter_uncertainty")
        if any(
            sub.uncertainty_reason and "clash" in sub.uncertainty_reason.lower()
            for sub in report.subclaim_reports
        ):
            modes.append("unresolved_argument_clash")
        if human_feedback:
            modes.append("human_feedback_available")
        return sorted(set(modes))
