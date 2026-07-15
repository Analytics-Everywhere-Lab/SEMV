from __future__ import annotations

from src.evaluation.label_normalizer import normalize_mv2026_label
from src.schemas.report_schema import VerificationReport


CONTEXT_LABELS = {
    "false_context",
    "out_of_context_cheapfake",
    "miscaptioned",
    "out_of_context",
}
MANIPULATION_LABELS = {"manipulated_or_synthetic", "manipulated", "synthetic", "fake_media"}

TOOL_FAILURE_FLAG_HINTS = (
    "cache_miss",
    "tool_failed",
    "adapter_unavailable",
    "search_failed",
    "search_unavailable",
    "media_file_missing",
    "unreadable",
)


class FailureClassifier:
    """Deterministic failure/success mode detection from the finished report."""

    def classify(
        self,
        report: VerificationReport,
        ground_truth_label: str | None,
        human_feedback: dict | None,
    ) -> list[str]:
        modes: list[str] = []
        predicted = normalize_mv2026_label(report.final_status)
        ground_truth_label = normalize_mv2026_label(ground_truth_label) if ground_truth_label else None
        if ground_truth_label and ground_truth_label != predicted:
            modes.append("final_label_mismatch")
            if report.final_confidence >= 0.8:
                modes.append("overconfident_wrong_prediction")
            if self._context_authenticity_confusion(predicted, ground_truth_label):
                modes.append("context_authenticity_confusion")
        if ground_truth_label and ground_truth_label == predicted:
            if report.final_confidence >= 0.7:
                modes.append("successful_strategy")

        all_flags = {flag for item in report.evidence for flag in item.uncertainty_flags}
        if any(item.uncertainty_flags for item in report.evidence):
            modes.append("tool_or_adapter_uncertainty")
        if any(self._is_tool_failure_flag(flag) for flag in all_flags):
            modes.append("retrieval_or_tool_failure")

        for sub in report.subclaim_reports:
            reason = (sub.uncertainty_reason or "").lower()
            if "clash" in reason:
                modes.append("unresolved_argument_clash")
            if sub.claim_type == "when" and ("publication" in reason or "publish" in reason):
                modes.append("temporal_publication_event_confusion")
            if sub.claim_type == "where" and ("camera" in reason or "filming" in reason):
                modes.append("camera_target_location_conflation")
            for argument in sub.top_support_arguments:
                if not argument.evidence_ids:
                    modes.append("ungrounded_supporting_argument")

        if report.evidence and all(not item.provenance for item in report.evidence):
            modes.append("weak_or_missing_provenance")
        elif report.evidence:
            reliabilities = [item.reliability for item in report.evidence]
            if reliabilities and max(reliabilities) < 0.35:
                modes.append("weak_or_missing_provenance")

        if human_feedback:
            modes.append("human_feedback_available")
            if self._human_rejected_or_edited(human_feedback):
                modes.append("human_rejection_or_edit")

        return sorted(set(modes))

    @staticmethod
    def _context_authenticity_confusion(predicted: str, gold: str) -> bool:
        return (predicted in CONTEXT_LABELS and gold in MANIPULATION_LABELS) or (
            predicted in MANIPULATION_LABELS and gold in CONTEXT_LABELS
        )

    @staticmethod
    def _is_tool_failure_flag(flag: str) -> bool:
        lowered = flag.lower()
        return any(hint in lowered for hint in TOOL_FAILURE_FLAG_HINTS)

    @staticmethod
    def _human_rejected_or_edited(human_feedback: dict) -> bool:
        batch = human_feedback.get("human_review_batch") or {}
        for review in batch.get("argument_reviews", []) or []:
            if review.get("decision") in {"reject", "edit"}:
                return True
        for review in batch.get("evidence_reviews", []) or []:
            if review.get("decision") in {"reject", "edit"}:
                return True
        summary = human_feedback.get("contestation_diff") or {}
        return bool(summary.get("removed_arguments") or summary.get("edited_arguments"))
