from __future__ import annotations

from src.schemas.report_schema import SubClaimReport


class FinalDecisionAggregator:
    def aggregate(
        self,
        subclaim_reports: list[SubClaimReport],
        bundle: object | None = None,
    ) -> tuple[str, float]:
        if not subclaim_reports:
            return "uncertain", 0.0
        by_type: dict[str, list[SubClaimReport]] = {}
        for report in subclaim_reports:
            by_type.setdefault(report.claim_type, []).append(report)

        def avg_score(claim_type: str) -> float | None:
            reports = by_type.get(claim_type, [])
            if not reports:
                return None
            return sum(report.score for report in reports) / len(reports)

        def is_refuted(claim_type: str) -> bool:
            return any(
                report.decision in {"refuted", "weakly_refuted"}
                for report in by_type.get(claim_type, [])
            )

        def is_supported(claim_type: str) -> bool:
            return any(
                report.decision in {"supported", "weakly_supported"}
                for report in by_type.get(claim_type, [])
            )

        scores = [report.score for report in subclaim_reports]
        aggregate_score = sum(scores) / len(scores)
        confidence = round(abs(aggregate_score - 0.5) * 2.0, 4)

        if is_refuted("authenticity"):
            return "manipulated_or_synthetic", max(confidence, 0.55)

        context_refuted = any(
            is_refuted(claim_type)
            for claim_type in ["where", "when", "who", "caption_context"]
        )
        what_supported = is_supported("what") or is_supported("main")
        if what_supported and context_refuted:
            label = (
                "out_of_context_cheapfake"
                if getattr(getattr(bundle, "dataset", None), "dataset_name", "") == "cosmos"
                else "false_context"
            )
            return label, max(confidence, 0.55)

        core_types = ["what", "where", "when", "who", "authenticity"]
        present_core = [claim_type for claim_type in core_types if by_type.get(claim_type)]
        if present_core and all(is_supported(claim_type) for claim_type in present_core):
            if all((avg_score(claim_type) or 0.0) >= 0.70 for claim_type in present_core):
                return "verified", max(confidence, 0.65)
            return "mostly_verified", max(confidence, 0.55)

        if any(report.decision == "weakly_supported" for report in subclaim_reports):
            return "partially_verified", max(confidence, 0.45)

        uncertain_count = sum(
            1 for report in subclaim_reports if report.decision == "uncertain"
        )
        if uncertain_count >= max(1, len(subclaim_reports) // 2):
            return "uncertain", round(1.0 - min(confidence, 1.0), 4)

        if aggregate_score < 0.45:
            return "insufficient_evidence", max(confidence, 0.35)
        return "uncertain", round(1.0 - min(confidence, 1.0), 4)
