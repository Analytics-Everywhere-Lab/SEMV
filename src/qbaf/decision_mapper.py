from __future__ import annotations


class DecisionMapper:
    def map(self, score: float) -> tuple[str, float]:
        if score < 0.30:
            decision = "refuted"
        elif score < 0.45:
            decision = "weakly_refuted"
        elif score <= 0.55:
            decision = "uncertain"
        elif score <= 0.70:
            decision = "weakly_supported"
        else:
            decision = "supported"
        confidence = abs(score - 0.5) * 2.0
        if decision == "uncertain":
            confidence = 1.0 - min(abs(score - 0.5) / 0.05, 1.0)
        return decision, round(confidence, 4)

    def to_subclaim_report(self, claim, graph, arguments):
        from src.main import _top_arguments, _uncertainty_reason
        from src.schemas.report_schema import SubClaimReport

        decision, confidence = self.map(graph.claim_score)
        return SubClaimReport(
            claim_id=claim.claim_id,
            claim_type=claim.claim_type,
            statement=claim.statement,
            score=graph.claim_score,
            decision=decision,
            confidence=confidence,
            top_support_arguments=_top_arguments(arguments, "support"),
            top_attack_arguments=_top_arguments(arguments, "attack"),
            uncertainty_reason=_uncertainty_reason(arguments, graph.uncertainty_flags),
        )
