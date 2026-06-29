from __future__ import annotations

from src.schemas.argument_schema import Argument
from src.schemas.report_schema import SubClaimReport


class ContributionAnalyzer:
    def analyze(self, subclaim_reports: list[SubClaimReport]) -> dict[str, float]:
        contributions: dict[str, float] = {}
        for report in subclaim_reports:
            for argument in report.top_support_arguments + report.top_attack_arguments:
                sign = 1.0 if argument.stance == "support" else -1.0
                contributions[argument.argument_id] = sign * argument.score * report.confidence
        return contributions
