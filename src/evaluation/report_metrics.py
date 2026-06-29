from __future__ import annotations

REQUIRED_SECTIONS = [
    "Case Summary",
    "Final Verification Status",
    "Content Classification / Tags",
    "Source Details",
    "What",
    "Where",
    "When",
    "Who",
    "Why",
    "Authenticity / Forensic Analysis",
    "Supporting Sources",
    "Evidence Table",
    "QBAF Reasoning",
    "Uncertainty Notes",
]


def report_structure_score(markdown: str) -> float:
    if not markdown:
        return 0.0
    present = sum(1 for section in REQUIRED_SECTIONS if section.lower() in markdown.lower())
    return present / len(REQUIRED_SECTIONS)


def uncertainty_explanation_coverage(report) -> float:
    if not report.subclaim_reports:
        return 0.0
    return sum(1 for row in report.subclaim_reports if row.uncertainty_reason or row.decision != "uncertain") / len(report.subclaim_reports)
