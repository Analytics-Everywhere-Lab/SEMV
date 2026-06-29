from __future__ import annotations

from pathlib import Path

from src.schemas.report_schema import VerificationReport


class MarkdownRenderer:
    def render(self, report: VerificationReport) -> str:
        lines = [
            f"# Verification Report: {report.case_id}",
            "",
            "## Case Summary",
            f"- Case ID: `{report.case_id}`",
            f"- Generated at: {report.generated_at.isoformat()}",
            "",
            "## Final Verification Status",
            f"- Final status: **{report.final_status}**",
            f"- Final confidence: **{report.final_confidence:.2f}**",
            "",
            "## Content Classification / Tags",
            f"- Dataset: {report.metadata.get('dataset', {}).get('dataset_name', 'unknown') if isinstance(report.metadata.get('dataset'), dict) else 'unknown'}",
            f"- Task: {report.metadata.get('task', {}).get('task_type', 'unknown') if isinstance(report.metadata.get('task'), dict) else 'unknown'}",
            "",
            "## Subclaim Verification Table",
            "| Claim | Decision | Score | Confidence |",
            "| --- | --- | ---: | ---: |",
        ]
        for subclaim in report.subclaim_reports:
            lines.append(
                f"| {subclaim.claim_type} | {subclaim.decision} | {subclaim.score:.3f} | {subclaim.confidence:.3f} |"
            )

        for heading, claim_types in [
            ("What", {"what", "main", "caption_context"}),
            ("Where", {"where"}),
            ("When", {"when"}),
            ("Who", {"who"}),
            ("Why", {"why"}),
            ("Authenticity / Forensic Analysis", {"authenticity"}),
        ]:
            lines.extend(["", f"## {heading}"])
            matched = [r for r in report.subclaim_reports if r.claim_type in claim_types]
            if not matched:
                lines.append("- Not applicable or not generated.")
            for subclaim in matched:
                lines.extend(_render_subclaim(subclaim))

        lines.extend(["", "## Evidence Pool"])
        for item in report.evidence:
            url = f" {item.url}" if item.url else ""
            lines.append(
                f"- `{item.evidence_id}` {item.title or item.source}{url} "
                f"(reliability={item.reliability:.2f}, relevance={item.relevance:.2f})"
            )

        lines.extend(["", "## QBAF Reasoning"])
        for subclaim in report.subclaim_reports:
            lines.append(
                f"- `{subclaim.claim_id}` {subclaim.decision} at {subclaim.score:.3f}; "
                f"support={len(subclaim.top_support_arguments)}, attack={len(subclaim.top_attack_arguments)}"
            )

        lines.extend(["", "## Memory Used"])
        if report.memory_used:
            for item in report.memory_used:
                lines.append(f"- `{item.memory_id}` {item.text[:240]}")
        else:
            lines.append("- No memory records were used.")

        lines.extend(["", "## Uncertainty Notes"])
        if report.uncertainty_flags:
            lines.extend(f"- {flag}" for flag in report.uncertainty_flags)
        else:
            lines.append("- No major uncertainty flags recorded.")

        lines.extend(["", "## Contestation Log", "- No human contestation recorded."])

        lines.extend(["", "## Reflection and Memory Update Candidate"])
        if report.reflection_logs:
            for reflection in report.reflection_logs:
                lines.append(
                    f"- Failure modes: {', '.join(reflection.failure_modes) or 'none'}"
                )
                lines.extend(f"  - Lesson: {lesson}" for lesson in reflection.lessons)
        elif report.memory_update_candidates:
            lines.extend(f"- {candidate.text}" for candidate in report.memory_update_candidates)
        else:
            lines.append("- No reflection was run for this case.")
        return "\n".join(lines) + "\n"

    def render_to_file(self, report: VerificationReport, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(report), encoding="utf-8")


def _render_subclaim(subclaim) -> list[str]:
    lines = [
        f"### {subclaim.claim_type.title()}",
        f"- Decision: **{subclaim.decision}**",
        f"- Score: {subclaim.score:.3f}",
        f"- Confidence: {subclaim.confidence:.3f}",
        f"- Statement: {subclaim.statement}",
    ]
    if subclaim.uncertainty_reason:
        lines.append(f"- Uncertainty: {subclaim.uncertainty_reason}")
    lines.append("- Top support:")
    lines.extend(
        f"  - {arg.title} ({arg.score:.2f}): {arg.text[:240]}"
        for arg in subclaim.top_support_arguments
    )
    lines.append("- Top attack:")
    lines.extend(
        f"  - {arg.title} ({arg.score:.2f}): {arg.text[:240]}"
        for arg in subclaim.top_attack_arguments
    )
    return lines
