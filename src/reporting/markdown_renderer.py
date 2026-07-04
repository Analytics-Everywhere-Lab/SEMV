from __future__ import annotations

from pathlib import Path

from src.schemas.evidence_schema import EvidenceItem
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

        lines.extend(_render_media_analysis(report))

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

        lines.extend(_render_human_contestation(report))

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


_MEDIA_SOURCE_TYPES = {
    "media_metadata": "Metadata",
    "metadata_exiftool": "Metadata",
    "metadata_ffprobe": "Metadata",
    "scene_keyframe": "Keyframe",
    "keyframe": "Keyframe",
    "visual_caption": "VLM",
    "visual_objects": "VLM",
    "visual_vqa": "VLM",
    "frame_analysis": "VLM",
    "ocr": "OCR",
    "asr": "ASR",
    "forensic_analysis": "Forensics",
    "reverse_image_local": "Local reverse search",
    "reverse_image_web_candidate": "Web reverse search",
    "visual_similarity": "Visual similarity",
    "geolocation_candidate": "Geolocation clue",
}


def _render_media_analysis(report: VerificationReport) -> list[str]:
    media_items = [item for item in report.evidence if _is_media_analysis_item(item)]
    lines = ["", "## Media Analysis"]
    if not media_items:
        lines.append("- No media-derived analysis evidence was generated.")
        return lines

    lines.extend([
        "| Type | Evidence | Source / frame | Reliability | Key finding |",
        "| --- | --- | --- | ---: | --- |",
    ])
    for item in media_items:
        source = item.frame_path or item.media_path or item.source
        content = item.content
        if item.uncertainty_flags:
            content = f"{content} Flags: {', '.join(item.uncertainty_flags)}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _clean_table_text(_MEDIA_SOURCE_TYPES.get(item.source_type, item.source_type)),
                    f"`{_clean_table_text(item.evidence_id)}`",
                    _clean_table_text(source, limit=80),
                    f"{item.reliability:.2f}",
                    _clean_table_text(content, limit=220),
                ]
            )
            + " |"
        )
    return lines


def _is_media_analysis_item(item: EvidenceItem) -> bool:
    if item.source_type in _MEDIA_SOURCE_TYPES:
        return True
    if item.source_type != "synthetic_uncertainty":
        return False
    if item.media_path or item.frame_path:
        return True
    adapter = ""
    if item.provenance:
        adapter = str(item.provenance.metadata.get("adapter", ""))
    return adapter in {"metadata", "ocr", "asr", "vlm", "forensics", "scene_keyframe", "local_reverse_image_search"}


def _clean_table_text(value: object, limit: int = 120) -> str:
    text = str(value or "")
    text = " ".join(text.split()).replace("|", "\\|")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _render_human_contestation(report: VerificationReport) -> list[str]:
    lines = ["", "## Human Contestation and Adaptive Revision"]
    if not report.human_review_applied:
        lines.append("Human review applied: no.")
        return lines

    lines.append("Human review applied: yes")
    lines.extend(["", "### Human Actions", "| Action | Argument | Subclaim | Reason |", "|---|---|---|---|"])
    batch = report.human_review_batch
    if batch:
        for item in batch.contestations:
            argument = item.target_argument_id or ""
            subclaim = item.added_subclaim_id or ""
            reason = (item.reason or "").replace("|", "\\|")
            lines.append(f"| {item.action} | {argument} | {subclaim} | {reason} |")

    plan = report.revision_plan
    lines.extend(["", "### Revision Plan"])
    if plan:
        lines.extend([
            f"- Rerun from step: {plan.rerun_from_step}",
            f"- Revision target: {plan.revision_target}",
            f"- Rationale: {plan.rationale}",
            "- Affected arguments: " + (", ".join(plan.affected_argument_ids) or "none"),
            "- Affected evidence: " + (", ".join(plan.affected_evidence_ids) or "none"),
        ])
    else:
        lines.append("- Rerun from step: none")

    summary = report.contestation_summary
    lines.extend(["", "### Effect on Final Decision"])
    lines.extend([
        "- Original final decision: " + str(summary.get("original_final_status", "unknown")),
        "- Revised final decision: " + str(summary.get("revised_final_status", report.final_status)),
        "- Original confidence: " + str(summary.get("original_confidence", "unknown")),
        "- Revised confidence: " + str(summary.get("revised_confidence", report.final_confidence)),
    ])
    return lines


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
