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
            media_analysis=_media_analysis_summary(evidence),
            metadata={"claim": case.claim, "context": case.context},
        )


def _media_analysis_summary(evidence: list[EvidenceItem]) -> dict[str, list[dict]]:
    buckets = {
        "metadata": {"media_metadata", "metadata_exiftool", "metadata_ffprobe"},
        "keyframes": {"scene_keyframe", "keyframe"},
        "ocr": {"ocr"},
        "asr": {"asr"},
        "visual_analysis": {"visual_caption", "visual_objects", "visual_vqa", "frame_analysis"},
        "forensics": {"forensic_analysis"},
        "reverse_similarity": {"reverse_image_local", "visual_similarity"},
        "web_candidate_matches": {"reverse_image_web_candidate"},
        "geolocation_candidates": {"geolocation_candidate"},
        "tool_availability": {"synthetic_uncertainty"},
    }
    summary = {key: [] for key in buckets}
    for item in evidence:
        for key, source_types in buckets.items():
            if item.source_type not in source_types:
                continue
            if key == "tool_availability" and not _is_tool_availability_item(item):
                continue
            summary[key].append(
                {
                    "evidence_id": item.evidence_id,
                    "source_type": item.source_type,
                    "title": item.title,
                    "content": item.content,
                    "media_path": item.media_path,
                    "frame_path": item.frame_path,
                    "reliability": item.reliability,
                    "uncertainty_flags": item.uncertainty_flags,
                    "raw_output": item.raw_output,
                }
            )
    return summary


def _is_tool_availability_item(item: EvidenceItem) -> bool:
    if item.source_type != "synthetic_uncertainty":
        return False
    if item.media_path or item.frame_path:
        return True
    if item.provenance:
        return bool(item.provenance.metadata.get("adapter"))
    return False
