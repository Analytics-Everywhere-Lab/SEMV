from __future__ import annotations

from src.schemas.case_bundle_schema import CaseBundle, Claim


def build_memory_query(bundle: CaseBundle, claim: Claim) -> dict:
    return {
        "case_id": bundle.case_id,
        "task_type": bundle.task.task_type,
        "subtask": bundle.task.subtask,
        "claim_type": claim.claim_type,
        "claim_statement": claim.statement,
        "scope_type": claim.scope_type,
        "media_type": bundle.task.media_type,
        "title": bundle.input.title,
        "caption": bundle.input.caption,
        "description": bundle.input.description,
        "location_hint": bundle.input.location_hint,
        "temporal_signals": bundle.temporal_context.time_reasoning_signals,
        "geolocation_cues": bundle.location_context.geolocation_cues,
        "source_cluster_names": [
            cluster.source_name for cluster in bundle.source_clusters if cluster.source_name
        ],
        "platforms": [cluster.platform for cluster in bundle.source_clusters if cluster.platform],
        "media_roles": [media.role for media in bundle.media_assets],
    }
