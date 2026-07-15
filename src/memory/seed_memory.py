from __future__ import annotations

from pathlib import Path

from src.memory.memory_similarity import canonical_key
from src.memory.memory_store import MemoryStore
from src.schemas.memory_schema import MemoryRecord


SEED_SEMANTIC_RULES = [
    {
        "memory_id": "rule_when_publication_bound_001",
        "memory_type": "semantic_rule",
        "claim_type": "when",
        "task_type": "multimedia_verification",
        "trigger_pattern": "only publication time is known; original recording metadata is unavailable",
        "lesson": "A publication timestamp establishes a latest-known occurrence bound, not an exact event time. Verify a bounded temporal claim instead of overclaiming exact timing.",
        "evidence_pattern": "publication timestamp; missing original metadata; no trusted recording timestamp",
        "argument_pattern": "Support event occurred no later than publication time; mark exact recording time as uncertain.",
        "recommended_action": "Map When to partially_verified when only a bounded interval is verified.",
        "confidence": 0.93,
        "support_count": 3,
        "status": "active",
    },
    {
        "memory_id": "rule_ooc_context_001",
        "memory_type": "semantic_rule",
        "claim_type": "general",
        "task_type": "cheapfake_or_out_of_context_verification",
        "trigger_pattern": "authentic media is paired with a false location, time, entity, or event caption",
        "lesson": "If the media appears authentic but the claimed Where, When, Who, or event context is contradicted, classify the case as false_context or out_of_context_cheapfake rather than manipulated_or_synthetic.",
        "evidence_pattern": "reverse search; earlier appearance; different geolocation; caption-media mismatch; no manipulation evidence",
        "argument_pattern": "Support authenticity but attack the false contextual subclaim.",
        "recommended_action": "Keep authenticity separate from contextual truth.",
        "confidence": 0.95,
        "support_count": 3,
        "status": "active",
    },
    {
        "memory_id": "rule_where_camera_target_001",
        "memory_type": "semantic_rule",
        "claim_type": "where",
        "task_type": "multimedia_verification",
        "trigger_pattern": "video is filmed from one location while the target event is visible at a distance",
        "lesson": "Separate camera location from target event location. A verified filming location does not automatically prove the exact target location.",
        "evidence_pattern": "distant explosion; smoke plume; known rooftop; view direction; skyline; shadows; map comparison",
        "argument_pattern": "Generate separate arguments for camera geolocation and target event geolocation.",
        "recommended_action": "Create Where-camera and Where-target claims when needed.",
        "confidence": 0.92,
        "support_count": 2,
        "status": "active",
    },
    {
        "memory_id": "rule_multiview_same_incident_001",
        "memory_type": "semantic_rule",
        "claim_type": "what",
        "task_type": "multimedia_verification",
        "trigger_pattern": "multiple videos show different viewpoints or stages of the same incident",
        "lesson": "Verify same-incident consistency using shared landmarks, actors, vehicles, sequence continuity, and independent source timing before merging the videos into one event claim.",
        "evidence_pattern": "multiple videos; same location; related timestamps; different viewpoints; same actors or vehicles",
        "argument_pattern": "Generate support for same-event linkage only when visual continuity and timing agree.",
        "recommended_action": "Build event clusters before final What and When reasoning.",
        "confidence": 0.9,
        "support_count": 2,
        "status": "active",
    },
    {
        "memory_id": "rule_why_low_confidence_001",
        "memory_type": "semantic_rule",
        "claim_type": "why",
        "task_type": "multimedia_verification",
        "trigger_pattern": "motivation or narrative intent is inferred from limited caption or source evidence",
        "lesson": "Why claims are interpretive and should receive lower confidence unless supported by explicit caption framing, repeated source behavior, hashtags, article narrative, or corroborating context.",
        "evidence_pattern": "limited caption; no explicit intent; no source analysis; no narrative corroboration",
        "argument_pattern": "Generate uncertainty argument when event documentation does not establish poster motivation.",
        "recommended_action": "Do not let Why dominate the final label unless the task specifically evaluates narrative framing.",
        "confidence": 0.87,
        "support_count": 3,
        "status": "active",
    },
]


def seed_semantic_rules(
    memory_dir: str | Path | None = None,
    store: MemoryStore | None = None,
) -> list[MemoryRecord]:
    store = store or MemoryStore(Path(memory_dir) if memory_dir else None)
    existing = store.load_all()
    existing_ids = {record.memory_id for record in existing}
    existing_keys = {record.canonical_key for record in existing if record.canonical_key}
    inserted: list[MemoryRecord] = []
    for row in SEED_SEMANTIC_RULES:
        record = MemoryRecord.model_validate({**row, "origin": "seed"})
        record = record.model_copy(
            update={
                "canonical_key": canonical_key(
                    record.memory_type, record.claim_type, record.task_type, record.text
                )
            }
        )
        if record.memory_id in existing_ids or record.canonical_key in existing_keys:
            continue
        store.append(record)
        inserted.append(record)
    return inserted
