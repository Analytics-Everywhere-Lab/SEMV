from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from src.memory.memory_config import MemoryConfig
from src.memory.memory_similarity import (
    SimilarityBackend,
    build_similarity_backend,
    canonical_key,
    content_tokens,
    normalize_text,
    semantic_signature,
)
from src.memory.memory_store import MemoryStore
from src.schemas.memory_schema import (
    ConsolidationEvent,
    ConsolidationResult,
    MemoryRecord,
    MemoryUpdateCandidate,
    ShortTermMemoryRecord,
    utc_now_iso,
)
from src.utils.hashing import stable_hash_text
from src.utils.llm_client import LLMClient


logger = logging.getLogger("run_case")


class _GeneralizedRule(BaseModel):
    trigger_pattern: str
    lesson: str
    evidence_pattern: str | None = None
    argument_pattern: str | None = None
    recommended_action: str
    applicability_scope: str | None = None
    exceptions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)


class MemoryConsolidator:
    """Periodic multi-case consolidation from short-term to long-term memory.

    Verified STM observations are blocked/grouped, compared against existing
    LTM through semantic relations, merged or promoted with independent-support
    counting, conflict tracking, Beta-style confidence recalibration, and
    explicit lifecycle transitions. Nothing is silently deleted."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        config: MemoryConfig | None = None,
        similarity: SimilarityBackend | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.config = config or (store.config if store is not None else MemoryConfig())
        self.store = store or MemoryStore(config=self.config)
        self.similarity = similarity or build_similarity_backend(self.config, llm_client=llm_client)
        self.llm_client = llm_client

    # ---------------------------------------------------------------- staging

    def apply(self, candidates: list[MemoryUpdateCandidate]) -> list[ShortTermMemoryRecord]:
        """Stage verified (or held-for-review) candidates into short-term memory.

        This intentionally no longer appends anything to long-term memory;
        promotion happens only through consolidate()."""
        staged: list[ShortTermMemoryRecord] = []
        for candidate in candidates:
            if candidate.verification_status not in {"verified", "under_review"}:
                continue
            candidate = candidate.model_copy(update={"staged_at": utc_now_iso()})
            staged.append(self.store.stage_candidate(ShortTermMemoryRecord.from_candidate(candidate)))
        return staged

    # ----------------------------------------------------------- consolidation

    def consolidate(self, dry_run: bool = False) -> ConsolidationResult:
        result = ConsolidationResult(dry_run=dry_run, counts_before=self.store.counts())
        events: list[ConsolidationEvent] = []
        now = datetime.now(timezone.utc)

        stm_all = self.store.load_short_term()
        ltm_all = self.store.load_long_term()
        ltm_by_id: dict[str, MemoryRecord] = {record.memory_id: record for record in ltm_all}
        changed_ltm: dict[str, MemoryRecord] = {}
        stm_updates: dict[str, ShortTermMemoryRecord] = {}
        archived_stm: list[ShortTermMemoryRecord] = []

        # A. Select verified, unprocessed STM records; expire stale ones.
        cutoff = now - timedelta(days=self.config.short_term.ttl_days)
        actionable: list[ShortTermMemoryRecord] = []
        for record in stm_all:
            if record.status in {"promoted", "merged", "expired"}:
                continue
            staged_at = _parse_time(record.staged_at) or now
            if staged_at < cutoff:
                expired = record.model_copy(
                    update={"status": "expired", "updated_at": utc_now_iso()}
                )
                stm_updates[record.stm_id] = expired
                archived_stm.append(expired)
                result.expired.append(record.stm_id)
                events.append(
                    ConsolidationEvent(
                        event_id=f"evt_{stable_hash_text('expired' + record.stm_id + utc_now_iso())}",
                        event_type="expired",
                        stm_ids=[record.stm_id],
                        details={"ttl_days": self.config.short_term.ttl_days},
                    )
                )
                continue
            if record.status == "staged" and record.verification_status == "verified":
                actionable.append(record)
            elif record.status == "under_review":
                result.under_review.append(record.stm_id)
        result.stm_considered = len(actionable)
        result.staged = [record.stm_id for record in actionable]

        def current_ltm() -> list[MemoryRecord]:
            merged = dict(ltm_by_id)
            merged.update(changed_ltm)
            return list(merged.values())

        # B/C/D. Compare each observation with existing LTM and handle relations.
        remaining: list[ShortTermMemoryRecord] = []
        for record in actionable:
            relation, target = self._best_ltm_relation(record, current_ltm())
            if relation in {"equivalent", "entails"} and target is not None:
                updated, incremented = self._merge_support(target, record, relation)
                changed_ltm[updated.memory_id] = updated
                stm_updates[record.stm_id] = record.model_copy(
                    update={
                        "status": "merged",
                        "promoted_to_memory_id": updated.memory_id,
                        "updated_at": utc_now_iso(),
                    }
                )
                result.merged.append(record.stm_id)
                if incremented:
                    result.support_increments[updated.memory_id] = (
                        result.support_increments.get(updated.memory_id, 0) + 1
                    )
                    events.append(
                        ConsolidationEvent(
                            event_id=f"evt_{stable_hash_text('merge' + record.stm_id)}",
                            event_type="support_increment",
                            memory_id=updated.memory_id,
                            stm_ids=[record.stm_id],
                            details={"relation": relation, "source_case_id": record.source_case_id},
                        )
                    )
                events.append(
                    ConsolidationEvent(
                        event_id=f"evt_{stable_hash_text('merged' + record.stm_id)}",
                        event_type="merged",
                        memory_id=updated.memory_id,
                        stm_ids=[record.stm_id],
                        details={"relation": relation},
                    )
                )
            elif relation == "contradicts" and target is not None:
                updated, counted = self._apply_conflict(target, record)
                changed_ltm[updated.memory_id] = updated
                stm_updates[record.stm_id] = record.model_copy(
                    update={"status": "under_review", "updated_at": utc_now_iso()}
                )
                result.conflicted.append(record.stm_id)
                if updated.status == "under_review" and target.status != "under_review":
                    result.under_review.append(updated.memory_id)
                    events.append(
                        ConsolidationEvent(
                            event_id=f"evt_{stable_hash_text('ur' + updated.memory_id + record.stm_id)}",
                            event_type="under_review",
                            memory_id=updated.memory_id,
                            stm_ids=[record.stm_id],
                            details={"reason": "conflict_ratio_exceeded"},
                        )
                    )
                events.append(
                    ConsolidationEvent(
                        event_id=f"evt_{stable_hash_text('conflict' + record.stm_id)}",
                        event_type="conflict",
                        memory_id=updated.memory_id,
                        stm_ids=[record.stm_id],
                        details={"counted": counted, "source_case_id": record.source_case_id},
                    )
                )
            else:
                remaining.append(record)

        # E/F. Cluster the rest and promote clusters that meet the thresholds.
        clusters = self._cluster(remaining)
        for cluster in clusters:
            promoted_record = self._try_promote(cluster, ltm_by_id, changed_ltm)
            if promoted_record is not None:
                changed_ltm[promoted_record.memory_id] = promoted_record
                for record in cluster:
                    stm_updates[record.stm_id] = record.model_copy(
                        update={
                            "status": "promoted",
                            "promoted_to_memory_id": promoted_record.memory_id,
                            "updated_at": utc_now_iso(),
                        }
                    )
                    result.promoted.append(record.stm_id)
                events.append(
                    ConsolidationEvent(
                        event_id=f"evt_{stable_hash_text('promote' + promoted_record.memory_id)}",
                        event_type="promoted",
                        memory_id=promoted_record.memory_id,
                        stm_ids=[record.stm_id for record in cluster],
                        details={
                            "memory_type": promoted_record.memory_type,
                            "support_count": promoted_record.support_count,
                            "confidence": promoted_record.confidence,
                        },
                    )
                )
            else:
                # G. Repeated observations that cannot merge into LTM may still
                # generalize into one synthesized semantic rule.
                generalized = self._try_generalize(cluster, events)
                if generalized is not None:
                    changed_ltm[generalized.memory_id] = generalized
                for record in cluster:
                    result.unchanged.append(record.stm_id)

        # I. Lifecycle re-check across affected long-term records.
        for memory_id, record in list(changed_ltm.items()):
            changed_ltm[memory_id] = self._lifecycle_check(record, result, events)
        for record in ltm_all:
            if record.memory_id in changed_ltm:
                continue
            rechecked = self._lifecycle_check(record, result, events)
            if rechecked is not record:
                changed_ltm[record.memory_id] = rechecked

        result.changed_long_term_ids = sorted(changed_ltm.keys())
        result.events = events

        if not dry_run:
            if changed_ltm:
                deprecated_now = [
                    record for record in changed_ltm.values() if record.status == "deprecated"
                ]
                if deprecated_now:
                    self.store.archive_records(deprecated_now, "deprecated_long_term")
                self.store.upsert_long_term(list(changed_ltm.values()))
            if stm_updates:
                if self.config.short_term.archive_expired and archived_stm:
                    self.store.archive_records(archived_stm, "short_term_expired")
                    kept = [
                        row
                        for row in {**{r.stm_id: r for r in stm_all}, **stm_updates}.values()
                        if row.status != "expired"
                    ]
                    self.store.replace_short_term(kept)
                else:
                    self.store.upsert_short_term(list(stm_updates.values()))
            for event in events:
                self.store.append_consolidation_event(event)
            result.counts_after = self.store.counts()
        else:
            result.counts_after = result.counts_before

        result.finished_at = utc_now_iso()
        return result

    # ------------------------------------------------------------- relations

    def _best_ltm_relation(
        self,
        record: ShortTermMemoryRecord,
        ltm: list[MemoryRecord],
    ) -> tuple[str, MemoryRecord | None]:
        candidates = [
            item
            for item in ltm
            if item.memory_type == record.memory_type
            and item.status in {"active", "under_review"}
        ]
        if not candidates:
            return "unrelated", None
        items = [item.model_dump() for item in candidates]
        by_id = {item.memory_id: item for item in candidates}
        record_item = record.model_dump()
        if hasattr(self.similarity, "shortlist"):
            shortlisted = [
                item
                for _, item in self.similarity.shortlist(  # type: ignore[attr-defined]
                    record.text, items, k=self.config.similarity.lexical_shortlist_k
                )
            ]
        else:
            shortlisted = items
        for item in shortlisted:
            relation = self.similarity.relation(record_item, item)
            if relation != "unrelated":
                return relation, by_id.get(item.get("memory_id", ""))
        return "unrelated", None

    # ---------------------------------------------------------------- merging

    def _merge_support(
        self,
        target: MemoryRecord,
        record: ShortTermMemoryRecord,
        relation: str,
    ) -> tuple[MemoryRecord, bool]:
        fingerprint = record.source_fingerprint or f"fp_case_{record.source_case_id}"
        new_case = record.source_case_id not in target.source_case_ids
        new_fingerprint = fingerprint not in target.source_fingerprints
        # Idempotency + independence: the same case or the same source
        # fingerprint (near-duplicate dataset rows) never counts twice.
        if not (new_case and new_fingerprint):
            return target, False

        alpha, beta = self._prior(target)
        alpha += record.confidence
        beta += 1.0 - record.confidence
        confidence = alpha / (alpha + beta)

        text = target.text
        metadata = dict(target.metadata)
        if relation == "entails":
            # Prefer the more precise statement; keep the broader one in metadata.
            target_tokens = content_tokens(target.text)
            record_tokens = content_tokens(record.text)
            if target_tokens < record_tokens:
                broader = metadata.setdefault("broader_variants", [])
                if target.text not in broader:
                    broader.append(target.text)
                text = record.text
            else:
                broader = metadata.setdefault("broader_variants", [])
                if record.text not in broader:
                    broader.append(record.text)
        metadata["alpha"] = alpha
        metadata["beta"] = beta

        return (
            target.model_copy(
                update={
                    "text": text,
                    "version": target.version + 1,
                    "confidence": confidence,
                    "support_count": target.support_count + 1,
                    "support_weight": target.support_weight + record.confidence,
                    "source_case_ids": target.source_case_ids + [record.source_case_id],
                    "source_fingerprints": target.source_fingerprints + [fingerprint],
                    "source_datasets": _append_unique(target.source_datasets, record.dataset_name),
                    "source_splits": _append_unique(target.source_splits, record.dataset_split),
                    "last_confirmed_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                    "metadata": metadata,
                }
            ),
            True,
        )

    def _apply_conflict(
        self,
        target: MemoryRecord,
        record: ShortTermMemoryRecord,
    ) -> tuple[MemoryRecord, bool]:
        fingerprint = record.source_fingerprint or f"fp_case_{record.source_case_id}"
        metadata = dict(target.metadata)
        conflict_fingerprints = list(metadata.get("conflict_fingerprints", []))
        if fingerprint in conflict_fingerprints:
            return target, False

        alpha, beta = self._prior(target)
        alpha += 1.0 - record.confidence
        beta += record.confidence
        confidence = alpha / (alpha + beta)
        conflict_fingerprints.append(fingerprint)
        metadata["alpha"] = alpha
        metadata["beta"] = beta
        metadata["conflict_fingerprints"] = conflict_fingerprints

        conflict_count = target.conflict_count + 1
        status = target.status
        ratio = conflict_count / max(1, target.support_count + conflict_count)
        if ratio > self.config.consolidation.under_review_conflict_ratio and status == "active":
            status = "under_review"

        return (
            target.model_copy(
                update={
                    "version": target.version + 1,
                    "confidence": confidence,
                    "conflict_count": conflict_count,
                    "conflict_weight": target.conflict_weight + record.confidence,
                    "status": status,
                    "updated_at": utc_now_iso(),
                    "metadata": metadata,
                }
            ),
            True,
        )

    @staticmethod
    def _prior(record: MemoryRecord) -> tuple[float, float]:
        metadata = record.metadata or {}
        if "alpha" in metadata and "beta" in metadata:
            return float(metadata["alpha"]), float(metadata["beta"])
        strength = max(1, record.support_count)
        return record.confidence * strength, (1.0 - record.confidence) * strength

    # -------------------------------------------------------------- clustering

    def _cluster(self, records: list[ShortTermMemoryRecord]) -> list[list[ShortTermMemoryRecord]]:
        blocks: dict[str, list[ShortTermMemoryRecord]] = {}
        for record in records:
            signature = record.semantic_signature or semantic_signature(
                record.memory_type,
                record.claim_type,
                record.task_type,
                record.failure_type,
                record.polarity,
                record.applicability_scope,
            )
            blocks.setdefault(signature, []).append(record)

        clusters: list[list[ShortTermMemoryRecord]] = []
        for block in blocks.values():
            unassigned = list(block)
            while unassigned:
                seed = unassigned.pop(0)
                cluster = [seed]
                still_unassigned = []
                for other in unassigned:
                    relation = self.similarity.relation(seed.model_dump(), other.model_dump())
                    if relation in {"equivalent", "entails"}:
                        cluster.append(other)
                    else:
                        still_unassigned.append(other)
                unassigned = still_unassigned
                clusters.append(cluster)
        return clusters

    # --------------------------------------------------------------- promotion

    def _try_promote(
        self,
        cluster: list[ShortTermMemoryRecord],
        ltm_by_id: dict[str, MemoryRecord],
        changed_ltm: dict[str, MemoryRecord],
    ) -> MemoryRecord | None:
        representative = max(cluster, key=lambda record: record.confidence)
        thresholds = self.config.consolidation.thresholds_for(representative.memory_type)

        unique: dict[str, ShortTermMemoryRecord] = {}
        cases: set[str] = set()
        for record in cluster:
            fingerprint = record.source_fingerprint or f"fp_case_{record.source_case_id}"
            if fingerprint in unique or record.source_case_id in cases:
                continue
            unique[fingerprint] = record
            cases.add(record.source_case_id)
        distinct_sources = len(unique)

        alpha = sum(record.confidence for record in unique.values())
        beta = sum(1.0 - record.confidence for record in unique.values())
        confidence = alpha / (alpha + beta) if alpha + beta > 0 else 0.0

        human_reviewed = any(
            record.supervision_source == "human_feedback" for record in unique.values()
        )
        min_cases = thresholds.min_distinct_cases
        if human_reviewed and representative.memory_type == "episodic":
            # One human-reviewed case may yield a strong episodic record, but a
            # single case never becomes a universal semantic rule.
            min_cases = 1

        if distinct_sources < min_cases or confidence < thresholds.min_confidence:
            return None
        if not representative.grounding_evidence_ids and not representative.grounding_argument_ids:
            return None

        memory_id = f"mem_{stable_hash_text(representative.canonical_key or representative.text)}"
        if memory_id in ltm_by_id or memory_id in changed_ltm:
            # A record with the same canonical identity already exists; this run
            # should have merged into it, so do not create a duplicate.
            return None

        now = utc_now_iso()
        return MemoryRecord(
            memory_id=memory_id,
            memory_type=representative.memory_type,
            memory_level="long_term",
            case_id=representative.source_case_id,
            claim_type=representative.claim_type,
            task_type=representative.task_type,
            text=representative.text,
            trigger_pattern=representative.trigger_pattern,
            lesson=representative.lesson,
            evidence_pattern=representative.evidence_pattern,
            argument_pattern=representative.argument_pattern,
            recommended_action=representative.recommended_action,
            failure_type=representative.failure_type,
            canonical_key=representative.canonical_key
            or canonical_key(
                representative.memory_type,
                representative.claim_type,
                representative.task_type,
                representative.text,
            ),
            semantic_signature=representative.semantic_signature,
            applicability_scope=representative.applicability_scope,
            exceptions=list(representative.exceptions),
            polarity=representative.polarity,
            source_case_ids=sorted(record.source_case_id for record in unique.values()),
            source_fingerprints=sorted(unique.keys()),
            source_datasets=sorted(
                {record.dataset_name for record in unique.values() if record.dataset_name}
            ),
            source_splits=sorted(
                {record.dataset_split for record in unique.values() if record.dataset_split}
            ),
            tags=[representative.memory_type, representative.claim_type or "general"],
            confidence=confidence,
            support_count=distinct_sources,
            support_weight=alpha,
            conflict_count=0,
            status="active",
            origin="consolidated",
            verified_by="memory_consolidator",
            created_at=now,
            updated_at=now,
            last_confirmed_at=now,
            metadata={
                "alpha": alpha,
                "beta": beta,
                "stm_ids": [record.stm_id for record in cluster],
                "supervision_sources": sorted(
                    {record.supervision_source for record in unique.values()}
                ),
            },
        )

    # ------------------------------------------------------------ generalization

    def _try_generalize(
        self,
        cluster: list[ShortTermMemoryRecord],
        events: list[ConsolidationEvent],
    ) -> MemoryRecord | None:
        if not self.config.consolidation.generalize_repeated_episodes:
            return None
        if self.llm_client is None:
            return None
        if cluster[0].memory_type not in {"episodic", "failure"}:
            return None
        fingerprints = {
            record.source_fingerprint or f"fp_case_{record.source_case_id}" for record in cluster
        }
        required = max(
            self.config.consolidation.min_distinct_sources,
            self.config.consolidation.semantic_rule.min_distinct_cases,
        )
        if len(fingerprints) < required:
            return None

        observations = [
            {"case_id": record.source_case_id, "text": record.text} for record in cluster
        ]
        prompt = (
            "These repeated observations from independent multimedia verification cases "
            "form one cluster. Synthesize ONE generalized rule as JSON: "
            '{"trigger_pattern": "...", "lesson": "...", "evidence_pattern": "...", '
            '"argument_pattern": "...", "recommended_action": "...", '
            '"applicability_scope": "...", "exceptions": [...], "confidence": 0.0-1.0}\n'
            f"Observations: {observations}"
        )
        try:
            rule = _GeneralizedRule.model_validate(self.llm_client.generate_json(prompt))
        except Exception as exc:
            logger.warning("Rule generalization failed: %s", exc)
            events.append(
                ConsolidationEvent(
                    event_id=f"evt_{stable_hash_text('genfail' + cluster[0].stm_id)}",
                    event_type="generalization_failed",
                    stm_ids=[record.stm_id for record in cluster],
                    details={"reason": "llm_generalization_unavailable"},
                )
            )
            return None

        verified = self._verify_generalization(rule, observations)
        status = "active" if verified else "under_review"
        memory_id = f"mem_{stable_hash_text('generalized' + normalize_text(rule.lesson))}"
        now = utc_now_iso()
        record = MemoryRecord(
            memory_id=memory_id,
            memory_type="semantic_rule",
            memory_level="long_term",
            claim_type=cluster[0].claim_type or "general",
            task_type=cluster[0].task_type,
            text=rule.lesson,
            trigger_pattern=rule.trigger_pattern,
            lesson=rule.lesson,
            evidence_pattern=rule.evidence_pattern,
            argument_pattern=rule.argument_pattern,
            recommended_action=rule.recommended_action,
            canonical_key=canonical_key(
                "semantic_rule", cluster[0].claim_type or "general", cluster[0].task_type, rule.lesson
            ),
            applicability_scope=rule.applicability_scope,
            exceptions=list(rule.exceptions),
            source_case_ids=sorted({record.source_case_id for record in cluster}),
            source_fingerprints=sorted(fingerprints),
            source_datasets=sorted({r.dataset_name for r in cluster if r.dataset_name}),
            source_splits=sorted({r.dataset_split for r in cluster if r.dataset_split}),
            tags=["semantic_rule", "generalized"],
            confidence=min(rule.confidence, 0.9),
            support_count=len(fingerprints),
            status=status,
            origin="consolidated",
            verified_by="memory_consolidator_generalization" if verified else None,
            created_at=now,
            updated_at=now,
            metadata={
                "generalized_from_stm_ids": [record.stm_id for record in cluster],
                "alpha": sum(r.confidence for r in cluster),
                "beta": sum(1.0 - r.confidence for r in cluster),
            },
        )
        events.append(
            ConsolidationEvent(
                event_id=f"evt_{stable_hash_text('generalized' + memory_id)}",
                event_type="generalized",
                memory_id=memory_id,
                stm_ids=[record.stm_id for record in cluster],
                details={"verified_against_sources": verified, "status": status},
            )
        )
        return record

    def _verify_generalization(self, rule: _GeneralizedRule, observations: list[dict]) -> bool:
        prompt = (
            "Check whether this generalized rule is consistent with EVERY source observation. "
            'Return JSON {"consistent_with_all": true/false, "reason": "..."}\n'
            f"Rule: {rule.model_dump()}\nObservations: {observations}"
        )
        try:
            data = self.llm_client.generate_json(prompt)  # type: ignore[union-attr]
            return bool(data.get("consistent_with_all", False))
        except Exception:
            return False

    # -------------------------------------------------------------- lifecycle

    def _lifecycle_check(
        self,
        record: MemoryRecord,
        result: ConsolidationResult,
        events: list[ConsolidationEvent],
    ) -> MemoryRecord:
        if record.status in {"deprecated", "merged", "promoted", "expired"}:
            return record
        ratio = record.conflict_count / max(1, record.support_count + record.conflict_count)
        if (
            record.confidence < self.config.consolidation.deprecate_confidence_below
            and record.conflict_count > 0
        ):
            # Deprecation requires contradiction pressure, never age alone.
            deprecated = record.model_copy(
                update={"status": "deprecated", "updated_at": utc_now_iso()}
            )
            result.deprecated.append(record.memory_id)
            events.append(
                ConsolidationEvent(
                    event_id=f"evt_{stable_hash_text('deprecate' + record.memory_id)}",
                    event_type="deprecated",
                    memory_id=record.memory_id,
                    details={"confidence": record.confidence, "conflict_ratio": ratio},
                )
            )
            return deprecated
        if record.status == "active" and ratio > self.config.consolidation.under_review_conflict_ratio:
            under_review = record.model_copy(
                update={"status": "under_review", "updated_at": utc_now_iso()}
            )
            if record.memory_id not in result.under_review:
                result.under_review.append(record.memory_id)
            events.append(
                ConsolidationEvent(
                    event_id=f"evt_{stable_hash_text('review' + record.memory_id)}",
                    event_type="under_review",
                    memory_id=record.memory_id,
                    details={"conflict_ratio": ratio},
                )
            )
            return under_review
        if (
            record.status == "under_review"
            and ratio <= self.config.consolidation.max_conflict_ratio
            and record.confidence >= self.config.consolidation.deprecate_confidence_below
        ):
            # Enough independent support accumulated to make the conflict ratio
            # acceptable again: the record returns to active (golden) status.
            reactivated = record.model_copy(
                update={"status": "active", "updated_at": utc_now_iso()}
            )
            events.append(
                ConsolidationEvent(
                    event_id=f"evt_{stable_hash_text('reactivate' + record.memory_id)}",
                    event_type="promoted",
                    memory_id=record.memory_id,
                    details={"reason": "conflict_ratio_recovered", "conflict_ratio": ratio},
                )
            )
            return reactivated
        return record


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _append_unique(values: list[str], value: str | None) -> list[str]:
    if value is None or value in values:
        return list(values)
    return list(values) + [value]
