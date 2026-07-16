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

    def consolidate(
        self, dry_run: bool = False, retry_under_review: bool = False
    ) -> ConsolidationResult:
        with self.store.transaction():
            return self._consolidate_locked(
                dry_run=dry_run, retry_under_review=retry_under_review
            )

    def _consolidate_locked(
        self, dry_run: bool = False, retry_under_review: bool = False
    ) -> ConsolidationResult:
        result = ConsolidationResult(dry_run=dry_run, counts_before=self.store.counts())
        events: list[ConsolidationEvent] = []
        now = datetime.now(timezone.utc)

        stm_all = self.store.load_short_term()
        ltm_all = self.store.load_long_term()
        ltm_by_id: dict[str, MemoryRecord] = {record.memory_id: record for record in ltm_all}
        changed_ltm: dict[str, MemoryRecord] = {}
        stm_updates: dict[str, ShortTermMemoryRecord] = {}
        archived_stm: list[ShortTermMemoryRecord] = []

        usage_updates, usage_events = self._rollup_usage(ltm_by_id)
        changed_ltm.update(usage_updates)
        events.extend(usage_events)

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
                        event_id=f"evt_{stable_hash_text('expired' + record.stm_id)}",
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
            if relation in {"equivalent", "entails", "a_entails_b", "b_entails_a"} and target is not None:
                updated, incremented = self._merge_support(target, record, relation)
                if incremented:
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
                if counted:
                    changed_ltm[updated.memory_id] = updated
                stm_updates[record.stm_id] = record.model_copy(
                    update={"status": "under_review", "promoted_to_memory_id": updated.memory_id, "updated_at": utc_now_iso()}
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
                generalized = self._try_generalize(cluster, current_ltm(), events)
                if generalized is None:
                    for record in cluster:
                        result.unchanged.append(record.stm_id)
                    continue
                generalized_record, verified, existed, changed, support_added, recovered = generalized
                if changed:
                    changed_ltm[generalized_record.memory_id] = generalized_record
                if support_added and existed:
                    result.support_increments[generalized_record.memory_id] = support_added
                for record in cluster:
                    if verified:
                        terminal = "merged" if existed else "promoted"
                        stm_updates[record.stm_id] = record.model_copy(update={
                            "status": terminal,
                            "promoted_to_memory_id": generalized_record.memory_id,
                            "updated_at": utc_now_iso(),
                        })
                        getattr(result, terminal).append(record.stm_id)
                    else:
                        stm_updates[record.stm_id] = record.model_copy(update={
                            "status": "under_review",
                            "promoted_to_memory_id": generalized_record.memory_id,
                            "updated_at": utc_now_iso(),
                        })
                        result.under_review.append(record.stm_id)
                if recovered:
                    linked_ids = set(generalized_record.metadata.get("generalized_from_stm_ids", []))
                    for linked in stm_all:
                        if linked.stm_id not in linked_ids or linked.status != "under_review":
                            continue
                        stm_updates[linked.stm_id] = linked.model_copy(update={
                            "status": "merged",
                            "promoted_to_memory_id": generalized_record.memory_id,
                            "updated_at": utc_now_iso(),
                        })
                        if linked.stm_id not in result.merged:
                            result.merged.append(linked.stm_id)

        # H. Explicit retry for generalized proposals whose LLM-dependent
        # attempt could not complete (or whose low confidence is being retried
        # deliberately). Conflict-driven under-review memories are excluded.
        if retry_under_review:
            attempted_this_run = {
                event.memory_id for event in events
                if event.event_type in {"generalization_failed", "generalization_recovered"}
            }
            retryable_reasons = {
                "generalization_synthesis_unavailable",
                "generalization_synthesis_failed",
                "generalization_verification_unavailable",
                "generalization_confidence_below_threshold",
            }
            for proposal in current_ltm():
                metadata = proposal.metadata or {}
                review_reason = metadata.get("generalization_review_reason")
                if (
                    proposal.memory_id in attempted_this_run
                    or proposal.memory_type != "semantic_rule"
                    or proposal.status != "under_review"
                    or not metadata.get("proposal_only", False)
                    or not metadata.get("generalized_from_stm_ids")
                    or review_reason not in retryable_reasons
                ):
                    continue
                retried, recovered = self._reconsider_generalization(
                    proposal,
                    events,
                    force_synthesis=(
                        review_reason
                        == "generalization_confidence_below_threshold"
                    ),
                )
                changed_ltm[retried.memory_id] = retried
                if recovered:
                    linked_ids = set(
                        retried.metadata.get("generalized_from_stm_ids", [])
                    )
                    for linked in stm_all:
                        if (
                            linked.stm_id not in linked_ids
                            or linked.status != "under_review"
                        ):
                            continue
                        stm_updates[linked.stm_id] = linked.model_copy(
                            update={
                                "status": "merged",
                                "promoted_to_memory_id": retried.memory_id,
                                "updated_at": utc_now_iso(),
                            }
                        )
                        if linked.stm_id not in result.merged:
                            result.merged.append(linked.stm_id)
                elif retried.memory_id not in result.under_review:
                    result.under_review.append(retried.memory_id)


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
        result.state_hash = self.store.state_hash()
        return result

    # ------------------------------------------------------------- relations

    def _best_ltm_relation(
        self, record: ShortTermMemoryRecord, ltm: list[MemoryRecord]
    ) -> tuple[str, MemoryRecord | None]:
        if record.semantic_relation == "contradicts" and record.related_memory_id:
            target = next(
                (item for item in ltm if item.memory_id == record.related_memory_id and item.status in {"active", "under_review"}),
                None,
            )
            if target is not None:
                return "contradicts", target
        candidates = [
            item for item in ltm
            if item.memory_type == record.memory_type and item.status in {"active", "under_review"}
        ]
        if not candidates:
            return "unrelated", None
        items = [item.model_dump() for item in candidates]
        if hasattr(self.similarity, "shortlist"):
            shortlisted = [
                item for _, item in self.similarity.shortlist(
                    record.text, items, k=self.config.similarity.lexical_shortlist_k
                )
            ]
        else:
            shortlisted = items
        pairs = [(record.model_dump(), item) for item in shortlisted]
        relations = self.similarity.relations_batch(pairs) if hasattr(self.similarity, "relations_batch") else [
            self.similarity.relation(a, b) for a, b in pairs
        ]
        by_id = {item.memory_id: item for item in candidates}
        priority = {"equivalent": 4, "contradicts": 3, "a_entails_b": 2, "b_entails_a": 2, "entails": 2}
        ranked = [
            (priority.get(relation, 0), self.similarity.similarity(record.text, item.get("text") or ""), relation, item)
            for relation, item in zip(relations, shortlisted)
            if relation != "unrelated"
        ]
        if not ranked:
            return "unrelated", None
        _, _, relation, item = max(ranked, key=lambda value: (value[0], value[1]))
        return relation, by_id.get(item.get("memory_id", ""))

    # ---------------------------------------------------------------- merging

    def _merge_support(
        self,
        target: MemoryRecord,
        record: ShortTermMemoryRecord,
        relation: str,
    ) -> tuple[MemoryRecord, bool]:
        accepted = _select_independent_observations(
            [record], target.source_case_ids, target.source_fingerprints
        )
        if not accepted:
            return target, False
        record = accepted[0]
        fingerprint = _source_fingerprint(record)
        # Idempotency + independence: both identity dimensions must be new.

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

    # Safety invariant: this path only updates conflict evidence on the target;
    # it never promotes the contradictory observation as a golden rule.
    def _apply_conflict(
        self,
        target: MemoryRecord,
        record: ShortTermMemoryRecord,
    ) -> tuple[MemoryRecord, bool]:
        fingerprint = record.source_fingerprint or f"fp_case_{record.source_case_id}"
        metadata = dict(target.metadata)
        conflict_fingerprints = list(metadata.get("conflict_fingerprints", []))
        conflict_case_ids = list(metadata.get("conflict_case_ids", []))
        if fingerprint in conflict_fingerprints or record.source_case_id in conflict_case_ids:
            return target, False

        alpha, beta = self._prior(target)
        alpha += 1.0 - record.confidence
        beta += record.confidence
        confidence = alpha / (alpha + beta)
        conflict_fingerprints.append(fingerprint)
        conflict_case_ids.append(record.source_case_id)
        metadata["alpha"] = alpha
        metadata["beta"] = beta
        metadata["conflict_fingerprints"] = conflict_fingerprints
        metadata["conflict_case_ids"] = conflict_case_ids

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
                    if relation in {"equivalent", "entails", "a_entails_b", "b_entails_a"}:
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

        independent = _select_independent_observations(cluster)
        distinct_sources = len(independent)
        distinct_cases = len(independent)

        alpha = sum(record.confidence for record in independent)
        beta = sum(1.0 - record.confidence for record in independent)
        confidence = alpha / (alpha + beta) if alpha + beta > 0 else 0.0

        human_reviewed = any(
            record.supervision_source == "human_feedback" for record in independent
        )
        min_cases = thresholds.min_distinct_cases
        if human_reviewed and representative.memory_type == "episodic":
            # One human-reviewed case may yield a strong episodic record, but a
            # single case never becomes a universal semantic rule.
            min_cases = 1

        if (
            distinct_cases < min_cases
            or distinct_sources < thresholds.min_distinct_sources
            or confidence < thresholds.min_confidence
        ):
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
            source_case_ids=sorted(record.source_case_id for record in independent),
            source_fingerprints=sorted(_source_fingerprint(record) for record in independent),
            source_datasets=sorted(
                {record.dataset_name for record in independent if record.dataset_name}
            ),
            source_splits=sorted(
                {record.dataset_split for record in independent if record.dataset_split}
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
                    {record.supervision_source for record in independent}
                ),
            },
        )

    # ------------------------------------------------------------ generalization

    def _try_generalize(
        self,
        cluster: list[ShortTermMemoryRecord],
        existing_ltm: list[MemoryRecord],
        events: list[ConsolidationEvent],
    ) -> tuple[MemoryRecord, bool, bool, bool, int, bool] | None:
        if not self.config.consolidation.generalize_repeated_episodes or not cluster:
            return None
        if cluster[0].memory_type not in {"episodic", "failure"}:
            return None

        independent = _select_independent_observations(cluster)
        source_signature = cluster[0].semantic_signature or semantic_signature(
            cluster[0].memory_type,
            cluster[0].claim_type,
            cluster[0].task_type,
            cluster[0].failure_type,
            cluster[0].polarity,
            cluster[0].applicability_scope,
        )
        existing_source_rule = self._find_source_generalization(
            independent, existing_ltm, source_signature, cluster[0].memory_type
        )
        stm_ids = sorted(row.stm_id for row in cluster)

        if existing_source_rule is not None:
            already_active = (
                existing_source_rule.status == "active"
                and not existing_source_rule.metadata.get("proposal_only", False)
            )
            updated, added = self._merge_generalized_support(
                existing_source_rule, independent
            )
            changed = updated != existing_source_rule
            if added:
                events.append(
                    ConsolidationEvent(
                        event_id=f"evt_{stable_hash_text('generalized_support' + existing_source_rule.memory_id + '|'.join(stm_ids))}",
                        event_type="support_increment",
                        memory_id=existing_source_rule.memory_id,
                        stm_ids=stm_ids,
                        details={"generalized": True, "new_support": added},
                    )
                )

            recovered = False
            if not already_active and updated.metadata.get("proposal_only", False) and added:
                updated, recovered = self._reconsider_generalization(updated, events)
                changed = updated != existing_source_rule
            verified = already_active or recovered
            return updated, verified, True, changed, added, recovered

        thresholds = self.config.consolidation.semantic_rule
        if (
            len(independent) < thresholds.min_distinct_cases
            or len(independent) < thresholds.min_distinct_sources
        ):
            return None

        observations = [_source_observation(row) for row in independent]
        rule, synthesized, synthesis_reason = self._synthesize_generalization(observations)
        verification_succeeded = False
        verification_reason = "generalization_verification_not_attempted"
        if synthesized:
            verification_succeeded, verification_reason = self._verify_generalization(
                rule, observations
            )
        review_reason = self._generalization_review_reason(
            rule,
            synthesized=synthesized,
            synthesis_reason=synthesis_reason,
            verification_succeeded=verification_succeeded,
            verification_reason=verification_reason,
            observations=observations,
        )
        verified = review_reason is None

        claim_type = cluster[0].claim_type or "general"
        rule_key = canonical_key(
            "semantic_rule", claim_type, cluster[0].task_type, rule.lesson
        )
        now = utc_now_iso()
        alpha = sum(row.confidence for row in independent)
        beta = sum(1.0 - row.confidence for row in independent)
        attempt = {
            "attempt": 1,
            "support": len(independent),
            "synthesis_succeeded": synthesized,
            "verification_succeeded": verification_succeeded,
            "reason": review_reason,
            "rule_confidence": rule.confidence,
        }
        proposed = MemoryRecord(
            memory_id=f"mem_{stable_hash_text(rule_key)}",
            memory_type="semantic_rule",
            memory_level="long_term",
            claim_type=claim_type,
            task_type=cluster[0].task_type,
            text=rule.lesson,
            trigger_pattern=rule.trigger_pattern,
            lesson=rule.lesson,
            evidence_pattern=rule.evidence_pattern,
            argument_pattern=rule.argument_pattern,
            recommended_action=rule.recommended_action,
            canonical_key=rule_key,
            semantic_signature=semantic_signature(
                "semantic_rule",
                claim_type,
                cluster[0].task_type,
                None,
                None,
                rule.applicability_scope,
            ),
            applicability_scope=rule.applicability_scope,
            exceptions=list(rule.exceptions),
            source_case_ids=sorted(row.source_case_id for row in independent),
            source_fingerprints=sorted(
                _source_fingerprint(row) for row in independent
            ),
            source_datasets=sorted(
                {row.dataset_name for row in independent if row.dataset_name}
            ),
            source_splits=sorted(
                {row.dataset_split for row in independent if row.dataset_split}
            ),
            tags=["semantic_rule", "generalized"],
            # This is the synthesized rule confidence. It is never raised or
            # clamped to the configured semantic threshold.
            confidence=rule.confidence,
            support_count=len(independent),
            support_weight=alpha,
            status="active" if verified else "under_review",
            origin="consolidated",
            verified_by=(
                "memory_consolidator_generalization" if verified else None
            ),
            proposal_only=not verified,
            generalization_verified=verified,
            created_at=now,
            updated_at=now,
            last_confirmed_at=now,
            metadata={
                "generalized_from_stm_ids": stm_ids,
                "alpha": alpha,
                "beta": beta,
                "support_confidence": alpha / (alpha + beta) if alpha + beta else 0.0,
                "generalization_verified": verified,
                "proposal_only": not verified,
                "generalization_review_reason": review_reason,
                "generalization_attempt_count": 1,
                "last_generalization_attempt_support": len(independent),
                "generalization_used_fallback": not synthesized,
                "generalization_attempts": [attempt],
                "source_memory_type": cluster[0].memory_type,
                "source_semantic_signature": source_signature,
                "source_observations": observations,
                "source_observation_texts": sorted(
                    {normalize_text(row.text) for row in independent}
                ),
            },
        )

        target = self._find_equivalent_semantic_rule(proposed, existing_ltm)
        if target is None:
            event_type = "generalized" if verified else "generalization_failed"
            events.append(
                ConsolidationEvent(
                    event_id=f"evt_{stable_hash_text(event_type + proposed.memory_id + '|'.join(stm_ids))}",
                    event_type=event_type,
                    memory_id=proposed.memory_id,
                    stm_ids=stm_ids,
                    details={
                        "verified_against_sources": verification_succeeded,
                        "status": proposed.status,
                        "proposal_created": True,
                        "reason": review_reason,
                    },
                )
            )
            return proposed, verified, False, True, len(independent), False

        updated, added = self._merge_generalized_support(target, independent)
        target_active = (
            target.status == "active"
            and not target.metadata.get("proposal_only", False)
        )
        recovered = False
        if verified and target.metadata.get("proposal_only", False):
            updated = self._activate_generalization(updated, rule)
            recovered = True
            events.append(self._recovery_event(updated))
        if added:
            events.append(
                ConsolidationEvent(
                    event_id=f"evt_{stable_hash_text('generalized_support' + target.memory_id + '|'.join(stm_ids))}",
                    event_type="support_increment",
                    memory_id=target.memory_id,
                    stm_ids=stm_ids,
                    details={"generalized": True, "new_support": added},
                )
            )
        return (
            updated,
            target_active or recovered,
            True,
            updated != target,
            added,
            recovered,
        )

    def _find_source_generalization(
        self,
        rows: list[ShortTermMemoryRecord],
        existing_ltm: list[MemoryRecord],
        source_signature: str,
        source_memory_type: str,
    ) -> MemoryRecord | None:
        for candidate_rule in existing_ltm:
            metadata = candidate_rule.metadata or {}
            if (
                candidate_rule.memory_type != "semantic_rule"
                or candidate_rule.status not in {"active", "under_review"}
                or metadata.get("source_memory_type") != source_memory_type
                or metadata.get("source_semantic_signature") != source_signature
            ):
                continue
            prototypes = metadata.get("source_observation_texts", [])
            if any(
                normalize_text(row.text) == normalize_text(prototype)
                for row in rows
                for prototype in prototypes
            ):
                return candidate_rule
            for row in rows:
                for prototype in prototypes:
                    prototype_item = {
                        **row.model_dump(),
                        "text": prototype,
                        "canonical_key": None,
                    }
                    if self.similarity.relation(
                        row.model_dump(), prototype_item
                    ) in {
                        "equivalent",
                        "a_entails_b",
                        "b_entails_a",
                        "entails",
                    }:
                        return candidate_rule
        return None

    def _find_equivalent_semantic_rule(
        self, proposed: MemoryRecord, existing_ltm: list[MemoryRecord]
    ) -> MemoryRecord | None:
        target = next(
            (
                row
                for row in existing_ltm
                if row.memory_type == "semantic_rule"
                and row.status in {"active", "under_review"}
                and (row.canonical_key or "") == proposed.canonical_key
            ),
            None,
        )
        if target is not None:
            return target
        candidates = [
            row
            for row in existing_ltm
            if row.memory_type == "semantic_rule"
            and row.status in {"active", "under_review"}
        ]
        relations = (
            self.similarity.relations_batch(
                [(proposed.model_dump(), row.model_dump()) for row in candidates]
            )
            if candidates and hasattr(self.similarity, "relations_batch")
            else [
                self.similarity.relation(proposed.model_dump(), row.model_dump())
                for row in candidates
            ]
        )
        related = [
            (self.similarity.similarity(proposed.text, row.text), row)
            for row, relation in zip(candidates, relations)
            if relation
            in {"equivalent", "a_entails_b", "b_entails_a", "entails"}
        ]
        return max(related, key=lambda pair: pair[0])[1] if related else None

    def _merge_generalized_support(
        self, target: MemoryRecord, rows: list[ShortTermMemoryRecord]
    ) -> tuple[MemoryRecord, int]:
        unseen = _select_independent_observations(
            rows, target.source_case_ids, target.source_fingerprints
        )
        if not unseen:
            return target, 0

        alpha, beta = self._prior(target)
        for row in unseen:
            alpha += row.confidence
            beta += 1.0 - row.confidence
        metadata = dict(target.metadata)
        metadata.update(
            {
                "alpha": alpha,
                "beta": beta,
                "support_confidence": (
                    alpha / (alpha + beta) if alpha + beta else 0.0
                ),
            }
        )
        metadata["source_observation_texts"] = sorted(
            set(metadata.get("source_observation_texts", []))
            | {normalize_text(row.text) for row in unseen}
        )
        metadata["generalized_from_stm_ids"] = sorted(
            set(metadata.get("generalized_from_stm_ids", []))
            | {row.stm_id for row in unseen}
        )
        stored_observations = self._proposal_source_observations(target)
        metadata["source_observations"] = _select_independent_observations(
            stored_observations + [_source_observation(row) for row in unseen]
        )
        return (
            target.model_copy(
                update={
                    "version": target.version + 1,
                    # Preserve synthesized semantic confidence; alpha/beta track
                    # independently grounded support in structured metadata.
                    "support_count": target.support_count + len(unseen),
                    "support_weight": target.support_weight
                    + sum(row.confidence for row in unseen),
                    "source_case_ids": sorted(
                        set(target.source_case_ids)
                        | {row.source_case_id for row in unseen}
                    ),
                    "source_fingerprints": sorted(
                        set(target.source_fingerprints)
                        | {_source_fingerprint(row) for row in unseen}
                    ),
                    "source_datasets": sorted(
                        set(target.source_datasets)
                        | {
                            row.dataset_name
                            for row in unseen
                            if row.dataset_name
                        }
                    ),
                    "source_splits": sorted(
                        set(target.source_splits)
                        | {
                            row.dataset_split
                            for row in unseen
                            if row.dataset_split
                        }
                    ),
                    "last_confirmed_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                    "metadata": metadata,
                }
            ),
            len(unseen),
        )

    def _reconsider_generalization(
        self,
        proposal: MemoryRecord,
        events: list[ConsolidationEvent],
        force_synthesis: bool = False,
    ) -> tuple[MemoryRecord, bool]:
        metadata = dict(proposal.metadata)
        observations = self._proposal_source_observations(proposal)
        observations = _select_independent_observations(observations)
        previous_reason = metadata.get("generalization_review_reason")
        rerun_synthesis = force_synthesis or bool(metadata.get("generalization_used_fallback")) or (
            previous_reason
            in {
                "generalization_synthesis_failed",
                "generalization_synthesis_unavailable",
            }
        )
        if rerun_synthesis:
            rule, synthesized, synthesis_reason = self._synthesize_generalization(
                observations
            )
        else:
            rule = _rule_from_record(proposal)
            synthesized = True
            synthesis_reason = None

        verification_succeeded = False
        verification_reason = "generalization_verification_not_attempted"
        if synthesized:
            verification_succeeded, verification_reason = self._verify_generalization(
                rule, observations
            )
        review_reason = self._generalization_review_reason(
            rule,
            synthesized=synthesized,
            synthesis_reason=synthesis_reason,
            verification_succeeded=verification_succeeded,
            verification_reason=verification_reason,
            observations=observations,
        )
        attempt_count = int(metadata.get("generalization_attempt_count", 1)) + 1
        attempts = list(metadata.get("generalization_attempts", []))
        attempts.append(
            {
                "attempt": attempt_count,
                "support": len(observations),
                "synthesis_succeeded": synthesized,
                "verification_succeeded": verification_succeeded,
                "reason": review_reason,
                "rule_confidence": rule.confidence,
            }
        )
        metadata.update(
            {
                "generalization_attempt_count": attempt_count,
                "last_generalization_attempt_support": len(observations),
                "generalization_review_reason": review_reason,
                "generalization_verified": review_reason is None,
                "proposal_only": review_reason is not None,
                "generalization_used_fallback": not synthesized,
                "generalization_attempts": attempts,
                "source_observations": observations,
            }
        )
        updates = {
            "version": proposal.version + 1,
            "proposal_only": review_reason is not None,
            "generalization_verified": review_reason is None,
            "updated_at": utc_now_iso(),
            "metadata": metadata,
        }
        if synthesized:
            updates.update(
                {
                    "text": rule.lesson,
                    "lesson": rule.lesson,
                    "trigger_pattern": rule.trigger_pattern,
                    "evidence_pattern": rule.evidence_pattern,
                    "argument_pattern": rule.argument_pattern,
                    "recommended_action": rule.recommended_action,
                    "applicability_scope": rule.applicability_scope,
                    "exceptions": list(rule.exceptions),
                    "confidence": rule.confidence,
                }
            )
        reconsidered = proposal.model_copy(update=updates)
        if review_reason is None:
            reconsidered = self._activate_generalization(reconsidered, rule)
            events.append(self._recovery_event(reconsidered))
            return reconsidered, True

        events.append(
            ConsolidationEvent(
                event_id=f"evt_{stable_hash_text('generalization_failed' + proposal.memory_id + str(attempt_count) + str(len(observations)))}",
                event_type="generalization_failed",
                memory_id=proposal.memory_id,
                stm_ids=sorted(
                    reconsidered.metadata.get("generalized_from_stm_ids", [])
                ),
                details={
                    "reason": review_reason,
                    "attempt": attempt_count,
                    "support": len(observations),
                    "proposal_created": False,
                },
            )
        )
        return reconsidered, False

    def _activate_generalization(
        self, record: MemoryRecord, rule: _GeneralizedRule
    ) -> MemoryRecord:
        metadata = dict(record.metadata)
        metadata.update(
            {
                "proposal_only": False,
                "generalization_verified": True,
                "generalization_review_reason": None,
            }
        )
        return record.model_copy(
            update={
                "status": "active",
                "proposal_only": False,
                "generalization_verified": True,
                "verified_by": "memory_consolidator_generalization",
                "confidence": rule.confidence,
                "updated_at": utc_now_iso(),
                "metadata": metadata,
            }
        )

    def _recovery_event(self, record: MemoryRecord) -> ConsolidationEvent:
        stm_ids = sorted(record.metadata.get("generalized_from_stm_ids", []))
        return ConsolidationEvent(
            event_id=f"evt_{stable_hash_text('generalization_recovered' + record.memory_id + '|'.join(stm_ids))}",
            event_type="generalization_recovered",
            memory_id=record.memory_id,
            stm_ids=stm_ids,
            details={
                "status": "active",
                "support": record.support_count,
                "attempt": record.metadata.get("generalization_attempt_count"),
            },
        )

    def _synthesize_generalization(
        self, observations: list[dict]
    ) -> tuple[_GeneralizedRule, bool, str | None]:
        try:
            if self.llm_client is None:
                raise RuntimeError("llm_generalization_unavailable")
            prompt = (
                "These repeated observations from independent multimedia verification cases "
                "form one cluster. Synthesize ONE generalized rule as JSON: "
                '{"trigger_pattern": "...", "lesson": "...", "evidence_pattern": "...", '
                '"argument_pattern": "...", "recommended_action": "...", '
                '"applicability_scope": "...", "exceptions": [...], "confidence": 0.0-1.0}\n'
                f"Observations: {observations}"
            )
            return (
                _GeneralizedRule.model_validate(
                    self.llm_client.generate_json(prompt)
                ),
                True,
                None,
            )
        except Exception as exc:
            representative = max(
                observations, key=lambda row: float(row.get("confidence", 0.0))
            )
            logger.warning("Rule generalization held for review: %s", exc)
            return (
                _GeneralizedRule(
                    trigger_pattern=representative.get("text") or "",
                    lesson=representative.get("text") or "",
                    recommended_action=(
                        "Review the repeated pattern before reuse."
                    ),
                    confidence=min(
                        0.6, float(representative.get("confidence", 0.0))
                    ),
                ),
                False,
                (
                    "generalization_synthesis_unavailable"
                    if self.llm_client is None
                    else "generalization_synthesis_failed"
                ),
            )

    def _generalization_review_reason(
        self,
        rule: _GeneralizedRule,
        *,
        synthesized: bool,
        synthesis_reason: str | None,
        verification_succeeded: bool,
        verification_reason: str,
        observations: list[dict],
    ) -> str | None:
        thresholds = self.config.consolidation.semantic_rule
        independent = _select_independent_observations(observations)
        if not synthesized:
            return synthesis_reason or "generalization_synthesis_failed"
        if not verification_succeeded:
            return verification_reason
        if len(independent) < thresholds.min_distinct_cases:
            return "generalization_distinct_cases_below_threshold"
        if len(independent) < thresholds.min_distinct_sources:
            return "generalization_distinct_sources_below_threshold"
        if rule.confidence < thresholds.min_confidence:
            return "generalization_confidence_below_threshold"
        if not independent or not all(
            observation.get("grounded", False) for observation in independent
        ):
            return "generalization_grounding_unavailable"
        return None

    def _proposal_source_observations(
        self, proposal: MemoryRecord
    ) -> list[dict]:
        structured = proposal.metadata.get("source_observations")
        if isinstance(structured, list) and structured:
            return [dict(row) for row in structured if isinstance(row, dict)]

        # Older proposals stored only parallel identity lists and normalized
        # source texts. Reconstruct conservatively without lowering legacy
        # support_count; future independent rows can still recover the proposal.
        texts = list(proposal.metadata.get("source_observation_texts", []))
        cases = list(proposal.source_case_ids)
        fingerprints = list(proposal.source_fingerprints)
        size = max(len(texts), len(cases), len(fingerprints))
        observations = []
        for index in range(size):
            case_id = cases[index] if index < len(cases) else f"legacy_case_{index}"
            fingerprint = (
                fingerprints[index]
                if index < len(fingerprints)
                else f"fp_case_{case_id}"
            )
            text = texts[index] if index < len(texts) else proposal.text
            observations.append(
                {
                    "case_id": case_id,
                    "source_fingerprint": fingerprint,
                    "stm_id": None,
                    "text": text,
                    "confidence": proposal.confidence,
                    "grounded": bool(text),
                }
            )
        return observations

    def _verify_generalization(
        self, rule: _GeneralizedRule, observations: list[dict]
    ) -> tuple[bool, str]:
        prompt = (
            "Check whether this generalized rule is consistent with EVERY source observation. "
            'Return JSON {"consistent_with_all": true/false, "reason": "..."}\n'
            f"Rule: {rule.model_dump()}\nObservations: {observations}"
        )
        try:
            if self.llm_client is None:
                raise RuntimeError("llm_generalization_unavailable")
            data = self.llm_client.generate_json(prompt)
            if bool(data.get("consistent_with_all", False)):
                return True, "generalization_verified"
            return False, "generalization_verification_failed"
        except Exception:
            return False, "generalization_verification_unavailable"

    def _rollup_usage(
        self, ltm_by_id: dict[str, MemoryRecord]
    ) -> tuple[dict[str, MemoryRecord], list[ConsolidationEvent]]:
        grouped: dict[tuple[str, str, str], list] = {}
        for event in self.store.load_usage_events():
            split = (event.dataset_split or "").lower()
            if event.frozen or split in {"validation", "val", "dev", "test", "test_hidden"}:
                continue
            if event.stage not in {"planner_cited", "argument_cited"}:
                continue
            key = (event.run_id or event.case_id, event.case_id, event.memory_id)
            grouped.setdefault(key, []).append(event)
        updates: dict[str, MemoryRecord] = {}
        consolidation_events: list[ConsolidationEvent] = []
        by_memory: dict[str, list[tuple[str, str, list]]] = {}
        for (run_id, case_id, memory_id), rows in grouped.items():
            by_memory.setdefault(memory_id, []).append((run_id, case_id, rows))
        for memory_id, uses in by_memory.items():
            target = ltm_by_id.get(memory_id)
            if target is None:
                continue
            metadata = dict(target.metadata)
            processed = set(metadata.get("usage_use_ids", []))
            additions = []
            for run_id, case_id, rows in uses:
                use_id = f"usage_{stable_hash_text(run_id + case_id + memory_id)}"
                if use_id in processed:
                    continue
                outcomes = {row.outcome for row in rows}
                outcome = "contested" if "contested" in outcomes else (
                    "successful" if "successful" in outcomes else "unknown"
                )
                additions.append((use_id, outcome, max((row.created_at or "") for row in rows)))
            if not additions:
                continue
            processed.update(use_id for use_id, _, _ in additions)
            metadata["usage_use_ids"] = sorted(processed)
            updated = target.model_copy(update={
                "version": target.version + 1,
                "usage_count": target.usage_count + len(additions),
                "successful_usage_count": target.successful_usage_count + sum(1 for _, outcome, _ in additions if outcome == "successful"),
                "contested_usage_count": target.contested_usage_count + sum(1 for _, outcome, _ in additions if outcome == "contested"),
                "last_used_at": max(timestamp for _, _, timestamp in additions) or target.last_used_at,
                "updated_at": utc_now_iso(),
                "metadata": metadata,
            })
            updates[memory_id] = updated
            use_ids = sorted(use_id for use_id, _, _ in additions)
            consolidation_events.append(ConsolidationEvent(
                event_id=f"evt_{stable_hash_text('usage_rollup' + memory_id + '|'.join(use_ids))}",
                event_type="usage_rollup",
                memory_id=memory_id,
                details={"use_ids": use_ids, "count": len(additions)},
            ))
        return updates, consolidation_events

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
            and not record.metadata.get("proposal_only", False)
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


def _source_fingerprint(observation) -> str:
    if isinstance(observation, dict):
        fingerprint = observation.get("source_fingerprint")
        case_id = observation.get("case_id") or observation.get("source_case_id")
    else:
        fingerprint = getattr(observation, "source_fingerprint", None)
        case_id = getattr(observation, "source_case_id", None)
    return str(fingerprint or f"fp_case_{case_id}")


def _source_case_id(observation) -> str:
    if isinstance(observation, dict):
        return str(observation.get("case_id") or observation.get("source_case_id"))
    return str(observation.source_case_id)


def _select_independent_observations(
    observations: list,
    seen_case_ids=(),
    seen_fingerprints=(),
) -> list:
    """Select observations for which both the case and source are unseen."""
    cases = set(seen_case_ids)
    fingerprints = set(seen_fingerprints)
    accepted = []
    for observation in observations:
        case_id = _source_case_id(observation)
        fingerprint = _source_fingerprint(observation)
        if case_id in cases or fingerprint in fingerprints:
            continue
        cases.add(case_id)
        fingerprints.add(fingerprint)
        accepted.append(observation)
    return accepted


def _source_observation(row: ShortTermMemoryRecord) -> dict:
    return {
        "case_id": row.source_case_id,
        "source_fingerprint": _source_fingerprint(row),
        "stm_id": row.stm_id,
        "text": row.text,
        "confidence": row.confidence,
        "grounded": bool(
            row.grounding_evidence_ids or row.grounding_argument_ids
        ),
        "dataset_name": row.dataset_name,
        "dataset_split": row.dataset_split,
    }


def _rule_from_record(record: MemoryRecord) -> _GeneralizedRule:
    return _GeneralizedRule(
        trigger_pattern=record.trigger_pattern or record.text,
        lesson=record.lesson or record.text,
        evidence_pattern=record.evidence_pattern,
        argument_pattern=record.argument_pattern,
        recommended_action=record.recommended_action
        or "Review the repeated pattern before reuse.",
        applicability_scope=record.applicability_scope,
        exceptions=list(record.exceptions),
        confidence=record.confidence,
    )


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
