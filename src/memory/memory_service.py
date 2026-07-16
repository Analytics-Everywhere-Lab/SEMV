from __future__ import annotations

import logging
from pathlib import Path

from src.memory.memory_config import MemoryConfig, load_memory_config
from src.memory.memory_consolidator import MemoryConsolidator
from src.memory.memory_retriever import MemoryRetriever
from src.memory.memory_similarity import build_similarity_backend
from src.memory.memory_store import MemoryStore
from src.memory.memory_verifier import MemoryVerifier
from src.schemas.memory_schema import (
    ConsolidationResult,
    MemoryUpdateCandidate,
    MemoryUsageEvent,
    ShortTermMemoryRecord,
)
from src.utils.hashing import stable_hash_text
from src.utils.llm_client import LLMClient


logger = logging.getLogger("run_case")


class MemoryFrozenError(RuntimeError):
    """Raised when a frozen memory service is asked to mutate memory."""


class MemoryService:
    """One shared, configured memory facade injected into retrieval, reflection,
    verification, consolidation, and the evaluation runners.

    A frozen service (validation/test) can retrieve and log usage to an external
    directory, but refuses to stage or consolidate."""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        store: MemoryStore | None = None,
        llm_client: LLMClient | None = None,
        frozen: bool = False,
        usage_log_path: str | Path | None = None,
    ) -> None:
        self.config = config or (store.config if store is not None else load_memory_config())
        if store is None and frozen:
            configured_dir = self.config.paths.resolved_memory_dir()
            if (configured_dir / "manifest.json").exists():
                store = MemoryStore(configured_dir, read_only=True)
                self.config = store.config
        self.store = store or MemoryStore(config=self.config, read_only=frozen)
        if frozen:
            self.store.read_only = True
        self.llm_client = llm_client
        self.frozen = frozen
        self.usage_log_path = Path(usage_log_path) if usage_log_path else None
        self.similarity = build_similarity_backend(self.config, llm_client=llm_client)
        self.retriever = MemoryRetriever(
            store=self.store, config=self.config, similarity=self.similarity
        )
        self.consolidator = MemoryConsolidator(
            store=self.store,
            config=self.config,
            similarity=self.similarity,
            llm_client=llm_client,
        )
        self._cases_since_consolidation = 0

    # -------------------------------------------------------------- factories

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path | None = "configs/memory.yaml",
        override_path: str | Path | None = None,
        memory_dir: str | Path | None = None,
        llm_client: LLMClient | None = None,
        frozen: bool = False,
        usage_log_path: str | Path | None = None,
    ) -> "MemoryService":
        config = load_memory_config(config_path, override_path=override_path)
        if memory_dir is not None:
            config = config.with_memory_dir(memory_dir)
        if frozen and memory_dir is not None and (Path(memory_dir) / "manifest.json").exists():
            store = MemoryStore(Path(memory_dir), read_only=True)
            return cls(config=store.config, store=store, llm_client=llm_client, frozen=True, usage_log_path=usage_log_path)
        return cls(config=config, llm_client=llm_client, frozen=frozen, usage_log_path=usage_log_path)

    def verifier(self, llm_client: LLMClient | None = None) -> MemoryVerifier:
        return MemoryVerifier(
            llm_client=llm_client or self.llm_client,
            store=self.store,
            config=self.config,
            similarity=self.similarity,
        )

    # -------------------------------------------------------------- retrieval

    def retrieve_for_claims(
        self,
        bundle,
        claims,
        evidence,
        source_clusters=None,
        top_k=None,
        include_short_term: bool = False,
    ):
        extra_records = None
        if include_short_term and self.config.short_term.retrieve_during_bootstrap:
            extra_records = [
                self._short_term_as_record(row)
                for row in self.store.load_short_term(statuses=["staged"])
                if row.verification_status == "verified"
            ]
        return self.retriever.retrieve_for_claims(
            bundle=bundle,
            claims=claims,
            evidence=evidence,
            source_clusters=source_clusters,
            top_k=top_k,
            extra_records=extra_records,
        )

    @staticmethod
    def _short_term_as_record(row: ShortTermMemoryRecord):
        from src.schemas.memory_schema import MemoryRecord

        return MemoryRecord(
            memory_id=row.stm_id,
            memory_type=row.memory_type,
            memory_level="short_term",
            case_id=row.source_case_id,
            claim_type=row.claim_type,
            task_type=row.task_type,
            text=row.text,
            trigger_pattern=row.trigger_pattern,
            lesson=row.lesson,
            evidence_pattern=row.evidence_pattern,
            argument_pattern=row.argument_pattern,
            recommended_action=row.recommended_action,
            failure_type=row.failure_type,
            canonical_key=row.canonical_key,
            confidence=row.confidence,
            support_count=1,
            status="active",
            origin="consolidated",
            metadata={"short_term": True, "stm_id": row.stm_id},
        )

    def retrieve(self, case, claim, evidence, top_k=None):
        return self.retriever.retrieve(case=case, claim=claim, evidence=evidence, top_k=top_k)

    # ---------------------------------------------------------------- staging

    def stage_candidates(
        self,
        candidates: list[MemoryUpdateCandidate],
    ) -> list[ShortTermMemoryRecord]:
        if self.frozen:
            raise MemoryFrozenError("Frozen memory cannot stage candidates.")
        return self.consolidator.apply(candidates)

    # ------------------------------------------------------------ consolidation

    def register_case_processed(self) -> ConsolidationResult | None:
        """Count a training/bootstrap case and consolidate on the configured schedule."""
        if self.frozen:
            raise MemoryFrozenError("Frozen memory cannot register training cases.")
        self._cases_since_consolidation += 1
        if self._cases_since_consolidation >= self.config.consolidation.every_n_cases:
            self._cases_since_consolidation = 0
            return self.consolidate()
        return None

    def consolidate(
        self, dry_run: bool = False, retry_under_review: bool = False
    ) -> ConsolidationResult:
        if self.frozen and not dry_run:
            raise MemoryFrozenError("Frozen memory cannot be consolidated.")
        return self.consolidator.consolidate(
            dry_run=dry_run, retry_under_review=retry_under_review
        )

    # ----------------------------------------------------------------- events

    def log_usage(
        self,
        case_id: str,
        memory_id: str,
        stage: str,
        claim_id: str | None = None,
        argument_id: str | None = None,
        outcome: str = "unknown",
        dataset_name: str | None = None,
        dataset_split: str | None = None,
        run_id: str | None = None,
        protocol_phase: str | None = None,
    ) -> MemoryUsageEvent:
        event = MemoryUsageEvent(
            event_id=f"use_{stable_hash_text((run_id or case_id) + case_id + memory_id + stage + (claim_id or chr(0)) + (argument_id or chr(0)) + outcome)}",
            run_id=run_id or case_id,
            protocol_phase=protocol_phase or ("frozen_eval" if self.frozen else "training"),
            case_id=case_id,
            memory_id=memory_id,
            stage=stage,  # type: ignore[arg-type]
            claim_id=claim_id,
            argument_id=argument_id,
            outcome=outcome,  # type: ignore[arg-type]
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            frozen=self.frozen,
        )
        # Frozen runs must never write inside the frozen snapshot directory;
        # usage events go to the evaluation output directory instead.
        override = self.usage_log_path if (self.frozen or self.usage_log_path) else None
        if self.frozen and override is None:
            logger.debug("Frozen memory usage event dropped (no usage_log_path set).")
            return event
        self.store.append_usage_event(event, path_override=override)
        return event

    # ------------------------------------------------------------- snapshotting

    def snapshot(self, label: str | None = None) -> Path:
        return self.store.snapshot(label)

    def state_hash(self, include_short_term: bool = False) -> str:
        return self.store.state_hash(include_short_term=include_short_term)
