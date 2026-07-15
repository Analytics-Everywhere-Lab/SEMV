from __future__ import annotations

import logging

from src.memory.memory_config import MemoryConfig
from src.memory.memory_similarity import (
    SimilarityBackend,
    build_similarity_backend,
    canonical_key,
    content_tokens,
    normalize_text,
)
from src.memory.memory_store import MemoryStore
from src.schemas.memory_schema import MemoryRecord, MemoryUpdateCandidate
from src.utils.llm_client import LLMClient


logger = logging.getLogger("run_case")

NON_TRAINING_SPLITS = {"validation", "val", "dev", "test", "test_hidden"}

_UNIVERSAL_TOKENS = {"always", "never", "all", "every", "any", "guaranteed"}


class MemoryVerifier:
    """Fail-closed candidate verification.

    Stages: (1) schema/grounding validation, (2) confidence threshold,
    (3) exact/canonical duplicate check, (4) semantic equivalence/contradiction
    check, (5) LLM safety/generalizability verification where required.

    Any LLM failure (timeout, parse error, unavailable model, any exception)
    yields verified=False with verification_status="under_review" — never
    automatic acceptance.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        store: MemoryStore | None = None,
        config: MemoryConfig | None = None,
        similarity: SimilarityBackend | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or (store.config if store is not None else MemoryConfig())
        self.store = store or MemoryStore(config=self.config)
        self.similarity = similarity or build_similarity_backend(self.config, llm_client=llm_client)

    def verify(
        self,
        candidate: MemoryUpdateCandidate,
        valid_evidence_ids: set[str] | None = None,
        valid_argument_ids: set[str] | None = None,
    ) -> MemoryUpdateCandidate:
        candidate = self._ensure_keys(candidate)

        # Stage 0: candidates derived from validation/test cases never enter memory.
        if candidate.dataset_split and candidate.dataset_split.lower() in NON_TRAINING_SPLITS:
            return self._rejected(candidate, "non_training_split")

        # Stage 1: schema and grounding validation.
        if candidate.verification_status == "under_review" and candidate.rejected_reason:
            # Lesson generation already flagged this candidate (e.g. invented IDs
            # or a structured-generation failure); keep it held for review.
            return self._under_review(candidate, candidate.rejected_reason)
        if self.config.verification.require_grounding:
            if not candidate.grounding_evidence_ids and not candidate.grounding_argument_ids:
                return self._rejected(candidate, "missing_grounding")
            if valid_evidence_ids is not None:
                invented = [
                    eid for eid in candidate.grounding_evidence_ids if eid not in valid_evidence_ids
                ]
                if invented:
                    return self._under_review(candidate, "grounding_evidence_ids_not_in_report")
            if valid_argument_ids is not None:
                invented = [
                    aid for aid in candidate.grounding_argument_ids if aid not in valid_argument_ids
                ]
                if invented:
                    return self._under_review(candidate, "grounding_argument_ids_not_in_report")

        # Stage 2: confidence threshold from configuration.
        if candidate.confidence < self.config.verification.min_confidence:
            return self._rejected(candidate, "confidence_below_threshold")

        existing = self.store.load_long_term(statuses=["active"])

        # Stage 3: exact/canonical duplicate check. A duplicate from a case (or
        # source fingerprint) that already supports the record adds nothing; an
        # equivalent observation from a NEW case is kept so consolidation can
        # count it as independent support.
        for record in existing:
            record_key = record.canonical_key or canonical_key(
                record.memory_type, record.claim_type, record.task_type, record.text
            )
            if record_key == candidate.canonical_key:
                known_case = candidate.source_case_id in record.source_case_ids
                known_fingerprint = (
                    candidate.source_fingerprint is not None
                    and candidate.source_fingerprint in record.source_fingerprints
                )
                if known_case or known_fingerprint:
                    return self._rejected(candidate, "duplicate_from_same_source")

        # Stage 4: semantic contradiction check against stronger active memory.
        conflict = self._find_contradiction(candidate, existing)
        if conflict is not None:
            if self.config.verification.reject_on_conflict and self._is_stronger(conflict, candidate):
                return self._rejected(
                    candidate, f"contradicts_active_memory:{conflict.memory_id}"
                )
            return self._under_review(candidate, f"contradicts_active_memory:{conflict.memory_id}")

        # Stage 5a: deterministic overgeneralization screen for single-case rules.
        if candidate.memory_type == "semantic_rule" and self._overgeneralizes(candidate):
            return self._under_review(candidate, "single_case_overgeneralization")

        # Stage 5b: LLM safety/generalizability verification.
        return self._llm_verify(candidate)

    # ------------------------------------------------------------------ steps

    def _llm_verify(self, candidate: MemoryUpdateCandidate) -> MemoryUpdateCandidate:
        prompt = (
            "Verify whether this memory lesson is safe to store for future multimedia "
            "verification cases. Reject lessons that are unsupported by the stated grounding, "
            "overgeneralize a single case, or present case-specific names, places, dates, or "
            "labels as universal rules. "
            'Return JSON with "verified" boolean and "reason" string.\n'
            f"Memory type: {candidate.memory_type}\n"
            f"Candidate: {candidate.text}\n"
            f"Trigger: {candidate.trigger_pattern or ''}\n"
            f"Recommended action: {candidate.recommended_action or ''}\n"
            f"Rationale: {candidate.rationale or ''}"
        )
        try:
            data = self.llm_client.generate_json(prompt)
            if not isinstance(data, dict) or "verified" not in data:
                return self._fail_closed(candidate, "llm_verification_invalid_response")
            if bool(data["verified"]):
                return candidate.model_copy(
                    update={
                        "verified": True,
                        "verification_status": "verified",
                        "verified_by": "memory_verifier_llm",
                        "rejected_reason": None,
                    }
                )
            return self._rejected(candidate, str(data.get("reason") or "llm_rejected"))
        except Exception as exc:
            logger.warning(
                "LLM memory verification unavailable for %s: %s", candidate.candidate_id, exc
            )
            return self._fail_closed(candidate, "llm_verification_unavailable")

    def _fail_closed(self, candidate: MemoryUpdateCandidate, reason: str) -> MemoryUpdateCandidate:
        if self.config.verification.fail_policy == "reject":
            return self._rejected(candidate, reason)
        return self._under_review(candidate, reason)

    def _find_contradiction(
        self,
        candidate: MemoryUpdateCandidate,
        existing: list[MemoryRecord],
    ) -> MemoryRecord | None:
        candidate_item = candidate.model_dump()
        items = [record.model_dump() for record in existing]
        if hasattr(self.similarity, "shortlist"):
            shortlisted = [
                item
                for _, item in self.similarity.shortlist(  # type: ignore[attr-defined]
                    candidate.text, items, k=self.config.similarity.lexical_shortlist_k
                )
            ]
        else:
            shortlisted = items
        by_id = {record.memory_id: record for record in existing}
        for record_item in shortlisted:
            if self.similarity.relation(candidate_item, record_item) == "contradicts":
                return by_id.get(record_item.get("memory_id", ""))
        return None

    @staticmethod
    def _is_stronger(record: MemoryRecord, candidate: MemoryUpdateCandidate) -> bool:
        return record.confidence >= candidate.confidence or record.independent_support() >= 2

    @staticmethod
    def _overgeneralizes(candidate: MemoryUpdateCandidate) -> bool:
        tokens = content_tokens(candidate.text)
        return bool(tokens & _UNIVERSAL_TOKENS)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _ensure_keys(candidate: MemoryUpdateCandidate) -> MemoryUpdateCandidate:
        updates: dict = {}
        if not candidate.normalized_text:
            updates["normalized_text"] = normalize_text(candidate.text)
        if not candidate.canonical_key:
            updates["canonical_key"] = canonical_key(
                candidate.memory_type, candidate.claim_type, candidate.task_type, candidate.text
            )
        return candidate.model_copy(update=updates) if updates else candidate

    @staticmethod
    def _rejected(candidate: MemoryUpdateCandidate, reason: str) -> MemoryUpdateCandidate:
        return candidate.model_copy(
            update={
                "verified": False,
                "verification_status": "rejected",
                "rejected_reason": reason,
            }
        )

    @staticmethod
    def _under_review(candidate: MemoryUpdateCandidate, reason: str) -> MemoryUpdateCandidate:
        return candidate.model_copy(
            update={
                "verified": False,
                "verification_status": "under_review",
                "rejected_reason": reason,
            }
        )
