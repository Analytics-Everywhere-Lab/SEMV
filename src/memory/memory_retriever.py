from __future__ import annotations

from src.memory.memory_config import MemoryConfig
from src.memory.memory_query_builder import build_memory_query
from src.memory.memory_similarity import (
    SimilarityBackend,
    build_similarity_backend,
    content_tokens,
    lexical_similarity,
)
from src.memory.memory_store import MemoryStore
from src.schemas.case_schema import MultimediaCase
from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceItem
from src.schemas.memory_schema import MemoryRecord


SCORE_WEIGHTS = {
    "similarity": 0.55,
    "compatibility": 0.15,
    "evidence_pattern": 0.10,
    "confidence": 0.10,
    "support": 0.05,
    "usage": 0.05,
}


class MemoryRetriever:
    """Retrieves active long-term memory for the current case.

    Retrieval scores are transparent, normalized to [0, 1]:
    0.55 query similarity + 0.15 claim/task/scope compatibility +
    0.10 evidence-pattern match + 0.10 memory confidence +
    0.05 independent support strength + 0.05 validated usage signal.

    Retrieved memory is guidance only — it must never be treated as evidence
    for the current claim."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        config: MemoryConfig | None = None,
        similarity: SimilarityBackend | None = None,
    ) -> None:
        self.config = config or (store.config if store is not None else MemoryConfig())
        self.store = store or MemoryStore(config=self.config)
        self.similarity = similarity or build_similarity_backend(self.config)

    # ------------------------------------------------------------------ public

    def retrieve(
        self,
        case: MultimediaCase,
        claim: SubClaim,
        evidence: list[EvidenceItem],
        top_k: int | None = None,
    ) -> list[MemoryRecord]:
        query_text = " ".join(
            str(part)
            for part in [case.claim, case.context or "", claim.claim_type, claim.statement]
            if part
        )
        query = {
            "query_text": query_text,
            "claim_type": claim.claim_type,
            "task_type": None,
            "scope_type": None,
        }
        return self._retrieve_ranked(query, evidence, top_k)

    def retrieve_for_claims(
        self,
        bundle,
        claims,
        evidence,
        source_clusters=None,
        top_k: int | None = None,
        extra_records: list[MemoryRecord] | None = None,
    ) -> dict[str, list[MemoryRecord]]:
        del source_clusters  # already part of build_memory_query via the bundle
        results: dict[str, list[MemoryRecord]] = {}
        for claim in claims:
            query = build_memory_query(bundle, claim)
            query["query_text"] = " ".join(
                str(part)
                for part in [
                    query.get("title"),
                    query.get("caption"),
                    query.get("description"),
                    query.get("claim_statement"),
                    query.get("claim_type"),
                    query.get("task_type"),
                    query.get("location_hint"),
                    " ".join(query.get("temporal_signals") or []),
                    " ".join(query.get("geolocation_cues") or []),
                ]
                if part
            )
            results[claim.claim_id] = self._retrieve_ranked(
                query, evidence, top_k, extra_records=extra_records
            )
        return results

    # ---------------------------------------------------------------- internal

    def _retrieve_ranked(
        self,
        query: dict,
        evidence: list[EvidenceItem],
        top_k: int | None,
        extra_records: list[MemoryRecord] | None = None,
    ) -> list[MemoryRecord]:
        retrieval_cfg = self.config.retrieval
        limit = top_k if top_k is not None else retrieval_cfg.top_k
        statuses = ["active"] if retrieval_cfg.active_only else None
        records = self.store.load_long_term(
            memory_types=list(retrieval_cfg.include_memory_types),
            statuses=statuses,
        )
        if extra_records:
            records = records + [
                record
                for record in extra_records
                if record.memory_type in retrieval_cfg.include_memory_types
            ]
        evidence_context = self._evidence_context(evidence)

        scored: list[tuple[float, MemoryRecord]] = []
        for record in records:
            score = self._score(record, query, evidence_context)
            if score >= retrieval_cfg.min_similarity:
                scored.append((score, record))
        scored.sort(key=lambda pair: (pair[0], pair[1].memory_id), reverse=True)

        deduped: list[tuple[float, MemoryRecord]] = []
        for score, record in scored:
            if any(self._equivalent(record, kept) for _, kept in deduped):
                continue
            deduped.append((score, record))
            if len(deduped) >= limit:
                break

        return [
            record.model_copy(
                update={"metadata": {**record.metadata, "retrieval_score": round(score, 4)}}
            )
            for score, record in deduped
        ]

    def _equivalent(self, record_a: MemoryRecord, record_b: MemoryRecord) -> bool:
        if record_a.canonical_key and record_a.canonical_key == record_b.canonical_key:
            return True
        return self.similarity.relation(record_a.model_dump(), record_b.model_dump()) == "equivalent"

    @staticmethod
    def _evidence_context(evidence: list[EvidenceItem] | None) -> dict:
        evidence = evidence or []
        tokens: set[str] = set()
        for item in evidence:
            tokens |= content_tokens(item.source_type or "")
            for flag in item.uncertainty_flags:
                tokens |= content_tokens(flag)
            provenance = item.provenance
            if provenance is not None and not isinstance(provenance, dict):
                provenance = provenance.model_dump(mode="json")
            if isinstance(provenance, dict):
                tokens |= content_tokens(" ".join(str(v) for v in provenance.values())[:200])
        reliabilities = [item.reliability for item in evidence]
        return {
            "tokens": tokens,
            "mean_reliability": sum(reliabilities) / len(reliabilities) if reliabilities else None,
            "linked_claim_types": {
                str(item.metadata.get("claim_id") or "") for item in evidence if item.metadata
            },
        }

    def _score(self, record: MemoryRecord, query: dict, evidence_context: dict) -> float:
        record_text = " ".join(
            str(part)
            for part in [
                record.text,
                record.lesson,
                record.trigger_pattern,
                record.argument_pattern,
                " ".join(record.tags),
            ]
            if part
        )
        similarity = lexical_similarity(query.get("query_text", ""), record_text)

        compatibility = 0.0
        claim_type = query.get("claim_type")
        if record.claim_type in {claim_type, "general"} or record.claim_type is None:
            compatibility += 0.5
        task_type = query.get("task_type")
        if not record.task_type or not task_type or record.task_type == task_type:
            compatibility += 0.3
        scope = query.get("scope_type")
        if not record.applicability_scope or not scope or scope in (record.applicability_scope or ""):
            compatibility += 0.2

        evidence_tokens = evidence_context["tokens"]
        pattern_tokens = content_tokens(record.evidence_pattern or "")
        if pattern_tokens and evidence_tokens:
            evidence_match = len(pattern_tokens & evidence_tokens) / len(pattern_tokens)
        else:
            evidence_match = 0.0

        support = min(1.0, record.independent_support() / 5.0)
        usage_total = record.successful_usage_count + record.contested_usage_count
        usage = record.successful_usage_count / usage_total if usage_total else 0.0

        score = (
            SCORE_WEIGHTS["similarity"] * similarity
            + SCORE_WEIGHTS["compatibility"] * compatibility
            + SCORE_WEIGHTS["evidence_pattern"] * evidence_match
            + SCORE_WEIGHTS["confidence"] * record.confidence
            + SCORE_WEIGHTS["support"] * support
            + SCORE_WEIGHTS["usage"] * usage
        )
        return max(0.0, min(1.0, score))
