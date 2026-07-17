from __future__ import annotations

from src.memory.memory_config import MemoryConfig
from src.memory.memory_query_builder import build_memory_query
from src.memory.memory_similarity import SimilarityBackend, build_similarity_backend, content_tokens
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
    """Backend-aware retrieval with separate semantic and final-score gates."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        config: MemoryConfig | None = None,
        similarity: SimilarityBackend | None = None,
    ) -> None:
        self.config = config or (store.config if store is not None else MemoryConfig())
        self.store = store or MemoryStore(config=self.config)
        self.similarity = similarity or build_similarity_backend(self.config)

    def retrieve(
        self, case: MultimediaCase, claim: SubClaim, evidence: list[EvidenceItem],
        top_k: int | None = None, memory_types: list[str] | None = None,
    ) -> list[MemoryRecord]:
        query_text = " ".join(
            str(part) for part in [case.claim, case.context or "", claim.claim_type, claim.statement] if part
        )
        query = {
            "query_text": query_text,
            "claim_type": claim.claim_type,
            "task_type": None,
            "subtask": None,
            "scope_type": claim.metadata.get("scope_type") if claim.metadata else None,
        }
        return self._retrieve_ranked(query, evidence, top_k, memory_types=memory_types)

    def retrieve_for_claims(
        self, bundle, claims, evidence, source_clusters=None, top_k: int | None = None,
        extra_records: list[MemoryRecord] | None = None,
        memory_types: list[str] | None = None,
    ) -> dict[str, list[MemoryRecord]]:
        del source_clusters
        results = {}
        for claim in claims:
            query = build_memory_query(bundle, claim)
            query["query_text"] = " ".join(
                str(part) for part in [
                    query.get("title"), query.get("caption"), query.get("description"),
                    query.get("claim_statement"), query.get("claim_type"), query.get("task_type"),
                    query.get("subtask"), query.get("location_hint"),
                    " ".join(query.get("temporal_signals") or []),
                    " ".join(query.get("geolocation_cues") or []),
                ] if part
            )
            results[claim.claim_id] = self._retrieve_ranked(
                query, evidence, top_k, extra_records, memory_types=memory_types
            )
        return results

    def _retrieve_ranked(
        self, query: dict, evidence: list[EvidenceItem], top_k: int | None,
        extra_records: list[MemoryRecord] | None = None,
        memory_types: list[str] | None = None,
    ) -> list[MemoryRecord]:
        cfg = self.config.retrieval
        limit = top_k if top_k is not None else cfg.top_k
        statuses = ["active"] if cfg.active_only else None
        selected_types = list(cfg.include_memory_types) if memory_types is None else memory_types
        records = self.store.load_long_term(selected_types, statuses)
        if extra_records:
            records += [row for row in extra_records if row.memory_type in selected_types]
        context = self._evidence_context(evidence)
        scored: list[tuple[float, MemoryRecord, dict[str, float]]] = []
        for record in records:
            components = self._score_components(record, query, context)
            semantic = components["semantic_similarity"]
            final = components["final_score"]
            if semantic >= cfg.min_semantic_similarity and final >= cfg.min_final_score:
                scored.append((final, record, components))
        scored.sort(key=lambda item: (item[0], item[1].memory_id), reverse=True)
        deduped: list[tuple[float, MemoryRecord, dict[str, float]]] = []
        for score, record, components in scored:
            if any(self._equivalent(record, kept) for _, kept, _ in deduped):
                continue
            deduped.append((score, record, components))
            if len(deduped) >= limit:
                break
        return [
            record.model_copy(update={"metadata": {
                **record.metadata,
                "retrieval_score": round(score, 4),
                "retrieval_components": {key: round(value, 4) for key, value in components.items()},
            }})
            for score, record, components in deduped
        ]

    def _equivalent(self, first: MemoryRecord, second: MemoryRecord) -> bool:
        if first.canonical_key and first.canonical_key == second.canonical_key:
            return True
        return self.similarity.relation(first.model_dump(), second.model_dump()) == "equivalent"

    @staticmethod
    def _evidence_context(evidence: list[EvidenceItem] | None) -> dict:
        rows = evidence or []
        tokens: set[str] = set()
        claim_types: set[str] = set()
        provenance_count = 0
        uncertainty_count = 0
        for item in rows:
            tokens |= content_tokens(item.source_type or "")
            tokens |= content_tokens(item.content[:300])
            claim_types.update(item.supports_claim_types)
            if item.provenance:
                provenance_count += 1
                provenance = item.provenance.model_dump(mode="json")
                tokens |= content_tokens(" ".join(str(value) for value in provenance.values())[:300])
            uncertainty_count += int(bool(item.uncertainty_flags))
            for flag in item.uncertainty_flags:
                tokens |= content_tokens(flag)
        return {
            "tokens": tokens,
            "claim_types": claim_types,
            "mean_reliability": sum(item.reliability for item in rows) / len(rows) if rows else 0.0,
            "provenance_ratio": provenance_count / len(rows) if rows else 0.0,
            "uncertainty_ratio": uncertainty_count / len(rows) if rows else 0.0,
        }

    def _score_components(self, record: MemoryRecord, query: dict, context: dict) -> dict[str, float]:
        record_text = " ".join(
            str(part) for part in [
                record.text, record.lesson, record.trigger_pattern, record.argument_pattern,
                " ".join(record.tags),
            ] if part
        )
        semantic = max(0.0, min(1.0, self.similarity.similarity(query.get("query_text", ""), record_text)))
        compatibility = 0.0
        claim_type = query.get("claim_type")
        compatible_claims = set(record.metadata.get("compatible_claim_types", []))
        if record.claim_type == claim_type:
            compatibility += 0.5
        elif record.claim_type in {None, "general"} and claim_type in compatible_claims:
            compatibility += 0.5
        task_type = query.get("task_type")
        if not record.task_type or not task_type or record.task_type == task_type:
            compatibility += 0.25
        subtask = query.get("subtask")
        record_subtask = record.metadata.get("subtask")
        if not record_subtask or not subtask or record_subtask == subtask:
            compatibility += 0.15
        scope = query.get("scope_type")
        compatible_scopes = set(record.metadata.get("compatible_scopes", []))
        if not record.applicability_scope or not scope or record.applicability_scope == scope or scope in compatible_scopes:
            compatibility += 0.10

        pattern_tokens = content_tokens(record.evidence_pattern or "")
        overlap = len(pattern_tokens & context["tokens"]) / len(pattern_tokens) if pattern_tokens else 0.0
        claim_fit = 1.0 if claim_type and claim_type in context["claim_types"] else 0.0
        provenance_quality = context["provenance_ratio"] * context["mean_reliability"]
        uncertainty_penalty = 1.0 - context["uncertainty_ratio"]
        evidence_pattern = max(0.0, min(1.0,
            0.5 * overlap + 0.2 * claim_fit + 0.2 * provenance_quality + 0.1 * uncertainty_penalty
        ))
        confidence = record.confidence
        support = min(1.0, record.independent_support() / 5.0)
        usage_denominator = (record.successful_usage_count + record.unsuccessful_usage_count
                             + record.contested_usage_count)
        usage = record.successful_usage_count / usage_denominator if usage_denominator else 0.0
        final = (
            SCORE_WEIGHTS["similarity"] * semantic
            + SCORE_WEIGHTS["compatibility"] * compatibility
            + SCORE_WEIGHTS["evidence_pattern"] * evidence_pattern
            + SCORE_WEIGHTS["confidence"] * confidence
            + SCORE_WEIGHTS["support"] * support
            + SCORE_WEIGHTS["usage"] * usage
        )
        return {
            "semantic_similarity": semantic,
            "compatibility": compatibility,
            "evidence_pattern_score": evidence_pattern,
            "confidence_contribution": confidence,
            "support_contribution": support,
            "usage_contribution": usage,
            "final_score": max(0.0, min(1.0, final)),
        }
