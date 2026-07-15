from __future__ import annotations

import json
import re
from typing import Any, Protocol

from src.memory.memory_config import MemoryConfig, SimilarityConfig
from src.schemas.memory_schema import SemanticRelation
from src.utils.hashing import stable_hash_text
from src.utils.llm_client import LLMClient


_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "with", "as",
    "by", "at", "from", "when", "while", "should", "must", "into",
}
_NEGATION_TOKENS = {"not", "never", "no", "avoid", "reject", "without", "dont", "don\x27t"}
_DIRECTIONAL = {"entails", "a_entails_b", "b_entails_a"}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    lowered = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(lowered.split())


def content_tokens(text: str | None) -> set[str]:
    return {token for token in normalize_text(text).split() if token not in _STOPWORDS}


def canonical_key(memory_type: str, claim_type: str | None, task_type: str | None, text: str | None) -> str:
    return "|".join([
        memory_type or "", claim_type or "general", task_type or "any",
        stable_hash_text(normalize_text(text) or "empty"),
    ])


def semantic_signature(
    memory_type: str,
    claim_type: str | None,
    task_type: str | None,
    failure_type: str | None,
    polarity: str | None,
    applicability_scope: str | None,
) -> str:
    return "|".join([
        memory_type or "", claim_type or "general", task_type or "any",
        failure_type or "none", polarity or "neutral",
        normalize_text(applicability_scope) or "any",
    ])


def lexical_similarity(text_a: str | None, text_b: str | None) -> float:
    tokens_a = content_tokens(text_a)
    tokens_b = content_tokens(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    token_jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    norm_a = normalize_text(text_a).replace(" ", "")
    norm_b = normalize_text(text_b).replace(" ", "")
    grams_a = {norm_a[i:i + 3] for i in range(max(0, len(norm_a) - 2))}
    grams_b = {norm_b[i:i + 3] for i in range(max(0, len(norm_b) - 2))}
    gram_jaccard = len(grams_a & grams_b) / len(grams_a | grams_b) if grams_a and grams_b else 0.0
    return 0.6 * token_jaccard + 0.4 * gram_jaccard


def _polarity_of(item: dict[str, Any]) -> str:
    polarity = item.get("polarity")
    if polarity in {"do", "avoid", "positive", "negative"}:
        return "positive" if polarity in {"do", "positive"} else "negative"
    tokens = content_tokens(item.get("text") or "") | content_tokens(item.get("recommended_action") or "")
    return "negative" if tokens & _NEGATION_TOKENS else "positive"


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("metadata")
    return value if isinstance(value, dict) else {}


def _compatible(item_a: dict[str, Any], item_b: dict[str, Any]) -> bool:
    if item_a.get("memory_type") != item_b.get("memory_type"):
        return False
    task_a, task_b = item_a.get("task_type"), item_b.get("task_type")
    if task_a and task_b and task_a != task_b:
        return False
    subtask_a = item_a.get("subtask") or _metadata(item_a).get("subtask")
    subtask_b = item_b.get("subtask") or _metadata(item_b).get("subtask")
    if subtask_a and subtask_b and subtask_a != subtask_b:
        compatible_a = set(_metadata(item_a).get("compatible_subtasks", []))
        compatible_b = set(_metadata(item_b).get("compatible_subtasks", []))
        if subtask_b not in compatible_a and subtask_a not in compatible_b:
            return False
    claim_a = item_a.get("claim_type") or "general"
    claim_b = item_b.get("claim_type") or "general"
    if claim_a != claim_b:
        compatible_a = set(_metadata(item_a).get("compatible_claim_types", []))
        compatible_b = set(_metadata(item_b).get("compatible_claim_types", []))
        if claim_b not in compatible_a and claim_a not in compatible_b:
            return False
    failure_a, failure_b = item_a.get("failure_type"), item_b.get("failure_type")
    if failure_a != failure_b and (failure_a or failure_b):
        return False
    scope_a = normalize_text(item_a.get("applicability_scope"))
    scope_b = normalize_text(item_b.get("applicability_scope"))
    if scope_a != scope_b and (scope_a or scope_b):
        scopes_a = {normalize_text(value) for value in _metadata(item_a).get("compatible_scopes", [])}
        scopes_b = {normalize_text(value) for value in _metadata(item_b).get("compatible_scopes", [])}
        if scope_b not in scopes_a and scope_a not in scopes_b:
            return False
    return True


class SimilarityBackend(Protocol):
    def similarity(self, text_a: str | None, text_b: str | None) -> float: ...
    def relation(self, item_a: dict[str, Any], item_b: dict[str, Any]) -> SemanticRelation: ...


class HybridSimilarityBackend:
    """Scope-safe hybrid relation backend. Ambiguity fails closed."""

    def __init__(self, config: SimilarityConfig | None = None, llm_client: LLMClient | None = None) -> None:
        self.config = config or SimilarityConfig()
        self.llm_client = llm_client
        self._relation_cache: dict[tuple[str, str], SemanticRelation] = {}

    def similarity(self, text_a: str | None, text_b: str | None) -> float:
        return lexical_similarity(text_a, text_b)

    def shortlist(self, query_text: str, candidates: list[dict[str, Any]], k: int | None = None) -> list[tuple[float, dict[str, Any]]]:
        limit = k or self.config.lexical_shortlist_k
        scored = [(self.similarity(query_text, item.get("text") or ""), item) for item in candidates]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:limit]

    @staticmethod
    def _identity(item: dict[str, Any]) -> str:
        relevant = {
            key: item.get(key)
            for key in (
                "memory_type", "claim_type", "task_type", "subtask", "failure_type",
                "applicability_scope", "polarity", "text", "trigger_pattern", "recommended_action",
            )
        }
        return stable_hash_text(json.dumps(relevant, sort_keys=True, default=str), length=32)

    @staticmethod
    def _reverse(relation: SemanticRelation) -> SemanticRelation:
        if relation == "a_entails_b":
            return "b_entails_a"
        if relation == "b_entails_a":
            return "a_entails_b"
        return relation

    def _cache(self, item_a: dict[str, Any], item_b: dict[str, Any], relation: SemanticRelation) -> None:
        key_a, key_b = self._identity(item_a), self._identity(item_b)
        self._relation_cache[(key_a, key_b)] = relation
        self._relation_cache[(key_b, key_a)] = self._reverse(relation)

    def _deterministic(self, item_a: dict[str, Any], item_b: dict[str, Any]) -> tuple[SemanticRelation | None, float]:
        if not _compatible(item_a, item_b):
            return "unrelated", 0.0
        key_a = item_a.get("canonical_key") or canonical_key(
            item_a.get("memory_type", ""), item_a.get("claim_type"), item_a.get("task_type"), item_a.get("text")
        )
        key_b = item_b.get("canonical_key") or canonical_key(
            item_b.get("memory_type", ""), item_b.get("claim_type"), item_b.get("task_type"), item_b.get("text")
        )
        if key_a == key_b:
            return "equivalent", 1.0
        text_a, text_b = item_a.get("text") or "", item_b.get("text") or ""
        sim = self.similarity(text_a, text_b)
        polarity_a, polarity_b = _polarity_of(item_a), _polarity_of(item_b)
        if sim >= self.config.duplicate_similarity and polarity_a == polarity_b:
            return "equivalent", sim
        structured = self._structured_match(item_a, item_b)
        if polarity_a != polarity_b and (
            sim >= self.config.contradiction_similarity
            or (structured and sim >= self.config.contradiction_similarity * 0.6)
        ):
            return "contradicts", sim
        tokens_a, tokens_b = content_tokens(text_a), content_tokens(text_b)
        if tokens_a and tokens_b and polarity_a == polarity_b and sim >= self.config.contradiction_similarity * 0.6:
            if tokens_a > tokens_b:
                return "a_entails_b", sim
            if tokens_b > tokens_a:
                return "b_entails_a", sim
        if sim >= self.config.contradiction_similarity and self.config.use_llm_relation_check:
            return None, sim
        return "unrelated", sim

    def relation(self, item_a: dict[str, Any], item_b: dict[str, Any]) -> SemanticRelation:
        cache_key = (self._identity(item_a), self._identity(item_b))
        if cache_key in self._relation_cache:
            return self._relation_cache[cache_key]
        relation, _ = self._deterministic(item_a, item_b)
        if relation is None:
            relation = self._llm_relations([(item_a, item_b)])[0]
        self._cache(item_a, item_b, relation)
        return relation

    def relations_batch(self, pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[SemanticRelation]:
        results: list[SemanticRelation | None] = [None] * len(pairs)
        pending: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        for index, (item_a, item_b) in enumerate(pairs):
            cache_key = (self._identity(item_a), self._identity(item_b))
            if cache_key in self._relation_cache:
                results[index] = self._relation_cache[cache_key]
                continue
            relation, _ = self._deterministic(item_a, item_b)
            if relation is None:
                pending.append((index, item_a, item_b))
            else:
                results[index] = relation
                self._cache(item_a, item_b, relation)
        if pending:
            classified = self._llm_relations([(a, b) for _, a, b in pending])
            for (index, item_a, item_b), relation in zip(pending, classified):
                results[index] = relation
                self._cache(item_a, item_b, relation)
        return [result or "unrelated" for result in results]

    def _llm_relations(self, pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[SemanticRelation]:
        if self.llm_client is None:
            return ["unrelated"] * len(pairs)
        payload = [
            {"pair_id": str(index), "statement_a": a.get("text") or "", "statement_b": b.get("text") or ""}
            for index, (a, b) in enumerate(pairs)
        ]
        prompt = (
            "Classify the semantic relation for every pair in one batch. Return JSON as "
            "{\"results\":[{\"pair_id\":\"0\",\"relation\":\"equivalent|a_entails_b|b_entails_a|contradicts|unrelated\"}]}. "
            "Entailment is directional. Omitted or ambiguous pairs are unrelated.\nPairs: " + str(payload)
        )
        try:
            data = self.llm_client.generate_json(prompt)
            rows = data.get("results") if isinstance(data, dict) else None
            if rows is None and len(pairs) == 1 and isinstance(data, dict):
                rows = [{"pair_id": "0", "relation": data.get("relation")}]
            by_id = {str(row.get("pair_id")): str(row.get("relation", "")).lower() for row in (rows or [])}
            valid = {"equivalent", "a_entails_b", "b_entails_a", "contradicts", "unrelated"}
            output: list[SemanticRelation] = []
            for index in range(len(pairs)):
                relation = by_id.get(str(index), "unrelated")
                if relation == "entails":
                    relation = "a_entails_b"
                output.append(relation if relation in valid else "unrelated")
            return output
        except Exception:
            return ["unrelated"] * len(pairs)

    @staticmethod
    def _structured_match(item_a: dict[str, Any], item_b: dict[str, Any]) -> bool:
        fields = ["task_type", "claim_type", "failure_type", "applicability_scope"]
        comparable = [(item_a.get(field), item_b.get(field)) for field in fields if item_a.get(field) or item_b.get(field)]
        if any(a != b for a, b in comparable):
            return False
        trigger_sim = lexical_similarity(item_a.get("trigger_pattern"), item_b.get("trigger_pattern"))
        action_sim = lexical_similarity(item_a.get("recommended_action"), item_b.get("recommended_action"))
        return trigger_sim >= 0.4 or action_sim >= 0.4


class EmbeddingSimilarityBackend(HybridSimilarityBackend):
    def __init__(self, model_name: str, config: SimilarityConfig | None = None, llm_client: LLMClient | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "similarity.optional_embedding_backend requires sentence-transformers."
            ) from exc
        self._model = SentenceTransformer(model_name)
        super().__init__(config=config, llm_client=llm_client)

    def similarity(self, text_a: str | None, text_b: str | None) -> float:
        import numpy as np
        vectors = self._model.encode([text_a or "", text_b or ""])
        denom = float(np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
        if denom == 0.0:
            return 0.0
        return max(0.0, min(1.0, float(np.dot(vectors[0], vectors[1]) / denom)))


def build_similarity_backend(config: MemoryConfig, llm_client: LLMClient | None = None) -> SimilarityBackend:
    similarity_cfg = config.similarity
    if similarity_cfg.backend == "embedding" or similarity_cfg.optional_embedding_backend:
        if not similarity_cfg.optional_embedding_backend:
            raise ValueError("similarity.backend=embedding requires similarity.optional_embedding_backend.")
        return EmbeddingSimilarityBackend(similarity_cfg.optional_embedding_backend, similarity_cfg, llm_client)
    return HybridSimilarityBackend(similarity_cfg, llm_client=llm_client)
