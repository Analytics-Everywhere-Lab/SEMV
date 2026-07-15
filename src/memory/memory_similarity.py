from __future__ import annotations

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

_NEGATION_TOKENS = {"not", "never", "no", "avoid", "reject", "without", "dont", "don't"}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    lowered = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(lowered.split())


def content_tokens(text: str | None) -> set[str]:
    return {token for token in normalize_text(text).split() if token not in _STOPWORDS}


def canonical_key(
    memory_type: str,
    claim_type: str | None,
    task_type: str | None,
    text: str | None,
) -> str:
    normalized = normalize_text(text)
    return "|".join(
        [
            memory_type or "",
            claim_type or "general",
            task_type or "any",
            stable_hash_text(normalized or "empty"),
        ]
    )


def semantic_signature(
    memory_type: str,
    claim_type: str | None,
    task_type: str | None,
    failure_type: str | None,
    polarity: str | None,
    applicability_scope: str | None,
) -> str:
    return "|".join(
        [
            memory_type or "",
            claim_type or "general",
            task_type or "any",
            failure_type or "none",
            polarity or "neutral",
            normalize_text(applicability_scope) or "any",
        ]
    )


def lexical_similarity(text_a: str | None, text_b: str | None) -> float:
    """Cheap token-Jaccard blended with character-trigram overlap; range [0, 1]."""
    tokens_a = content_tokens(text_a)
    tokens_b = content_tokens(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    token_jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    norm_a = normalize_text(text_a).replace(" ", "")
    norm_b = normalize_text(text_b).replace(" ", "")
    grams_a = {norm_a[i : i + 3] for i in range(max(0, len(norm_a) - 2))}
    grams_b = {norm_b[i : i + 3] for i in range(max(0, len(norm_b) - 2))}
    gram_jaccard = len(grams_a & grams_b) / len(grams_a | grams_b) if grams_a and grams_b else 0.0
    return 0.6 * token_jaccard + 0.4 * gram_jaccard


def _polarity_of(item: dict[str, Any]) -> str:
    polarity = item.get("polarity")
    if polarity in {"do", "avoid", "positive", "negative"}:
        return "positive" if polarity in {"do", "positive"} else "negative"
    tokens = content_tokens(item.get("text") or "") | content_tokens(item.get("recommended_action") or "")
    return "negative" if tokens & _NEGATION_TOKENS else "positive"


class SimilarityBackend(Protocol):
    def similarity(self, text_a: str | None, text_b: str | None) -> float:
        ...

    def relation(self, item_a: dict[str, Any], item_b: dict[str, Any]) -> SemanticRelation:
        ...


class HybridSimilarityBackend:
    """Default backend: canonical/lexical/structured matching plus optional
    batched LLM classification for ambiguous shortlisted pairs.

    Works entirely without embeddings, a vector DB, or network access; an LLM
    client only refines ambiguous pairs and every LLM failure degrades to the
    conservative `unrelated` relation (never to a merge or a conflict).
    """

    def __init__(
        self,
        config: SimilarityConfig | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.config = config or SimilarityConfig()
        self.llm_client = llm_client
        self._relation_cache: dict[tuple[str, str], SemanticRelation] = {}

    def similarity(self, text_a: str | None, text_b: str | None) -> float:
        return lexical_similarity(text_a, text_b)

    def shortlist(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        k: int | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        limit = k or self.config.lexical_shortlist_k
        scored = [
            (self.similarity(query_text, item.get("text") or ""), item) for item in candidates
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:limit]

    def relation(self, item_a: dict[str, Any], item_b: dict[str, Any]) -> SemanticRelation:
        if item_a.get("memory_type") != item_b.get("memory_type"):
            return "unrelated"

        key_a = item_a.get("canonical_key") or canonical_key(
            item_a.get("memory_type", ""), item_a.get("claim_type"), item_a.get("task_type"), item_a.get("text")
        )
        key_b = item_b.get("canonical_key") or canonical_key(
            item_b.get("memory_type", ""), item_b.get("claim_type"), item_b.get("task_type"), item_b.get("text")
        )
        if key_a == key_b:
            return "equivalent"

        text_a = item_a.get("text") or ""
        text_b = item_b.get("text") or ""
        sim = self.similarity(text_a, text_b)
        polarity_a = _polarity_of(item_a)
        polarity_b = _polarity_of(item_b)
        structured_match = self._structured_match(item_a, item_b)

        if sim >= self.config.duplicate_similarity and polarity_a == polarity_b:
            return "equivalent"

        if polarity_a != polarity_b and (
            sim >= self.config.contradiction_similarity
            or (structured_match and sim >= self.config.contradiction_similarity * 0.6)
        ):
            return "contradicts"

        tokens_a = content_tokens(text_a)
        tokens_b = content_tokens(text_b)
        if tokens_a and tokens_b and polarity_a == polarity_b:
            if tokens_a <= tokens_b or tokens_b <= tokens_a:
                return "entails"

        if sim >= self.config.contradiction_similarity and self.config.use_llm_relation_check:
            llm_relation = self._llm_relation(text_a, text_b)
            if llm_relation is not None:
                return llm_relation
            # Ambiguous but structurally aligned pairs with matching polarity are
            # close enough to merge even without an LLM ruling.
            if structured_match and polarity_a == polarity_b:
                return "equivalent"
            return "unrelated"

        if sim >= self.config.contradiction_similarity and structured_match and polarity_a == polarity_b:
            return "equivalent"

        return "unrelated"

    def relations_batch(
        self,
        pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> list[SemanticRelation]:
        return [self.relation(a, b) for a, b in pairs]

    @staticmethod
    def _structured_match(item_a: dict[str, Any], item_b: dict[str, Any]) -> bool:
        fields = ["task_type", "claim_type", "failure_type"]
        matches = 0
        comparable = 0
        for field in fields:
            value_a, value_b = item_a.get(field), item_b.get(field)
            if value_a or value_b:
                comparable += 1
                if value_a == value_b:
                    matches += 1
        trigger_sim = lexical_similarity(item_a.get("trigger_pattern"), item_b.get("trigger_pattern"))
        action_sim = lexical_similarity(item_a.get("recommended_action"), item_b.get("recommended_action"))
        field_match = comparable == 0 or matches == comparable
        return field_match and (trigger_sim >= 0.4 or action_sim >= 0.4)

    def _llm_relation(self, text_a: str, text_b: str) -> SemanticRelation | None:
        """Classify one ambiguous pair; returns None (not a merge) on any failure."""
        if self.llm_client is None:
            return None
        cache_key = (stable_hash_text(text_a), stable_hash_text(text_b))
        if cache_key in self._relation_cache:
            return self._relation_cache[cache_key]
        prompt = (
            "Classify the semantic relation between two verification memory statements. "
            'Return JSON exactly as {"relation": "equivalent|entails|contradicts|unrelated"}.\n'
            "equivalent: same guidance. entails: one is a more specific form of the other. "
            "contradicts: they recommend incompatible actions for the same situation. "
            "unrelated: different situations or guidance.\n"
            f"Statement A: {text_a}\nStatement B: {text_b}"
        )
        try:
            data = self.llm_client.generate_json(prompt)
            relation = str(data.get("relation", "")).strip().lower()
            if relation in {"equivalent", "entails", "contradicts", "unrelated"}:
                self._relation_cache[cache_key] = relation  # type: ignore[assignment]
                return relation  # type: ignore[return-value]
        except Exception:
            pass
        return None


class EmbeddingSimilarityBackend:
    """Optional embedding-based backend. Never imported unless explicitly
    configured via similarity.optional_embedding_backend; the default install
    must work without sentence-transformers or any model download."""

    def __init__(self, model_name: str, config: SimilarityConfig | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "similarity.optional_embedding_backend requires the optional "
                "'sentence-transformers' package; install it or set the backend to null."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._hybrid = HybridSimilarityBackend(config)

    def similarity(self, text_a: str | None, text_b: str | None) -> float:  # pragma: no cover
        import numpy as np

        vectors = self._model.encode([text_a or "", text_b or ""])
        denom = float(np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
        if denom == 0.0:
            return 0.0
        return float(np.dot(vectors[0], vectors[1]) / denom)

    def relation(self, item_a: dict[str, Any], item_b: dict[str, Any]) -> SemanticRelation:  # pragma: no cover
        return self._hybrid.relation(item_a, item_b)


def build_similarity_backend(
    config: MemoryConfig,
    llm_client: LLMClient | None = None,
) -> SimilarityBackend:
    similarity_cfg = config.similarity
    if similarity_cfg.backend == "embedding" or similarity_cfg.optional_embedding_backend:
        if not similarity_cfg.optional_embedding_backend:
            raise ValueError(
                "similarity.backend=embedding requires similarity.optional_embedding_backend."
            )
        return EmbeddingSimilarityBackend(similarity_cfg.optional_embedding_backend, similarity_cfg)
    return HybridSimilarityBackend(similarity_cfg, llm_client=llm_client)
