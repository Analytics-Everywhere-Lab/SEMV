from __future__ import annotations

from difflib import SequenceMatcher

from src.schemas.argument_schema import Argument
from src.schemas.claim_schema import SubClaim
from src.schemas.contestation_schema import ArgumentProvenance
from src.schemas.evidence_schema import EvidenceGraph, EvidenceItem
from src.schemas.memory_schema import MemoryRecord
from src.utils.diagnostics import record_fallback
from src.utils.hashing import stable_hash_text
from src.utils.llm_client import LLMClient


DEFAULT_STRENGTH_WEIGHTS = {
    "source_reliability": 0.35,
    "cross_source_corroboration": 0.25,
    "cross_modal_consistency": 0.25,
    "claim_relevance": 0.15,
}

# Flags that only signal a tool/adapter didn't run or data was absent — not
# evidence of manipulation. Anything else (e.g. software_tag_suspicious,
# metadata_stripped, black_border_or_recapture_cue) is a real detected signal
# and should still be able to drive an "attack" stance.
NEUTRAL_ONLY_FLAG_PREFIXES = (
    "exiftool_missing",
    "ffprobe_missing",
    "metadata_missing",
    "metadata_json_parse_failed",
    "metadata_tool_failed",
    "gps_missing",
    "creation_time_missing",
    "media_file_missing",
    "forensic_media_missing",
    "forensic_image_unreadable",
    "trufor_adapter_unavailable",
    "asr_media_missing",
    "vlm_media_missing",
    "ocr_media_missing",
    "ocr_adapter_unavailable",
    "local_reverse_search_disabled",
    "local_reverse_query_missing",
    "local_reverse_search_failed",
    "free_web_search_unavailable",
    "free_web_search_failed",
    "video_keyframes_unavailable",
    "external_research_cache_miss",
)


def _is_neutral_only_flag(flag: str) -> bool:
    return any(flag == prefix or flag.startswith(f"{prefix}:") for prefix in NEUTRAL_ONLY_FLAG_PREFIXES)


def _argument_provenance(claim: SubClaim, evidence_ids: list[str]) -> ArgumentProvenance:
    return ArgumentProvenance(
        source_step="argument_construction",
        subclaim_id=claim.claim_id,
        evidence_ids=list(evidence_ids),
        retrieval_query_ids=list(claim.search_queries),
        upstream_steps=[
            "claim_decomposition",
            "evidence_retrieval",
            "evidence_validation",
            "argument_construction",
        ],
    )


class ArgumentGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        claim: SubClaim,
        evidence: list[EvidenceItem],
        evidence_graph: EvidenceGraph,
        memory_items: list[MemoryRecord],
        case_id: str | None = None,
    ) -> list[Argument]:
        del evidence_graph
        memory_lines = [f"[{item.memory_id}] {item.text}" for item in memory_items[:3]]
        valid_memory_ids = {item.memory_id for item in memory_items[:3]}
        prompt = (
            "Generate concise support and attack arguments for this verification "
            "sub-claim. Return JSON as "
            '{"arguments":[{"stance":"support","title":"...","text":"...",'
            '"evidence_ids":["..."],"used_memory_ids":["..."],"rationale":"..."}]}.' "\n"
            "Memory items are guidance from past cases, NOT evidence: every argument "
            "must cite current-case evidence_ids, and used_memory_ids may only list a "
            "memory id that actually influenced the argument.\n"
            f"Sub-claim: {claim.statement}\n"
            f"Evidence: {[{'id': item.evidence_id, 'text': item.content[:220]} for item in evidence]}\n"
            f"Memory: {memory_lines}"
        )
        try:
            data = self.llm_client.generate_json(prompt)
            arguments = []
            for item in data.get("arguments", []):
                stance = item.get("stance", "neutral")
                if stance not in {"support", "attack", "mixed", "neutral"}:
                    stance = "neutral"
                text = item.get("text") or "No argument text returned."
                evidence_ids, flags = repair_evidence_ids(item.get("evidence_ids", []), text, evidence)
                used_memory_ids = sorted(
                    {
                        memory_id
                        for memory_id in item.get("used_memory_ids", [])
                        if memory_id in valid_memory_ids
                    }
                )
                if used_memory_ids and not item.get("evidence_ids"):
                    evidence_ids = []
                    flags = sorted(set(flags + ["missing_argument_evidence"]))
                argument = self._build_argument(
                    claim=claim,
                    stance=stance,
                    title=item.get("title") or f"{stance.title()} argument",
                    text=text,
                    evidence_ids=evidence_ids,
                    evidence=evidence,
                    rationale=item.get("rationale"),
                    uncertainty_flags=flags,
                    used_memory_ids=used_memory_ids,
                )
                arguments.append(argument)
            if arguments:
                return arguments
        except Exception as exc:
            record_fallback("argument_generation", exc, "evidence_direct_arguments", case_id=case_id, claim_id=claim.claim_id)
        return self._fallback_arguments(claim, evidence)

    def _fallback_arguments(self, claim: SubClaim, evidence: list[EvidenceItem]) -> list[Argument]:
        arguments = []
        for item in evidence:
            stance = self._infer_stance(item)
            arguments.append(
                self._build_argument(
                    claim=claim,
                    stance=stance,
                    title=f"{stance.title()} from {item.title or item.source}",
                    text=f"{item.content}",
                    evidence_ids=[item.evidence_id],
                    evidence=evidence,
                    rationale="Fallback argument generated directly from one evidence item.",
                    uncertainty_flags=item.uncertainty_flags,
                )
            )
        return arguments

    def _build_argument(
        self,
        claim: SubClaim,
        stance: str,
        title: str,
        text: str,
        evidence_ids: list[str],
        evidence: list[EvidenceItem],
        rationale: str | None,
        uncertainty_flags: list[str] | None = None,
        used_memory_ids: list[str] | None = None,
    ) -> Argument:
        components = strength_components(evidence_ids, evidence)
        intrinsic_strength = intrinsic_strength_from_components(components)
        flags = list(uncertainty_flags or [])
        if not evidence_ids:
            flags.append("missing_argument_evidence")
            intrinsic_strength *= 0.35
        if used_memory_ids and not evidence_ids:
            # Memory is guidance, not case evidence: an argument grounded only
            # in memory has no valid current-case grounding.
            flags.append("memory_only_grounding")
        return Argument(
            argument_id=self._argument_id(claim.claim_id, text),
            claim_id=claim.claim_id,
            stance=stance,  # type: ignore[arg-type]
            title=title,
            text=text,
            evidence_ids=evidence_ids,
            provenance_summary=_provenance_summary(evidence_ids, evidence),
            rationale=(
                rationale
                or ("Argument grounded in linked evidence." if evidence_ids else "Argument lacks validated evidence links.")
            ),
            intrinsic_strength=intrinsic_strength,
            intrinsic_score=intrinsic_strength,
            strength_components=components,
            provenance=_argument_provenance(claim, evidence_ids),
            relevance=components["claim_relevance"],
            reliability=components["source_reliability"],
            corroboration=components["cross_source_corroboration"],
            cross_modal_consistency=components["cross_modal_consistency"],
            uncertainty_flags=sorted(set(flags)),
            metadata={"used_memory_ids": sorted(used_memory_ids or [])},
        )

    @staticmethod
    def _infer_stance(item: EvidenceItem) -> str:
        metadata_stance = item.metadata.get("stance")
        if metadata_stance in {"support", "attack", "mixed", "neutral"}:
            return metadata_stance

        # Evidence items produced when a tool/adapter didn't run at all
        # (source_type="synthetic_uncertainty") are never attack signals.
        if item.source_type == "synthetic_uncertainty":
            return "neutral"

        text = item.content.lower()
        attack_terms = ["false", "misleading", "not ", "unrelated", "different", "old", "out of context"]
        if any(term in text for term in attack_terms):
            return "attack"

        flags = item.uncertainty_flags or []
        if flags:
            if all(_is_neutral_only_flag(flag) for flag in flags):
                return "neutral"
            return "attack"

        return "support"

    @staticmethod
    def _argument_id(claim_id: str, seed: str) -> str:
        return f"arg_{stable_hash_text(claim_id + seed)}"


def repair_evidence_ids(raw_ids: list[str], text: str, evidence: list[EvidenceItem]) -> tuple[list[str], list[str]]:
    valid_ids = {item.evidence_id for item in evidence}
    repaired = [evidence_id for evidence_id in raw_ids if evidence_id in valid_ids]
    flags = []
    invalid = [evidence_id for evidence_id in raw_ids if evidence_id not in valid_ids]
    if invalid:
        flags.append("invalid_evidence_id_repaired")
    if not repaired:
        nearest = _nearest_evidence(text, evidence)
        if nearest:
            repaired = [nearest.evidence_id]
            flags.append("missing_evidence_id_repaired")
    return repaired, flags


def _nearest_evidence(text: str, evidence: list[EvidenceItem]) -> EvidenceItem | None:
    if not evidence:
        return None
    best = None
    best_score = 0.0
    for item in evidence:
        score = SequenceMatcher(None, text.lower(), item.content.lower()).ratio()
        if score > best_score:
            best = item
            best_score = score
    return best if best_score >= 0.12 else None


def strength_components(evidence_ids: list[str], evidence: list[EvidenceItem]) -> dict[str, float]:
    linked = [item for item in evidence if item.evidence_id in evidence_ids]
    if not linked:
        return {
            "source_reliability": 0.0,
            "cross_source_corroboration": 0.0,
            "cross_modal_consistency": 0.0,
            "claim_relevance": 0.0,
        }
    return {
        "source_reliability": sum(item.reliability for item in linked) / len(linked),
        "cross_source_corroboration": min(len({item.source for item in linked}) / 3.0, 1.0),
        "cross_modal_consistency": min(
            len({item.metadata.get("modality", item.source_type) for item in linked}) / 2.0,
            1.0,
        ),
        "claim_relevance": sum(item.relevance for item in linked) / len(linked),
    }


def intrinsic_strength_from_components(components: dict[str, float]) -> float:
    score = sum(
        DEFAULT_STRENGTH_WEIGHTS[key] * components.get(key, 0.0)
        for key in DEFAULT_STRENGTH_WEIGHTS
    )
    return max(0.0, min(1.0, score))


def _provenance_summary(evidence_ids: list[str], evidence: list[EvidenceItem]) -> str:
    titles = [item.title or item.source for item in evidence if item.evidence_id in evidence_ids]
    return "; ".join(titles) if titles else "No validated evidence provenance."
