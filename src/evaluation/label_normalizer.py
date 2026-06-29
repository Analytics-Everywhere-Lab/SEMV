from __future__ import annotations

INTERNAL_LABELS = [
    "verified",
    "mostly_verified",
    "partially_verified",
    "false_context",
    "out_of_context_cheapfake",
    "misleading",
    "manipulated_or_synthetic",
    "uncertain",
    "insufficient_evidence",
    "not_applicable",
]

CHEAPFAKE_BINARY_MAP = {
    "verified": "not_cheapfake",
    "mostly_verified": "not_cheapfake",
    "partially_verified": "not_cheapfake_or_uncertain",
    "false_context": "cheapfake",
    "out_of_context_cheapfake": "cheapfake",
    "misleading": "cheapfake",
    "manipulated_or_synthetic": "manipulated",
    "uncertain": "abstain",
    "insufficient_evidence": "abstain",
    "not_applicable": "abstain",
}


def normalize_mv2026_label(text: str | None) -> str:
    if not text:
        return "uncertain"
    t = str(text).strip().lower()
    if t in INTERNAL_LABELS:
        return t
    if "manipulated" in t or "synthetic" in t or "ai-generated" in t:
        return "manipulated_or_synthetic"
    if "out of context" in t or "false context" in t:
        return "false_context"
    if "false" in t or "not taken" in t or "do not show" in t:
        return "false_context"
    if "status: verified" in t:
        return "verified"
    if "verified" in t and ("geo" in t or "time" in t or "location" in t):
        return "verified"
    if "mostly" in t:
        return "mostly_verified"
    if "partially" in t:
        return "partially_verified"
    if "uncertain" in t or "not possible" in t or "cannot determine" in t:
        return "uncertain"
    return "uncertain"


def normalize_cosmos_label(label) -> str:
    if label is None or label == "":
        return "uncertain"
    if isinstance(label, bool):
        return "out_of_context_cheapfake" if label else "verified"
    if isinstance(label, int):
        return "out_of_context_cheapfake" if label == 1 else "verified"
    t = str(label).strip().lower()
    if t in INTERNAL_LABELS:
        return t
    if t in {"1", "true"}:
        return "out_of_context_cheapfake"
    if t in {"0", "false"}:
        return "verified"
    if t in {"ooc", "out_of_context", "out-of-context", "fake", "misleading"}:
        return "out_of_context_cheapfake"
    if t in {"nooc", "not_out_of_context", "not-ooc", "real", "genuine"}:
        return "verified"
    return "uncertain"


def to_binary_cheapfake(label: str) -> str:
    return CHEAPFAKE_BINARY_MAP.get(label, "abstain")
