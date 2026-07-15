from __future__ import annotations

import re
from typing import Any

from src.memory.memory_config import MemoryConfig, load_memory_config
from src.memory.memory_service import MemoryService
from src.schemas.memory_schema import MemoryRecord, MemoryUpdateCandidate
from src.utils.hashing import stable_hash_text

from tests.conftest import FakeLLMClient


class MemoryFakeLLM:
    """Deterministic fake LLM for memory tests. No network, no vLLM."""

    def __init__(
        self,
        verify_result: bool = True,
        verify_reason: str = "test",
        relation: str | None = None,
        lessons: dict | None = None,
        raise_on_verify: bool = False,
        raise_on_lessons: bool = False,
    ) -> None:
        self.verify_result = verify_result
        self.verify_reason = verify_reason
        self.relation = relation
        self.lessons = lessons
        self.raise_on_verify = raise_on_verify
        self.raise_on_lessons = raise_on_lessons
        self.calls: list[str] = []

    def generate_json(self, prompt: str, system: str | None = None, schema: dict | None = None, **kwargs: Any):
        self.calls.append(prompt)
        if "Verify whether this memory lesson" in prompt:
            if self.raise_on_verify:
                raise RuntimeError("LLM verification backend unavailable")
            return {"verified": self.verify_result, "reason": self.verify_reason}
        if "Classify the semantic relation" in prompt:
            if self.relation is None:
                raise RuntimeError("relation check unavailable")
            return {"relation": self.relation}
        if "You are the reflection module" in prompt:
            if self.raise_on_lessons or self.lessons is None:
                raise RuntimeError("structured lesson generation unavailable")
            return self.lessons
        if "Synthesize ONE generalized rule" in prompt:
            raise RuntimeError("generalization unavailable")
        if "consistent with EVERY source observation" in prompt:
            return {"consistent_with_all": True}
        raise ValueError(f"MemoryFakeLLM has no handler for prompt: {prompt[:80]}")

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        self.calls.append(prompt)
        return "{}"

    def generate_text(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        return self.generate(user_prompt, system=system_prompt)


class CitingFakeLLM(FakeLLMClient):
    """Pipeline fake that cites the first offered memory id in plans/arguments
    and answers reflection prompts with structured lessons."""

    def __init__(self, lessons: dict | None = None) -> None:
        super().__init__()
        self.lessons = lessons

    def generate_json(self, prompt: str, system: str | None = None, schema: dict | None = None, **kwargs: Any):
        if "Generate concise support and attack arguments" in prompt:
            self.calls.append(("generate_json", prompt))
            evidence_ids = re.findall(r"'id': '([^']+)'", prompt)
            memory_ids = re.findall(r"\[((?:mem|rule)_[A-Za-z0-9_]+)\]", prompt)
            argument = {
                "stance": "support",
                "title": "Support argument",
                "text": "The linked evidence supports the sub-claim.",
                "evidence_ids": evidence_ids[:1],
                "rationale": "grounded",
            }
            if memory_ids:
                argument["used_memory_ids"] = memory_ids[:1]
            return {"arguments": [argument]}
        if "Create a concise research plan" in prompt:
            self.calls.append(("generate_json", prompt))
            memory_ids = re.findall(r"\[((?:mem|rule)_[A-Za-z0-9_]+)\]", prompt)
            return {
                "questions": ["What supports this?"],
                "search_queries": ["cached evidence"],
                "preferred_sources": ["cached fact checks"],
                "uncertainty_checks": ["Check provenance"],
                "used_memory_ids": memory_ids[:1],
            }
        if "You are the reflection module" in prompt:
            self.calls.append(("generate_json", prompt))
            if self.lessons is None:
                raise ValueError("no structured lessons configured")
            return self.lessons
        if "Classify the semantic relation" in prompt:
            self.calls.append(("generate_json", prompt))
            return {"relation": "unrelated"}
        return super().generate_json(prompt, system=system, schema=schema, **kwargs)


def make_memory_config(tmp_path, **overrides: Any) -> MemoryConfig:
    config = load_memory_config(config_path=None, overrides=overrides or None)
    return config.with_memory_dir(tmp_path / "memory")


def make_service(tmp_path, llm_client=None, frozen=False, usage_log_path=None, **overrides: Any) -> MemoryService:
    return MemoryService(
        config=make_memory_config(tmp_path, **overrides),
        llm_client=llm_client,
        frozen=frozen,
        usage_log_path=usage_log_path,
    )


def make_candidate(
    case_id: str = "case1",
    text: str = "When reverse search finds an earlier upload, attack the temporal claim.",
    memory_type: str = "failure",
    confidence: float = 0.8,
    fingerprint: str | None = None,
    split: str = "train",
    verified: bool = False,
    **kwargs: Any,
) -> MemoryUpdateCandidate:
    candidate = MemoryUpdateCandidate(
        candidate_id=f"cand_{stable_hash_text(case_id + memory_type + text)}",
        memory_type=memory_type,
        text=text,
        source_case_id=case_id,
        dataset_name="mv2026",
        dataset_split=split,
        task_type="multimedia_verification",
        source_fingerprint=fingerprint or f"fp_{case_id}",
        grounding_evidence_ids=kwargs.pop("grounding_evidence_ids", ["ev1"]),
        grounding_argument_ids=kwargs.pop("grounding_argument_ids", ["arg1"]),
        confidence=confidence,
        **kwargs,
    )
    if verified:
        candidate = candidate.model_copy(
            update={"verified": True, "verification_status": "verified"}
        )
    return candidate


def make_record(
    memory_id: str = "mem_existing",
    text: str = "When reverse search finds an earlier upload, attack the temporal claim.",
    memory_type: str = "failure",
    status: str = "active",
    confidence: float = 0.8,
    **kwargs: Any,
) -> MemoryRecord:
    kwargs.setdefault("task_type", "multimedia_verification")
    return MemoryRecord(
        memory_id=memory_id,
        memory_type=memory_type,
        text=text,
        status=status,
        confidence=confidence,
        **kwargs,
    )
