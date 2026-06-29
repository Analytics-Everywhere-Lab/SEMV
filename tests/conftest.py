from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompt: str, system: str | None = None, **kwargs):
        self.calls.append(("generate", prompt))
        return "{}"

    def generate_text(self, system_prompt: str, user_prompt: str, **kwargs):
        self.calls.append(("generate_text", system_prompt, user_prompt))
        return "{}"

    def generate_json(self, prompt: str, system: str | None = None, schema: dict | None = None, **kwargs):
        self.calls.append(("generate_json", prompt))
        if "Decompose this multimedia verification claim" in prompt:
            return {
                "subclaims": [
                    {"claim_type": "what", "statement": "The media depicts the claimed event.", "search_queries": ["event"]},
                    {"claim_type": "where", "statement": "The media was captured in the claimed location.", "search_queries": ["location"]},
                    {"claim_type": "when", "statement": "The media was captured at the claimed time.", "search_queries": ["time"]},
                    {"claim_type": "who", "statement": "The claimed person appears in the media.", "search_queries": ["person"]},
                    {"claim_type": "why", "statement": "The sharing context is justified.", "search_queries": ["context"]},
                    {"claim_type": "authenticity", "statement": "The media is authentic and in context.", "search_queries": ["authenticity"]},
                ]
            }
        if "Create a concise research plan" in prompt:
            return {
                "questions": ["What supports this?", "What attacks this?"],
                "search_queries": ["cached evidence"],
                "preferred_sources": ["cached fact checks"],
                "uncertainty_checks": ["Check provenance"],
            }
        if "Check whether the argument is grounded" in prompt:
            return {"valid": True, "notes": "Grounded in linked evidence."}
        if "Verify whether this memory lesson" in prompt:
            return {"verified": True, "reason": "Safe lesson."}
        raise ValueError("FakeLLM intentionally falls back for this prompt")
