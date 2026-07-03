from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol

import requests

from src.utils.env_loader import load_env_file


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        ...

    def generate_text(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        ...

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        schema: dict | None = None,
        **kwargs: Any,
    ) -> Any:
        ...


class OllamaLLMClient:
    """Shared Ollama client used by all agent-like pipeline components."""

    def __init__(self, env_path: str | None = None) -> None:
        load_env_file(env_path)
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL")
        if not self.model or self.model == "your_model_name_here":
            raise ValueError(
                "OLLAMA_MODEL must be set in .env to a locally available Ollama model."
            )
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        options = {
            "temperature": kwargs.pop("temperature", self.temperature),
            "num_ctx": kwargs.pop("num_ctx", self.num_ctx),
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": options,
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=kwargs.pop("timeout", self.timeout),
        )
        response.raise_for_status()
        return str(response.json().get("response", "")).strip()

    def generate_text(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        return self.generate(user_prompt, system=system_prompt, **kwargs)

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        schema: dict | None = None,
        **kwargs: Any,
    ) -> Any:
        if schema:
            kwargs.setdefault("format", "json")
            if system is not None:
                prompt, system = system, prompt
        response_text = self.generate(prompt, system=system, **kwargs)
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(response_text[start : end + 1])
            raise


class LoggingLLMClient:
    """LLM client wrapper that logs raw model outputs as they are produced."""

    def __init__(self, wrapped: LLMClient, logger_name: str = "llm") -> None:
        self.wrapped = wrapped
        self.logger = logging.getLogger(logger_name)
        self.call_count = 0

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        self.call_count += 1
        call_id = self.call_count
        self.logger.info(
            "LLM call %s started (prompt_chars=%s, system_chars=%s)",
            call_id,
            len(prompt),
            len(system or ""),
        )
        response_text = self.wrapped.generate(prompt, system=system, **kwargs)
        self._log_response(call_id, response_text)
        return response_text

    def generate_text(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        return self.generate(user_prompt, system=system_prompt, **kwargs)

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        schema: dict | None = None,
        **kwargs: Any,
    ) -> Any:
        if schema:
            kwargs.setdefault("format", "json")
            if system is not None:
                prompt, system = system, prompt
        response_text = self.generate(prompt, system=system, **kwargs)
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(response_text[start : end + 1])
            raise

    def _log_response(self, call_id: int, response_text: str) -> None:
        self.logger.info(
            "LLM call %s completed (response_chars=%s)\n%s",
            call_id,
            len(response_text),
            response_text,
        )
