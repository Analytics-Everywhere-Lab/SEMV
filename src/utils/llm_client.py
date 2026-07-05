from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import requests

from src.utils.env_loader import load_env_file


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        ...

    def generate_with_images(
        self,
        prompt: str,
        image_paths: list[Path],
        system: str | None = None,
        format: str | dict = "json",
        **kwargs: Any,
    ) -> str:
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


logger = logging.getLogger("run_case")


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
        response_format = kwargs.pop("format", None)
        payload = {
            "model": kwargs.pop("model", self.model),
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": options,
        }
        if response_format is not None:
            payload["format"] = response_format
        timeout = kwargs.pop("timeout", self.timeout)
        logger.info("Calling Ollama model=%s timeout=%s", self.model, timeout)
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama call timed out after {timeout}s. "
                "Check that Ollama is running and the configured model is pulled."
            ) from exc
        response.raise_for_status()
        logger.info("Ollama response received")
        return str(response.json().get("response", "")).strip()

    def generate_with_images(
        self,
        prompt: str,
        image_paths: list[Path],
        system: str | None = None,
        format: str | dict = "json",
        **kwargs: Any,
    ) -> str:
        options = {
            "temperature": kwargs.pop("temperature", self.temperature),
            "num_ctx": kwargs.pop("num_ctx", self.num_ctx),
        }
        payload = {
            "model": kwargs.pop("model", self.model),
            "prompt": prompt,
            "system": system,
            "stream": False,
            "format": format,
            "images": [_encode_image(path) for path in image_paths],
            "options": options,
        }
        timeout = kwargs.pop("timeout", self.timeout)
        logger.info("Calling Ollama model=%s timeout=%s", self.model, timeout)
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama call timed out after {timeout}s. "
                "Check that Ollama is running and the configured model is pulled."
            ) from exc
        response.raise_for_status()
        logger.info("Ollama response received")
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
        response_text = self.generate(prompt, system=system, **kwargs)
        return _parse_json_response(response_text)


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

    def generate_with_images(
        self,
        prompt: str,
        image_paths: list[Path],
        system: str | None = None,
        format: str | dict = "json",
        **kwargs: Any,
    ) -> str:
        self.call_count += 1
        call_id = self.call_count
        self.logger.info(
            "LLM image call %s started (prompt_chars=%s, images=%s)",
            call_id,
            len(prompt),
            len(image_paths),
        )
        if not hasattr(self.wrapped, "generate_with_images"):
            raise NotImplementedError("Wrapped LLM client does not support image generation")
        response_text = self.wrapped.generate_with_images(
            prompt, image_paths, system=system, format=format, **kwargs
        )
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
        if hasattr(self.wrapped, "generate_json"):
            self.call_count += 1
            call_id = self.call_count
            self.logger.info(
                "LLM JSON call %s started (prompt_chars=%s, system_chars=%s)",
                call_id,
                len(prompt),
                len(system or ""),
            )
            result = self.wrapped.generate_json(prompt, system=system, schema=schema, **kwargs)
            self._log_response(call_id, json.dumps(result, default=str))
            return result
        if schema:
            kwargs.setdefault("format", "json")
        response_text = self.generate(prompt, system=system, **kwargs)
        return _parse_json_response(response_text)

    def _log_response(self, call_id: int, response_text: str) -> None:
        self.logger.info(
            "LLM call %s completed (response_chars=%s)\n%s",
            call_id,
            len(response_text),
            response_text,
        )


def _encode_image(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _parse_json_response(response_text: str) -> Any:
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(response_text[start : end + 1])
        raise
