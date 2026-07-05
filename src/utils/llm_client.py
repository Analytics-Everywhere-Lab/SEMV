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


class VLLMOpenAIClient:
    """Shared vLLM OpenAI-compatible client used by all SEMV agent components."""

    def __init__(self, env_path: str | None = None) -> None:
        load_env_file(env_path)
        self.base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
        self.api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        self.model = os.getenv("VLLM_MODEL")

        if not self.model or self.model == "your_model_name_here":
            raise ValueError(
                "VLLM_MODEL must be set in .env, e.g. VLLM_MODEL=Qwen/Qwen3.5-9B."
            )

        self.temperature = float(os.getenv("VLLM_TEMPERATURE", "0.0"))
        self.top_p = float(os.getenv("VLLM_TOP_P", "1.0"))
        self.top_k = int(os.getenv("VLLM_TOP_K", "20"))
        self.max_tokens = int(os.getenv("VLLM_MAX_TOKENS", "4096"))
        self.timeout = float(os.getenv("VLLM_TIMEOUT", "120"))
        self.enable_thinking = _env_bool("VLLM_ENABLE_THINKING", False)

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response_format = kwargs.pop("format", None)
        return self._chat(messages, response_format=response_format, **kwargs)

    def generate_with_images(
        self,
        prompt: str,
        image_paths: list[Path],
        system: str | None = None,
        format: str | dict = "json",
        **kwargs: Any,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(path)},
                }
            )

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        return self._chat(messages, response_format=format, **kwargs)

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
            kwargs.setdefault("format", {"type": "json_object"})
        else:
            kwargs.setdefault("format", {"type": "json_object"})

        response_text = self.generate(prompt, system=system, **kwargs)
        return _parse_json_response(response_text)

    def _chat(
        self,
        messages: list[dict[str, Any]],
        response_format: str | dict | None = None,
        **kwargs: Any,
    ) -> str:
        model = kwargs.pop("model", self.model)
        timeout = kwargs.pop("timeout", self.timeout)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": kwargs.pop("top_p", self.top_p),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
        }

        top_k = kwargs.pop("top_k", self.top_k)
        if top_k is not None:
            payload["top_k"] = top_k

        if response_format is not None:
            if response_format == "json":
                payload["response_format"] = {"type": "json_object"}
            elif isinstance(response_format, dict):
                payload["response_format"] = response_format

        payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        extra_body = kwargs.pop("extra_body", None)
        if isinstance(extra_body, dict):
            payload.update(extra_body)

        payload.update(kwargs)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info("Calling vLLM model=%s timeout=%s", model, timeout)
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"vLLM call timed out after {timeout}s. "
                "Check that the vLLM server is running and reachable."
            ) from exc
        response.raise_for_status()
        logger.info("vLLM response received")

        data = response.json()
        message = data["choices"][0]["message"]
        content = message.get("content", "")

        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            ).strip()

        return str(content).strip()


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


def build_llm_client(env_path: str | None = None) -> LLMClient:
    provider = os.getenv("SEMV_LLM_PROVIDER", "vllm").strip().lower()
    if provider != "vllm":
        raise ValueError(f"Unsupported SEMV_LLM_PROVIDER={provider!r}. Use 'vllm'.")
    return VLLMOpenAIClient(env_path=env_path)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _image_data_url(path: Path) -> str:
    import mimetypes

    path = Path(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    image_b64 = _encode_image(path)
    return f"data:{mime_type};base64,{image_b64}"


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
