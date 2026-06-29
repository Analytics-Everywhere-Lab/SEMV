from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.llm_client import OllamaLLMClient


def test_ollama_model_required_and_placeholder_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("OLLAMA_MODEL=your_model_name_here\n", encoding="utf-8")

    with pytest.raises(ValueError):
        OllamaLLMClient(env_path=str(env_path))


def test_source_does_not_hardcode_known_model_names():
    root = Path(__file__).resolve().parents[1] / "src"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))

    assert "llama3.1" not in text
    assert "qwen2.5" not in text
    assert "mistral" not in text.lower()
