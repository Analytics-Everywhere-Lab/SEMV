from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.env_loader import get_bool_env, get_int_env
from src.utils.llm_client import VLLMOpenAIClient


def test_vllm_model_required_and_placeholder_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_MODEL", "your_model_name_here")

    with pytest.raises(ValueError):
        VLLMOpenAIClient()


def test_source_does_not_hardcode_known_model_names():
    root = Path(__file__).resolve().parents[1] / "src"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))

    assert "llama3.1" not in text
    assert "qwen2.5" not in text
    assert "mistral" not in text.lower()


def test_env_helpers_parse_ints_and_defaults(monkeypatch):
    monkeypatch.delenv("SEMV_TEST_INT", raising=False)
    assert get_int_env("SEMV_TEST_INT", 2) == 2

    monkeypatch.setenv("SEMV_TEST_INT", "4")
    assert get_int_env("SEMV_TEST_INT", 2) == 4

    monkeypatch.setenv("SEMV_TEST_INT", "")
    assert get_int_env("SEMV_TEST_INT", 2) == 2

    monkeypatch.setenv("SEMV_TEST_INT", "not-an-int")
    assert get_int_env("SEMV_TEST_INT", 2) == 2


def test_env_helpers_parse_bools_and_defaults(monkeypatch):
    monkeypatch.delenv("SEMV_TEST_BOOL", raising=False)
    assert get_bool_env("SEMV_TEST_BOOL", True) is True
    assert get_bool_env("SEMV_TEST_BOOL", False) is False

    for value in ["1", "true", "yes", "y", "on", " TRUE "]:
        monkeypatch.setenv("SEMV_TEST_BOOL", value)
        assert get_bool_env("SEMV_TEST_BOOL", False) is True

    for value in ["0", "false", "no", "off", ""]:
        monkeypatch.setenv("SEMV_TEST_BOOL", value)
        assert get_bool_env("SEMV_TEST_BOOL", True) is False
