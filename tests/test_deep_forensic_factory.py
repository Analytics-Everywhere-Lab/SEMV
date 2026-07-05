from __future__ import annotations

import pytest

from src.processing.deep_forensics.factory import get_deep_forensic_backend
from src.processing.deep_forensics.trufor_backend import TruForBackend


def test_factory_returns_trufor_backend_by_default(tmp_path):
    backend = get_deep_forensic_backend({"forensic_deep_backend": "trufor"})
    assert isinstance(backend, TruForBackend)


def test_factory_raises_for_unknown_backend():
    with pytest.raises(ValueError, match="unknown_deep_forensic_backend:not_a_real_backend"):
        get_deep_forensic_backend({"forensic_deep_backend": "not_a_real_backend"})


def test_factory_respects_env_override(monkeypatch):
    from src.utils.tool_config import load_tools_config, media_config

    monkeypatch.setenv("SEMV_FORENSIC_DEEP_BACKEND", "not_a_real_backend")
    config = media_config(load_tools_config())
    assert config["forensic_deep_backend"] == "not_a_real_backend"

    with pytest.raises(ValueError, match="unknown_deep_forensic_backend:not_a_real_backend"):
        get_deep_forensic_backend(config)
