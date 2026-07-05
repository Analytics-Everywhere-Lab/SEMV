from __future__ import annotations

from src.utils.tool_config import load_tools_config


def test_forensic_env_overrides(monkeypatch):
    monkeypatch.setenv("SEMV_FORENSIC_ENGINE", "trufor")
    monkeypatch.setenv("SEMV_FORENSIC_DEEP_BACKEND", "trufor")
    monkeypatch.setenv("SEMV_FORENSIC_DEVICE", "cpu")
    monkeypatch.setenv("SEMV_TRUFOR_REPO_DIR", "/tmp/trufor_repo")
    monkeypatch.setenv("SEMV_TRUFOR_WEIGHTS", "/tmp/trufor_repo/weights.pth.tar")
    monkeypatch.setenv("SEMV_TRUFOR_PYTHON", "/opt/trufor/bin/python")
    monkeypatch.setenv("SEMV_TRUFOR_TIMEOUT_SEC", "42")
    monkeypatch.setenv("SEMV_TRUFOR_EXPERIMENT", "trufor_custom")

    config = load_tools_config()["media"]

    assert config["forensic_engine"] == "trufor"
    assert config["forensic_deep_backend"] == "trufor"
    assert config["forensic_device"] == "cpu"
    assert config["forensic_external_repo_dir"] == "/tmp/trufor_repo"
    assert config["forensic_trufor_weights"] == "/tmp/trufor_repo/weights.pth.tar"
    assert config["forensic_trufor_python"] == "/opt/trufor/bin/python"
    assert config["forensic_trufor_timeout_sec"] == 42
    assert config["forensic_trufor_experiment"] == "trufor_custom"


def test_forensic_defaults_without_env(monkeypatch):
    for var in (
        "SEMV_FORENSIC_ENGINE",
        "SEMV_FORENSIC_DEEP_BACKEND",
        "SEMV_FORENSIC_DEVICE",
        "SEMV_TRUFOR_REPO_DIR",
        "SEMV_TRUFOR_WEIGHTS",
        "SEMV_TRUFOR_PYTHON",
        "SEMV_TRUFOR_TIMEOUT_SEC",
        "SEMV_TRUFOR_EXPERIMENT",
    ):
        monkeypatch.delenv(var, raising=False)

    config = load_tools_config()["media"]

    assert config["forensic_engine"] == "basic"
    assert config["forensic_deep_backend"] == "trufor"
    assert config["forensic_trufor_python"] == "python"
    assert config["forensic_trufor_timeout_sec"] == 300
