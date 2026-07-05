from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.processing.deep_forensics.trufor_backend import TruForBackend


def _config(tmp_path: Path, **overrides) -> dict:
    config = {
        "forensic_external_repo_dir": str(tmp_path / "TruFor_train_test"),
        "forensic_trufor_weights": str(tmp_path / "TruFor_train_test" / "pretrained_models" / "trufor.pth.tar"),
        "forensic_trufor_experiment": "trufor_ph3",
        "forensic_device": "cpu",
        "forensic_manipulation_threshold": 0.50,
        "forensic_min_confidence": 0.30,
    }
    config.update(overrides)
    return config


def test_trufor_backend_raises_when_repo_missing(tmp_path):
    backend = TruForBackend(_config(tmp_path))
    with pytest.raises(FileNotFoundError):
        backend.analyze_images([tmp_path / "img.jpg"], tmp_path / "out")


def test_trufor_backend_raises_when_weights_missing(tmp_path):
    repo_dir = tmp_path / "TruFor_train_test"
    repo_dir.mkdir(parents=True)
    (repo_dir / "test.py").write_text("# stub")

    backend = TruForBackend(_config(tmp_path))
    with pytest.raises(FileNotFoundError):
        backend.analyze_images([tmp_path / "img.jpg"], tmp_path / "out")


def test_trufor_backend_parses_npz_result(tmp_path, monkeypatch):
    from PIL import Image

    repo_dir = tmp_path / "TruFor_train_test"
    repo_dir.mkdir(parents=True)
    (repo_dir / "test.py").write_text("# stub")

    weights_path = repo_dir / "pretrained_models" / "trufor.pth.tar"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"fake")

    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), "white").save(image_path)

    backend = TruForBackend(_config(tmp_path))

    def fake_run_trufor(self, image_path, output_dir):
        anomaly = np.random.rand(32, 32).astype(np.float32)
        conf = np.random.rand(32, 32).astype(np.float32)
        np.savez(output_dir / "result.npz", map=anomaly, conf=conf, score=np.array([0.91]), imgsize=(32, 32))

    monkeypatch.setattr(TruForBackend, "_run_trufor", fake_run_trufor)

    results = backend.analyze_images([image_path], tmp_path / "out")

    assert len(results) == 1
    result = results[0]
    assert result.manipulation_score == pytest.approx(0.91)
    assert result.anomaly_map_path is not None
    assert Path(result.anomaly_map_path).exists()
    assert result.confidence_map_path is not None
    assert Path(result.confidence_map_path).exists()
    assert "deep_forensic_high_manipulation_score" in result.flags


def test_trufor_backend_uses_configured_python_and_timeout(tmp_path, monkeypatch):
    from PIL import Image

    repo_dir = tmp_path / "TruFor_train_test"
    repo_dir.mkdir(parents=True)
    (repo_dir / "test.py").write_text("# stub")

    weights_path = repo_dir / "pretrained_models" / "trufor.pth.tar"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"fake")

    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), "white").save(image_path)

    backend = TruForBackend(
        _config(tmp_path, forensic_trufor_python="/opt/trufor/bin/python", forensic_trufor_timeout_sec=42)
    )

    captured = {}

    class FakeCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_subprocess_run(cmd, cwd, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        return FakeCompletedProcess()

    monkeypatch.setattr("src.processing.deep_forensics.trufor_backend.subprocess.run", fake_subprocess_run)

    try:
        backend._run_trufor(image_path=image_path, output_dir=tmp_path / "out")
    except Exception:
        pass

    assert captured["cmd"][0] == "/opt/trufor/bin/python"
    assert captured["timeout"] == 42


def test_trufor_backend_save_maps_false_skips_png_but_keeps_stats(tmp_path, monkeypatch):
    from PIL import Image

    repo_dir = tmp_path / "TruFor_train_test"
    repo_dir.mkdir(parents=True)
    (repo_dir / "test.py").write_text("# stub")

    weights_path = repo_dir / "pretrained_models" / "trufor.pth.tar"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"fake")

    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), "white").save(image_path)

    backend = TruForBackend(_config(tmp_path, forensic_save_maps=False))

    def fake_run_trufor(self, image_path, output_dir):
        anomaly = np.random.rand(32, 32).astype(np.float32)
        conf = np.random.rand(32, 32).astype(np.float32)
        np.savez(output_dir / "result.npz", map=anomaly, conf=conf, score=np.array([0.91]), imgsize=(32, 32))

    monkeypatch.setattr(TruForBackend, "_run_trufor", fake_run_trufor)

    results = backend.analyze_images([image_path], tmp_path / "out")

    assert len(results) == 1
    result = results[0]
    assert result.manipulation_score == pytest.approx(0.91)
    assert result.anomaly_map_path is None
    assert result.confidence_map_path is None
    assert result.heatmap_overlay_path is None
    assert result.max_anomaly is not None
    assert result.mean_anomaly is not None
    assert result.mean_confidence is not None
    assert "deep_forensic_high_manipulation_score" in result.flags
    saved_pngs = list((tmp_path / "out").rglob("*.png"))
    assert saved_pngs == []


def test_trufor_backend_marks_inference_failed_on_subprocess_error(tmp_path, monkeypatch):
    from PIL import Image

    repo_dir = tmp_path / "TruFor_train_test"
    repo_dir.mkdir(parents=True)
    (repo_dir / "test.py").write_text("# stub")

    weights_path = repo_dir / "pretrained_models" / "trufor.pth.tar"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"fake")

    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), "white").save(image_path)

    backend = TruForBackend(_config(tmp_path))

    def fake_run_trufor(self, image_path, output_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(TruForBackend, "_run_trufor", fake_run_trufor)

    results = backend.analyze_images([image_path], tmp_path / "out")

    assert len(results) == 1
    assert results[0].manipulation_score is None
    assert "deep_forensic_inference_failed" in results[0].flags
