from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from src.processing.deep_forensics.base import DeepForensicBackend, DeepForensicResult
from src.utils.io import project_root


class TruForBackend(DeepForensicBackend):
    model_name = "trufor"

    def __init__(self, config: dict) -> None:
        root = project_root()
        self.config = config
        self.repo_dir = _resolve_path(config.get("forensic_external_repo_dir"), root)
        self.weights = _resolve_path(config.get("forensic_trufor_weights"), root)
        self.experiment = str(config.get("forensic_trufor_experiment", "trufor_ph3"))
        self.device = str(config.get("forensic_device", "cuda"))
        self.threshold = float(config.get("forensic_manipulation_threshold", 0.50))
        self.min_confidence = float(config.get("forensic_min_confidence", 0.30))

    def analyze_images(self, image_paths: list[Path], output_dir: Path) -> list[DeepForensicResult]:
        self._validate()
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[DeepForensicResult] = []
        for image_path in image_paths:
            target_out = output_dir / image_path.stem
            target_out.mkdir(parents=True, exist_ok=True)

            try:
                self._run_trufor(image_path=image_path, output_dir=target_out)
                result = self._read_result(image_path=image_path, output_dir=target_out)
            except Exception as exc:
                result = DeepForensicResult(
                    target_path=str(image_path),
                    model_name=self.model_name,
                    flags=["deep_forensic_inference_failed"],
                    raw_output={"error": str(exc)},
                )

            results.append(result)

        return results

    def _validate(self) -> None:
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"TruFor repo dir not found: {self.repo_dir}")
        if not (self.repo_dir / "test.py").exists():
            raise FileNotFoundError(f"TruFor test.py not found in: {self.repo_dir}")
        if not self.weights.exists():
            raise FileNotFoundError(f"TruFor weights not found: {self.weights}")

    def _run_trufor(self, image_path: Path, output_dir: Path) -> None:
        gpu_arg = "0" if self.device.startswith("cuda") else "-1"
        cmd = [
            "python",
            "test.py",
            "-g",
            gpu_arg,
            "-in",
            str(image_path.resolve()),
            "-out",
            str(output_dir.resolve()),
            "-exp",
            self.experiment,
            "TEST.MODEL_FILE",
            str(self.weights.resolve()),
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(self.repo_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"TruFor failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

    def _read_result(self, image_path: Path, output_dir: Path) -> DeepForensicResult:
        npz_files = sorted(output_dir.rglob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No TruFor .npz output found in {output_dir}")

        npz_path = npz_files[0]
        data = np.load(npz_path)

        anomaly = np.asarray(data["map"], dtype=np.float32) if "map" in data else None
        conf = np.asarray(data["conf"], dtype=np.float32) if "conf" in data else None
        score = float(np.asarray(data["score"]).reshape(-1)[0]) if "score" in data else None

        anomaly_map_path = None
        confidence_map_path = None
        overlay_path = None

        if anomaly is not None:
            anomaly_map_path = str(output_dir / f"{image_path.stem}_trufor_anomaly.png")
            _save_grayscale_map(anomaly, Path(anomaly_map_path))

            overlay_path = str(output_dir / f"{image_path.stem}_trufor_overlay.png")
            _save_overlay(image_path, anomaly, Path(overlay_path))

        if conf is not None:
            confidence_map_path = str(output_dir / f"{image_path.stem}_trufor_confidence.png")
            _save_grayscale_map(conf, Path(confidence_map_path))

        flags: list[str] = []
        if score is not None and score >= self.threshold:
            flags.append("deep_forensic_high_manipulation_score")

        mean_confidence = float(np.mean(conf)) if conf is not None else None
        if mean_confidence is not None and mean_confidence < self.min_confidence:
            flags.append("deep_forensic_low_confidence")

        manipulated_area_ratio = None
        max_anomaly = None
        mean_anomaly = None
        if anomaly is not None:
            max_anomaly = float(np.max(anomaly))
            mean_anomaly = float(np.mean(anomaly))
            manipulated_area_ratio = float(np.mean(anomaly >= self.threshold))

        return DeepForensicResult(
            target_path=str(image_path),
            model_name=self.model_name,
            manipulation_score=score,
            anomaly_map_path=anomaly_map_path,
            confidence_map_path=confidence_map_path,
            heatmap_overlay_path=overlay_path,
            manipulated_area_ratio=manipulated_area_ratio,
            max_anomaly=max_anomaly,
            mean_anomaly=mean_anomaly,
            mean_confidence=mean_confidence,
            flags=flags,
            raw_output={"npz_path": str(npz_path)},
        )


def _resolve_path(value: str | None, root: Path) -> Path:
    if not value:
        return root
    path = Path(value)
    return path if path.is_absolute() else root / path


def _save_grayscale_map(arr: np.ndarray, path: Path) -> None:
    arr = np.nan_to_num(arr)
    arr = arr - arr.min()
    denom = arr.max() if arr.max() > 0 else 1.0
    arr = (arr / denom * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_overlay(image_path: Path, anomaly: np.ndarray, path: Path) -> None:
    base = Image.open(image_path).convert("RGB")
    anomaly_img = Image.fromarray(_normalize_uint8(anomaly)).resize(base.size)
    anomaly_rgb = Image.merge("RGB", (anomaly_img, Image.new("L", base.size), Image.new("L", base.size)))
    overlay = Image.blend(base, anomaly_rgb, alpha=0.35)
    overlay.save(path)


def _normalize_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr)
    arr = arr - arr.min()
    denom = arr.max() if arr.max() > 0 else 1.0
    return (arr / denom * 255).astype(np.uint8)
