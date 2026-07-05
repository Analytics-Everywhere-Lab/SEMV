from __future__ import annotations

from typing import Any

from src.processing.deep_forensics.base import DeepForensicBackend


def get_deep_forensic_backend(config: dict[str, Any]) -> DeepForensicBackend:
    backend_name = str(config.get("forensic_deep_backend", "trufor"))

    if backend_name == "trufor":
        from src.processing.deep_forensics.trufor_backend import TruForBackend

        return TruForBackend(config)

    raise ValueError(f"unknown_deep_forensic_backend:{backend_name}")
