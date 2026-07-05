from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from src.utils.io import read_yaml


_DEFAULT_TOOLS_CONFIG: dict[str, Any] = {
    "retrieval": {
        "live_web_enabled": False,
        "cached_evidence_enabled": True,
        "manual_case_evidence_enabled": True,
        "free_web_search_enabled": False,
        "max_web_results_per_claim": 5,
        "max_downloaded_candidate_images": 20,
        "web_image_compare_enabled": True,
        "web_image_min_width": 160,
        "web_image_min_height": 120,
        "geolocation_enabled": True,
        "geocoding_enabled": False,
        "geocoding_provider": "nominatim",
        "geocoding_cache_path": "data/cache/geocoding_cache.json",
    },
    "media": {
        "enable_ffmpeg_keyframes": True,
        "keyframe_strategy": "scene_detect",
        "max_keyframes_per_video": 8,
        "deduplicate_keyframes": True,
        "enable_vlm_adapter": True,
        "vlm_provider": "ollama",
        "vlm_model": "llava",
        "vlm_timeout_sec": 120,
        "enable_ocr_adapter": True,
        "ocr_engine": "easyocr",
        "ocr_languages": ["en"],
        "enable_asr_adapter": True,
        "asr_engine": "faster_whisper",
        "asr_model_size": "base",
        "asr_language": None,
        "enable_forensic_adapter": True,
        "forensic_engine": "basic",
        "enable_local_reverse_search": True,
        "local_reverse_methods": ["phash", "clip_faiss"],
        "visual_index_dir": "data/visual_index",
        "phash_threshold": 10,
        "clip_similarity_threshold": 0.84,
        "clip_model_name": "ViT-B-32",
        "clip_pretrained": "openai",
    },
}


def load_tools_config() -> dict[str, Any]:
    config = deepcopy(_DEFAULT_TOOLS_CONFIG)
    _deep_update(config, read_yaml("configs/tools.yaml"))
    media = config.setdefault("media", {})
    retrieval = config.setdefault("retrieval", {})

    _env_bool(media, "enable_vlm_adapter", "SEMV_ENABLE_VLM")
    _env_str(media, "vlm_provider", "SEMV_VLM_PROVIDER")
    _env_str(media, "vlm_model", "SEMV_VLM_MODEL")
    _env_bool(media, "enable_ocr_adapter", "SEMV_ENABLE_OCR")
    _env_bool(media, "enable_asr_adapter", "SEMV_ENABLE_ASR")
    _env_bool(media, "enable_forensic_adapter", "SEMV_ENABLE_FORENSICS")
    _env_bool(media, "enable_local_reverse_search", "SEMV_ENABLE_LOCAL_REVERSE")
    _env_bool(retrieval, "free_web_search_enabled", "SEMV_ENABLE_FREE_WEB_SEARCH")
    return config


def media_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return (config or load_tools_config()).get("media", {})


def retrieval_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return (config or load_tools_config()).get("retrieval", {})


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _env_bool(target: dict[str, Any], key: str, env_name: str) -> None:
    raw = os.getenv(env_name)
    if raw is not None:
        target[key] = raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(target: dict[str, Any], key: str, env_name: str) -> None:
    raw = os.getenv(env_name)
    if raw:
        target[key] = raw
