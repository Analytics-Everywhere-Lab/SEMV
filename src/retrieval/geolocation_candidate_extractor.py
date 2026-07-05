from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.io import project_root
from src.utils.tool_config import retrieval_config


LOCATION_KEYS = {
    "city",
    "country",
    "place",
    "location",
    "candidate_name",
    "address",
    "region",
    "province",
    "state",
}
BLOCKED_KEYS = {
    "chroma_location",
    "sample_aspect_ratio",
    "display_aspect_ratio",
    "color_space",
    "color_transfer",
    "color_primaries",
}
GENERIC_LOCATION_VALUES = {
    "center",
    "left",
    "right",
    "top",
    "bottom",
    "unknown",
    "none",
    "n/a",
}


class GeolocationCandidateExtractor:
    def __init__(self, config: dict | None = None) -> None:
        self.config = retrieval_config(config)
        self.cache_path = project_root() / self.config.get("geocoding_cache_path", "data/cache/geocoding_cache.json")

    def extract(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        if not self.config.get("geolocation_enabled", True):
            return []
        clues = []
        gps_items = []
        for item in evidence:
            gps = _extract_gps(item)
            if gps:
                gps_items.append((item, gps))
            for name in _extract_location_names(item):
                clues.append({"name": name, "item": item})

        candidates: list[EvidenceItem] = []
        for item, gps in gps_items:
            candidates.append(self._candidate_from_gps(item, gps))

        grouped: dict[str, list[EvidenceItem]] = {}
        for clue in clues:
            grouped.setdefault(clue["name"].lower(), []).append(clue["item"])
        for key, source_items in grouped.items():
            name = _title_location(key)
            if not name:
                continue
            if any(candidate.raw_output.get("candidate_name", "").lower() == name.lower() for candidate in candidates):
                continue
            candidates.append(self._candidate_from_name(name, source_items))
        return candidates

    def _candidate_from_gps(self, item: EvidenceItem, gps: tuple[float, float]) -> EvidenceItem:
        suspicious = any("suspicious" in flag or "stripped" in flag for flag in item.uncertainty_flags)
        reliability = 0.65 if suspicious else 0.85
        name = f"GPS {gps[0]:.6f}, {gps[1]:.6f}"
        return self._item(
            candidate_name=name,
            source_items=[item],
            reliability=reliability,
            lat=gps[0],
            lon=gps[1],
            geocoder="metadata_gps",
            confidence_reason="GPS coordinates found in media metadata" + (" with metadata caution flags" if suspicious else ""),
        )

    def _candidate_from_name(self, name: str, source_items: list[EvidenceItem]) -> EvidenceItem:
        source_types = {item.source_type for item in source_items}
        lat = lon = None
        geocoder = "none"
        if self.config.get("geocoding_enabled", False):
            resolved = self._geocode(name)
            if resolved:
                lat, lon = resolved
                geocoder = str(self.config.get("geocoding_provider", "nominatim"))
        text_sources = {"ocr", "web_article", "news_article", "factcheck_article"}
        visual_sources = {"frame_analysis", "visual_vqa", "visual_caption"}
        if (
            len(source_types) >= 2
            and any(source in source_types for source in text_sources)
            and any(source in source_types for source in visual_sources)
        ):
            reliability = 0.75
            reason = "Location clue corroborated across OCR/article and visual evidence"
        elif source_types <= {"frame_analysis", "visual_vqa", "visual_caption"}:
            reliability = 0.45
            reason = "Location clue inferred from visual analysis only"
        elif source_types <= {"ocr"}:
            reliability = 0.55
            reason = "Ambiguous location clue found in OCR only"
        else:
            reliability = 0.60
            reason = "Location clue found in media or source evidence"
        if lat is not None and lon is not None:
            reliability = min(0.85, reliability + 0.10)
            reason += "; geocoder returned coordinates"
        return self._item(name, source_items, reliability, lat, lon, geocoder, reason)

    def _item(
        self,
        candidate_name: str,
        source_items: list[EvidenceItem],
        reliability: float,
        lat: float | None,
        lon: float | None,
        geocoder: str,
        confidence_reason: str,
    ) -> EvidenceItem:
        source_ids = [item.evidence_id for item in source_items]
        clues = [item.content[:240] for item in source_items]
        evidence_id = f"geo_candidate_{stable_hash_text(candidate_name + ''.join(source_ids))}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="geolocation_candidate",
            source=source_items[0].source if source_items else "geolocation",
            title="Candidate location",
            content=f"Candidate location: {candidate_name}",
            reliability=reliability,
            relevance=0.90,
            media_path=source_items[0].media_path if source_items else None,
            raw_output={
                "candidate_name": candidate_name,
                "lat": lat,
                "lon": lon,
                "source_evidence_ids": source_ids,
                "source_clues": clues,
                "geocoder": geocoder,
                "confidence_reason": confidence_reason,
            },
            metadata={"candidate_name": candidate_name, "lat": lat, "lon": lon, "source_clues": source_ids},
            supports_claim_types=["where"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="geolocation_candidate",
                source=source_items[0].source if source_items else "geolocation",
                retrieval_method="media_clue_extraction",
                metadata={"source_evidence_ids": source_ids, "geocoder": geocoder},
            ),
        )

    def _geocode(self, name: str) -> tuple[float, float] | None:
        cache = self._load_cache()
        if name in cache:
            value = cache[name]
            return (float(value["lat"]), float(value["lon"])) if value else None
        if self.config.get("geocoding_provider", "nominatim") != "nominatim":
            return None
        try:
            time.sleep(float(self.config.get("geocoding_rate_limit_sec", 1.0)))
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": name, "format": "json", "limit": 1},
                headers={"User-Agent": "SEMV geolocation extractor"},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            if data:
                result = {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
                cache[name] = result
                self._save_cache(cache)
                return result["lat"], result["lon"]
        except Exception:
            pass
        cache[name] = None
        self._save_cache(cache)
        return None

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, indent=2, default=str) + "\n", encoding="utf-8")


def _extract_gps(item: EvidenceItem) -> tuple[float, float] | None:
    text = json.dumps({"metadata": item.metadata, "raw": item.raw_output}, default=str)
    pairs = re.findall(
        r"(?:lat(?:itude)?|GPSLatitude)['\"\s:=]+(-?\d+(?:\.\d+)?).*?"
        r"(?:lon(?:gitude)?|GPSLongitude)['\"\s:=]+(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if pairs:
        return float(pairs[0][0]), float(pairs[0][1])
    coords = re.findall(r"(-?\d{1,2}\.\d{3,})\s*,\s*(-?\d{1,3}\.\d{3,})", text)
    if coords:
        lat, lon = map(float, coords[0])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    return None


def _extract_location_names(item: EvidenceItem) -> list[str]:
    metadata = item.metadata or {}
    raw = item.raw_output or {}
    values = []
    if item.source_type in {"frame_analysis", "visual_vqa", "visual_caption"}:
        values.extend(map(str, raw.get("location_clues") or []))
    if item.source_type in {"ocr", "asr", "web_article", "news_article", "factcheck_article", "reverse_image_web_candidate"}:
        values.extend(_named_location_phrases(f"{item.title or ''} {item.content}"))
    if item.source_type == "geolocation_candidate":
        name = raw.get("candidate_name") or metadata.get("candidate_name")
        if name:
            values.append(str(name))
    values.extend(_collect_location_values(metadata))
    values.extend(_collect_location_values(raw))
    cleaned = []
    seen = set()
    for value in values:
        name = _title_location(value)
        key = name.lower()
        if name and key not in seen:
            cleaned.append(name)
            seen.add(key)
    return cleaned


def _collect_location_values(obj: Any) -> list[str]:
    values: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()
            if key_lower in BLOCKED_KEYS:
                continue
            if key_lower in LOCATION_KEYS and isinstance(value, str):
                values.append(value)
                continue
            if isinstance(value, (dict, list)):
                values.extend(_collect_location_values(value))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(_collect_location_values(item))
    return values


def _named_location_phrases(text: str) -> list[str]:
    tokens = re.findall(r"\b(?:[A-Z][a-zA-Z'’-]+(?:\s+[A-Z][a-zA-Z'’-]+){0,3})\b", text)
    stop = {"Visible Text", "Candidate Location", "Web Article", "Local Visual", "GPS"}
    return [token for token in tokens if token not in stop and len(token) > 2]


def _title_location(value: str) -> str:
    cleaned = " ".join(str(value).replace("Candidate location:", "").split())
    cleaned = cleaned.strip(" .,:;|-_")
    normalized = cleaned.casefold()
    if not cleaned or len(cleaned) < 3 or len(cleaned) > 100:
        return ""
    if normalized in GENERIC_LOCATION_VALUES:
        return ""
    return cleaned.title() if cleaned.islower() else cleaned
