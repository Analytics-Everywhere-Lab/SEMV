from __future__ import annotations

import json
import mimetypes
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from src.schemas.case_schema import MediaItem
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import sha256_file, stable_hash_text


class MetadataExtractor:
    def extract(self, media: MediaItem, base_dir: Path | None = None) -> list[EvidenceItem]:
        path = media.resolved_path(base_dir)
        basic = self.basic_metadata(media, base_dir)
        items: list[EvidenceItem] = []

        exiftool = shutil.which("exiftool")
        if exiftool and path.exists():
            payload, flag = self._run_json([exiftool, "-json", str(path)])
            parsed = payload[0] if isinstance(payload, list) and payload else {}
            flags = list(basic.get("uncertainty_flags", []))
            if flag:
                flags.append(flag)
            flags.extend(self._metadata_quality_flags(parsed, basic, "exiftool"))
            items.append(
                self._item(
                    media=media,
                    source_type="metadata_exiftool",
                    method="exiftool -json",
                    title="ExifTool metadata inspection",
                    parsed={"basic": basic, "exiftool": parsed},
                    flags=flags,
                    reliability=0.80 if parsed else 0.65,
                )
            )
        else:
            reason = "exiftool_missing" if path.exists() else "metadata_missing"
            items.append(self._uncertainty_item(media, reason, "ExifTool metadata unavailable", basic))

        ffprobe = shutil.which("ffprobe")
        if ffprobe and path.exists():
            payload, flag = self._run_json(
                [
                    ffprobe,
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(path),
                ]
            )
            parsed = payload if isinstance(payload, dict) else {}
            flags = list(basic.get("uncertainty_flags", []))
            if flag:
                flags.append(flag)
            flags.extend(self._metadata_quality_flags(parsed, basic, "ffprobe"))
            items.append(
                self._item(
                    media=media,
                    source_type="metadata_ffprobe",
                    method="ffprobe json",
                    title="FFprobe stream metadata inspection",
                    parsed={"basic": basic, "ffprobe": parsed},
                    flags=flags,
                    reliability=0.80 if parsed else 0.65,
                )
            )
        else:
            reason = "ffprobe_missing" if path.exists() else "metadata_missing"
            items.append(self._uncertainty_item(media, reason, "FFprobe metadata unavailable", basic))

        return items

    def basic_metadata(self, media: MediaItem, base_dir: Path | None = None) -> dict[str, Any]:
        path = media.resolved_path(base_dir)
        metadata: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "declared_media_type": media.media_type,
            "description": media.description,
        }
        if not path.exists():
            metadata["uncertainty_flags"] = ["media_file_missing", "metadata_missing"]
            return metadata

        mime_type, _ = mimetypes.guess_type(path)
        metadata.update(
            {
                "mime_type": mime_type or "application/octet-stream",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

        if self._looks_like_image(media, mime_type):
            try:
                with Image.open(path) as image:
                    exif = image.getexif()
                    metadata.update(
                        {
                            "width": image.width,
                            "height": image.height,
                            "format": image.format,
                            "mode": image.mode,
                            "orientation": exif.get(274),
                            "software": exif.get(305),
                            "datetime_original": exif.get(36867),
                            "camera_model": exif.get(272),
                            "color_profile": "icc" if image.info.get("icc_profile") else None,
                        }
                    )
            except Exception as exc:  # pragma: no cover - corrupt media depends on fixtures
                metadata.setdefault("uncertainty_flags", []).append(
                    f"image_metadata_unreadable:{exc.__class__.__name__}"
                )
        return metadata

    @staticmethod
    def _looks_like_image(media: MediaItem, mime_type: str | None) -> bool:
        return media.media_type == "image" or bool(mime_type and mime_type.startswith("image/"))

    @staticmethod
    def _run_json(command: list[str]) -> tuple[Any, str | None]:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
            return json.loads(result.stdout or "{}"), None
        except json.JSONDecodeError:
            return {}, "metadata_json_parse_failed"
        except Exception as exc:  # pragma: no cover - external binary behavior varies
            return {}, f"metadata_tool_failed:{exc.__class__.__name__}"

    @staticmethod
    def _metadata_quality_flags(parsed: dict[str, Any], basic: dict[str, Any], engine: str) -> list[str]:
        flags: list[str] = []
        if not parsed:
            flags.append("metadata_missing")
        text = json.dumps(parsed, default=str).lower()
        if "gps" not in text:
            flags.append("gps_missing")
        if "creation" not in text and "datetimeoriginal" not in text and not basic.get("datetime_original"):
            flags.append("creation_time_missing")
        if engine == "exiftool":
            software = str(parsed.get("Software") or basic.get("software") or "").lower()
            if any(token in software for token in ("photoshop", "gimp", "snapseed", "canva")):
                flags.append("software_tag_suspicious")
            if not parsed and basic.get("exists"):
                flags.append("metadata_stripped")
        return sorted(set(flags))

    def _item(
        self,
        media: MediaItem,
        source_type: str,
        method: str,
        title: str,
        parsed: dict[str, Any],
        flags: list[str],
        reliability: float,
    ) -> EvidenceItem:
        evidence_id = f"{source_type}_{stable_hash_text(media.path + method + str(parsed))}"
        summary = self._summarize_metadata(source_type, parsed)
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type=source_type,  # type: ignore[arg-type]
            source=media.path,
            title=title,
            content=summary,
            reliability=reliability,
            relevance=0.65,
            media_path=media.path,
            metadata=parsed,
            raw_output=parsed,
            uncertainty_flags=flags,
            supports_claim_types=["where", "when", "what", "authenticity"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type=source_type,  # type: ignore[arg-type]
                source=media.path,
                retrieval_method=method,
                metadata={"flags": flags},
            ),
        )

    def _uncertainty_item(
        self, media: MediaItem, flag: str, title: str, basic: dict[str, Any]
    ) -> EvidenceItem:
        evidence_id = f"uncertainty_{stable_hash_text(media.path + flag)}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="synthetic_uncertainty",
            source=media.path,
            title=title,
            content=f"{title} for {media.path}; adapter did not run ({flag}).",
            reliability=0.25,
            relevance=0.45,
            media_path=media.path,
            metadata={"basic": basic},
            uncertainty_flags=[flag],
            supports_claim_types=["where", "when", "what", "authenticity"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="synthetic_uncertainty",
                source=media.path,
                retrieval_method="local_capability_check",
                metadata={"adapter": "metadata", "flag": flag},
            ),
        )

    @staticmethod
    def _summarize_metadata(source_type: str, parsed: dict[str, Any]) -> str:
        basic = parsed.get("basic", {})
        details = parsed.get("exiftool") or parsed.get("ffprobe") or {}
        parts = [
            f"{source_type} inspection for {basic.get('path', 'media')}.",
            f"File exists: {bool(basic.get('exists'))}.",
        ]
        if basic.get("sha256"):
            parts.append(f"SHA256: {basic['sha256']}.")
        if basic.get("width") and basic.get("height"):
            parts.append(f"Image dimensions: {basic['width']}x{basic['height']}.")
        if details.get("format"):
            fmt = details["format"]
            parts.append(f"Container: {fmt.get('format_name', 'unknown')}; duration: {fmt.get('duration', 'unknown')}.")
        streams = details.get("streams") or []
        if streams:
            codecs = ", ".join(filter(None, [stream.get("codec_name") for stream in streams]))
            parts.append(f"Streams/codecs: {codecs or 'unknown'}.")
        return " ".join(parts)
