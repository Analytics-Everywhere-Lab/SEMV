from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from PIL import Image

from src.schemas.case_schema import MediaItem
from src.utils.hashing import sha256_file


class MetadataExtractor:
    def extract(self, media: MediaItem, base_dir: Path | None = None) -> dict[str, Any]:
        path = media.resolved_path(base_dir)
        metadata: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "declared_media_type": media.media_type,
            "description": media.description,
        }
        if not path.exists():
            metadata["uncertainty_flags"] = ["media_file_missing"]
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
                    metadata["width"] = image.width
                    metadata["height"] = image.height
                    metadata["format"] = image.format
            except Exception as exc:  # pragma: no cover - depends on corrupt inputs
                metadata.setdefault("uncertainty_flags", []).append(
                    f"image_metadata_unreadable:{exc.__class__.__name__}"
                )
        return metadata

    @staticmethod
    def _looks_like_image(media: MediaItem, mime_type: str | None) -> bool:
        return media.media_type == "image" or bool(mime_type and mime_type.startswith("image/"))
