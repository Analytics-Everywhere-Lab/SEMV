from __future__ import annotations

from pathlib import Path

from src.schemas.case_bundle_schema import MediaAsset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def build_media_assets(
    case_id: str,
    media_dir: Path,
    *,
    is_gold_only: bool = False,
    role_override: str | None = None,
) -> list[MediaAsset]:
    if not media_dir.exists():
        return []
    files = sorted(
        path
        for path in media_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    )
    assets: list[MediaAsset] = []
    for idx, path in enumerate(files, start=1):
        role = role_override or infer_media_role(path, idx, len(files), is_gold_only)
        assets.append(
            MediaAsset(
                media_id=f"{case_id}_media_{idx:03d}"
                if not is_gold_only
                else f"{case_id}_gold_media_{idx:03d}",
                case_id=case_id,
                media_type=infer_media_type(path),
                role=role,  # type: ignore[arg-type]
                local_path=str(path),
                platform=infer_platform_from_path(path),
                creator_or_uploader=infer_creator_from_path(path),
                group_id=infer_group_id(path),
                sequence_index=idx,
                is_gold_only=is_gold_only,
            )
        )
    return assets


def infer_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    return "unknown"


def infer_media_role(path: Path, idx: int, total: int, is_gold_only: bool = False) -> str:
    if is_gold_only:
        return "report_attachment"
    text = str(path).lower()
    if any(token in text for token in ["map", "satellite", "geolocation", "reference"]):
        return "reference_media"
    if any(token in text for token in ["story", "instagram story", "status"]):
        return "context_media"
    if total == 1 or idx == 1:
        return "primary_claim_media"
    return "related_claim_media"


def infer_platform_from_path(path: Path) -> str | None:
    text = str(path).lower()
    for platform in ["twitter", "x.com", "facebook", "instagram", "tiktok", "youtube"]:
        if platform in text:
            return "x" if platform in {"twitter", "x.com"} else platform
    return None


def infer_creator_from_path(path: Path) -> str | None:
    parent = path.parent.name.strip()
    if parent and parent.lower() not in {"media", "input", "images", "videos"}:
        return parent
    return None


def infer_group_id(path: Path) -> str | None:
    parent = path.parent.name.strip()
    return parent or None


def infer_case_media_type(media_assets: list[MediaAsset]) -> str:
    types = {asset.media_type for asset in media_assets if asset.media_type != "unknown"}
    if types == {"image"}:
        return "image" if len(media_assets) <= 1 else "multi_image"
    if types == {"video"}:
        return "video" if len(media_assets) <= 1 else "multi_video"
    if not types:
        return "mixed"
    return "mixed"
