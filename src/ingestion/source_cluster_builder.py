from __future__ import annotations

from urllib.parse import urlparse

from src.utils.hashing import stable_hash_text
from src.schemas.case_bundle_schema import MediaAsset, SourceCluster


def build_source_clusters(
    case_id: str,
    raw_input: dict,
    media_assets: list[MediaAsset],
) -> list[SourceCluster]:
    clusters: dict[str, SourceCluster] = {}
    social_link = (
        raw_input.get("social media link")
        or raw_input.get("social_media_link")
        or raw_input.get("source_url")
        or raw_input.get("url")
    )
    if social_link:
        platform = _platform_from_url(str(social_link))
        cluster_id = f"{case_id}_source_001"
        clusters[cluster_id] = SourceCluster(
            cluster_id=cluster_id,
            case_id=case_id,
            source_name=platform or "provided source",
            platform=platform,
            source_type="social_media_user" if platform else "unknown",
            media_ids=[asset.media_id for asset in media_assets],
            source_urls=[str(social_link)],
            notes="Source inferred from input metadata.",
        )
    for asset in media_assets:
        key = asset.platform or asset.creator_or_uploader or asset.group_id or "input_media"
        cluster_id = f"{case_id}_source_{stable_hash_text(key, 8)}"
        existing = clusters.get(cluster_id)
        if existing:
            existing.media_ids.append(asset.media_id)
            continue
        clusters[cluster_id] = SourceCluster(
            cluster_id=cluster_id,
            case_id=case_id,
            source_name=asset.creator_or_uploader or key,
            platform=asset.platform,
            source_type="benchmark_provider" if key == "input_media" else "unknown",
            media_ids=[asset.media_id],
            source_urls=[asset.source_url] if asset.source_url else [],
        )
    return list(clusters.values())


def _platform_from_url(url: str) -> str | None:
    host = urlparse(url).netloc.lower()
    if "twitter" in host or "x.com" in host:
        return "x"
    for platform in ["facebook", "instagram", "tiktok", "youtube", "telegram"]:
        if platform in host:
            return platform
    return host or None
