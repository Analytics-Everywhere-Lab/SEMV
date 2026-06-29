from __future__ import annotations

import re

from src.schemas.case_bundle_schema import Claim, MediaAsset, SourceCluster


CORE_CLAIM_TYPES = ["what", "where", "when", "who", "why", "authenticity"]


def build_default_claims(
    case_id: str,
    raw_input: dict,
    media_assets: list[MediaAsset],
    source_clusters: list[SourceCluster],
) -> list[Claim]:
    title = raw_input.get("title")
    caption = raw_input.get("caption")
    description = raw_input.get("description")
    location_hint = raw_input.get("location") or raw_input.get("location_hint")
    main_statement = (
        title
        or caption
        or description
        or "The media depicts the event described in the source context."
    )
    all_media_ids = [asset.media_id for asset in media_assets if not asset.is_gold_only]
    claims = [
        Claim(
            claim_id=f"{case_id}_main",
            claim_type="main",
            statement=str(main_statement),
            media_ids=all_media_ids,
            expected_evidence_types=["mixed"],
        )
    ]
    scope_type = "case"
    source_cluster_id = None
    if len(source_clusters) > 1:
        scope_type = "source_cluster"
        source_cluster_id = source_clusters[0].cluster_id
    templates = {
        "what": "The media depicts the event or situation described by the title/caption.",
        "where": f"The media was recorded in or near {location_hint}."
        if location_hint
        else "The media was recorded at the claimed location.",
        "when": "The media corresponds to the claimed date or time."
        if _has_date_text(" ".join(str(v) for v in [title, description, caption] if v))
        else "The media corresponds to the claimed temporal context.",
        "who": (
            "The people, organizations, or sources associated with the media are "
            "correctly identified."
        ),
        "why": (
            "The narrative or motivation implied by the post is supported by "
            "available evidence."
        ),
        "authenticity": (
            "The media is authentic and has not been manipulated or synthetically "
            "generated in a way that changes the claim."
        ),
    }
    for claim_type in CORE_CLAIM_TYPES:
        claims.append(
            Claim(
                claim_id=f"{case_id}_{claim_type}",
                claim_type=claim_type,  # type: ignore[arg-type]
                scope_type=scope_type,  # type: ignore[arg-type]
                statement=templates[claim_type],
                media_ids=all_media_ids,
                source_cluster_id=source_cluster_id,
                expected_evidence_types=_expected_evidence_types(claim_type),
            )
        )
    return _expand_scope_rules(claims, media_assets, location_hint)


def _expand_scope_rules(
    claims: list[Claim],
    media_assets: list[MediaAsset],
    location_hint: str | None,
) -> list[Claim]:
    media_text = " ".join(
        f"{asset.local_path or ''} {asset.description or ''}".lower()
        for asset in media_assets
    )
    if location_hint and any(token in media_text for token in ["smoke", "explosion", "fire"]):
        where = next((claim for claim in claims if claim.claim_type == "where"), None)
        if where:
            claims.append(
                Claim(
                    claim_id=f"{where.claim_id}_camera",
                    claim_type="where",
                    scope_type="case",
                    statement=(
                        "The camera location for the media is consistent with the "
                        f"claimed location context: {location_hint}."
                    ),
                    media_ids=where.media_ids,
                    expected_evidence_types=["geolocation"],
                )
            )
            claims.append(
                Claim(
                    claim_id=f"{where.claim_id}_target",
                    claim_type="where",
                    scope_type="event_cluster",
                    statement=(
                        "The visible target event location is consistent with the "
                        f"claimed location context: {location_hint}."
                    ),
                    media_ids=where.media_ids,
                    expected_evidence_types=["geolocation"],
                )
            )
    return claims


def _expected_evidence_types(claim_type: str) -> list[str]:
    return {
        "what": ["visual", "text", "cross_source"],
        "where": ["geolocation", "visual", "metadata"],
        "when": ["temporal", "metadata", "publication_time"],
        "who": ["entity", "source"],
        "why": ["source_context", "caption"],
        "authenticity": ["forensic", "provenance", "metadata"],
    }.get(claim_type, ["mixed"])


def _has_date_text(text: str) -> bool:
    return bool(
        re.search(r"\b(20\d{2}|19\d{2})\b", text)
        or re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
    )
