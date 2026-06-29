from __future__ import annotations

from src.ingestion.default_claim_builder import build_default_claims
from src.schemas.case_bundle_schema import MediaAsset, SourceCluster


def test_camera_target_where_claims_for_distant_explosion():
    media = [MediaAsset(media_id="m1", case_id="c1", media_type="video", local_path="smoke_explosion.mp4", role="primary_claim_media")]
    clusters = [SourceCluster(cluster_id="s1", case_id="c1")]
    claims = build_default_claims("c1", {"title": "Explosion", "location": "City"}, media, clusters)

    where_statements = [claim.statement.lower() for claim in claims if claim.claim_type == "where"]
    assert any("camera location" in statement for statement in where_statements)
    assert any("target event location" in statement for statement in where_statements)
