from __future__ import annotations

from src.memory.memory_query_builder import build_memory_query
from src.schemas.case_bundle_schema import CaseBundle, Claim, DatasetInfo, InputMetadata, MediaAsset, SourceCluster, TaskInfo


def test_memory_query_contains_bundle_context():
    bundle = CaseBundle(
        case_id="c1",
        dataset=DatasetInfo(dataset_name="mv2026"),
        task=TaskInfo(task_type="multimedia_verification", media_type="image"),
        input=InputMetadata(title="Title", location_hint="Place"),
        media_assets=[MediaAsset(media_id="m1", case_id="c1", media_type="image", role="primary_claim_media")],
        source_clusters=[SourceCluster(cluster_id="s1", case_id="c1", source_name="source", platform="x")],
    )
    claim = Claim(claim_id="c1_where", claim_type="where", statement="Where claim")

    query = build_memory_query(bundle, claim)

    assert query["case_id"] == "c1"
    assert query["claim_type"] == "where"
    assert query["media_roles"] == ["primary_claim_media"]
    assert query["platforms"] == ["x"]
