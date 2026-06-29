from __future__ import annotations

from src.ingestion.media_manifest_builder import build_media_assets


def test_media_manifest_roles_and_types(tmp_path):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "a.jpg").write_bytes(b"x")
    (media_dir / "map_reference.png").write_bytes(b"x")

    assets = build_media_assets("case1", media_dir)

    assert [asset.media_type for asset in assets] == ["image", "image"]
    assert assets[0].role == "primary_claim_media"
    assert any(asset.role == "reference_media" for asset in assets)
