from __future__ import annotations

from src.ingestion.cosmos_adapter import build_cosmos_pair_case, build_cosmos_triplet_case


def test_cosmos_adapter_creates_caption_context_claim():
    row = {"case_id": "cosmos_001", "image_path": "image.jpg", "caption": "Claimed event caption.", "label": "out_of_context"}
    bundle = build_cosmos_pair_case(row, split="train")

    claim_types = {claim.claim_type for claim in bundle.claims}
    assert "caption_context" in claim_types
    assert bundle.media_assets[0].media_type == "image"
    assert bundle.run_config.allow_web_search is False
    assert bundle.gold.gold_final_label == "out_of_context_cheapfake"


def test_cosmos_triplet_mode_adds_contradiction_claim():
    bundle = build_cosmos_triplet_case(
        {"case_id": "cosmos_002", "image_path": "image.jpg", "caption": "A", "caption_2": "B", "label": 0},
        split="test",
    )

    assert bundle.task.subtask == "image_caption_triplet"
    assert any("captions" in claim.statement for claim in bundle.claims)
