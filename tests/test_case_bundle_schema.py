from __future__ import annotations

from src.schemas.case_bundle_schema import CaseBundle, multimedia_case_to_case_bundle, case_bundle_to_multimedia_case
from src.schemas.case_schema import MultimediaCase, MediaItem


def test_case_bundle_round_trip_from_legacy_case():
    legacy = MultimediaCase(
        case_id="case1",
        claim="A photo shows the claimed event.",
        media=[MediaItem(path="image.jpg", media_type="image")],
        context="caption context",
    )
    bundle = multimedia_case_to_case_bundle(legacy)
    restored = case_bundle_to_multimedia_case(bundle)

    assert isinstance(bundle, CaseBundle)
    assert bundle.case_id == "case1"
    assert bundle.task.media_type == "image"
    assert restored.claim == legacy.claim
    assert restored.media[0].path == "image.jpg"
