from __future__ import annotations

from src.retrieval.geolocation_candidate_extractor import GeolocationCandidateExtractor
from src.schemas.evidence_schema import EvidenceItem


def test_exif_gps_creates_candidate():
    item = EvidenceItem(evidence_id="m", source_type="metadata_exiftool", source="img", content="metadata", metadata={"GPSLatitude": 44.65, "GPSLongitude": -63.57})
    candidates = GeolocationCandidateExtractor({"retrieval": {"geocoding_enabled": False}}).extract([item])
    assert candidates[0].raw_output["lat"] == 44.65
    assert candidates[0].reliability == 0.85


def test_ocr_and_vlm_same_place_increases_reliability():
    evidence = [
        EvidenceItem(evidence_id="o", source_type="ocr", source="img", content="Visible text: Halifax"),
        EvidenceItem(evidence_id="v", source_type="frame_analysis", source="img", content="clue", raw_output={"location_clues": ["Halifax"]}),
    ]
    candidates = GeolocationCandidateExtractor({"retrieval": {"geocoding_enabled": False}}).extract(evidence)
    assert any(c.raw_output["candidate_name"] == "Halifax" and c.reliability >= 0.75 for c in candidates)


def test_mock_geocoder_returns_lat_lon(monkeypatch, tmp_path):
    extractor = GeolocationCandidateExtractor({"retrieval": {"geocoding_enabled": True, "geocoding_cache_path": str(tmp_path / "geo.json")}})
    monkeypatch.setattr(extractor, "_geocode", lambda name: (1.0, 2.0))
    item = EvidenceItem(evidence_id="o", source_type="ocr", source="img", content="Visible text: Halifax")
    candidates = extractor.extract([item])
    assert any(c.raw_output["lat"] == 1.0 for c in candidates)


def test_geolocation_does_not_extract_chroma_location_center():
    item = EvidenceItem(
        evidence_id="ffprobe1",
        source_type="metadata_ffprobe",
        source="video.mp4",
        title="FFprobe metadata",
        content="Video metadata",
        reliability=0.8,
        relevance=0.6,
        raw_output={"streams": [{"chroma_location": "center"}]},
        metadata={"ffprobe": {"streams": [{"chroma_location": "center"}]}},
    )

    candidates = GeolocationCandidateExtractor({"retrieval": {"geocoding_enabled": False}}).extract([item])

    assert candidates == []
