from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from src.retrieval.deep_researcher import DeepResearcher
from src.retrieval.yandex_reverse_image_search import YandexReverseImageSearch
from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem


def _claim(kind="what"):
    return SubClaim(claim_id="c1", claim_type=kind, statement="What happened in this image?")


def _config(**overrides):
    retrieval = {
        "yandex_reverse_enabled": True,
        "yandex_reverse_cache_enabled": False,
        "yandex_reverse_min_interval_sec": 0,
        "yandex_reverse_max_retries": 0,
        "yandex_reverse_timeout_sec": 1,
    }
    retrieval.update(overrides)
    return {"retrieval": retrieval}


def _image(path: Path) -> Path:
    Image.new("RGB", (16, 16), color=(200, 20, 20)).save(path)
    return path


def test_yandex_reverse_disabled_returns_empty(tmp_path):
    search = YandexReverseImageSearch(config={"retrieval": {"yandex_reverse_enabled": False}})
    result = search.search(_claim(), ResearchPlan(claim_id="c1"), [_image(tmp_path / "query.jpg")])
    assert result == []


def test_yandex_reverse_missing_credentials_returns_uncertainty(monkeypatch, tmp_path):
    monkeypatch.delenv("SEMV_YANDEX_API_KEY", raising=False)
    monkeypatch.delenv("SEMV_YANDEX_IAM_TOKEN", raising=False)
    monkeypatch.delenv("SEMV_YANDEX_FOLDER_ID", raising=False)
    search = YandexReverseImageSearch(config=_config())

    result = search.search(_claim(), ResearchPlan(claim_id="c1"), [_image(tmp_path / "query.jpg")])

    assert len(result) == 1
    assert result[0].source_type == "synthetic_uncertainty"
    assert "yandex_reverse_missing_credentials" in result[0].uncertainty_flags


def test_yandex_reverse_missing_folder_returns_uncertainty(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMV_YANDEX_API_KEY", "test-key")
    monkeypatch.delenv("SEMV_YANDEX_IAM_TOKEN", raising=False)
    monkeypatch.delenv("SEMV_YANDEX_FOLDER_ID", raising=False)
    search = YandexReverseImageSearch(config=_config())

    result = search.search(_claim(), ResearchPlan(claim_id="c1"), [_image(tmp_path / "query.jpg")])

    assert len(result) == 1
    assert result[0].source_type == "synthetic_uncertainty"
    assert "yandex_reverse_missing_folder_id" in result[0].uncertainty_flags


def test_yandex_reverse_converts_response_to_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMV_YANDEX_API_KEY", "test-key")
    monkeypatch.setenv("SEMV_YANDEX_FOLDER_ID", "folder-1")
    search = YandexReverseImageSearch(config=_config())
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "images": [
            {
                "url": "https://example.com/image.jpg",
                "format": "IMAGE_FORMAT_JPEG",
                "width": "800",
                "height": "600",
                "passage": "Original event photo",
                "host": "example.com",
                "pageTitle": "Original source page",
                "pageUrl": "https://example.com/article",
            }
        ],
        "page": "0",
        "maxPage": "1",
        "id": "mock-cbir-id",
    }

    monkeypatch.setattr("requests.post", MagicMock(return_value=response))
    result = search.search(_claim(), ResearchPlan(claim_id="c1"), [_image(tmp_path / "query.jpg")])

    assert len(result) == 1
    item = result[0]
    assert item.source_type == "reverse_image_web_candidate"
    assert item.metadata["adapter"] == "yandex_reverse_image_search"
    assert item.url == "https://example.com/article"
    assert item.provenance.retrieval_method == "yandex_search_api_search_by_image"


def test_deep_researcher_calls_yandex_before_free_web(tmp_path):
    llm_client = MagicMock()
    researcher = DeepResearcher(llm_client)
    query_path = _image(tmp_path / "query.jpg")
    existing = [
        EvidenceItem(
            evidence_id="frame_1",
            source_type="frame_analysis",
            content="frame",
            source="case",
            media_path=str(query_path),
            supports_claim_types=["what"],
        )
    ]
    yandex_item = EvidenceItem(
        evidence_id="yandex_1",
        source_type="reverse_image_web_candidate",
        content="yandex result",
        source="yandex",
        supports_claim_types=["what"],
    )
    ddg_item = EvidenceItem(
        evidence_id="ddg_1",
        source_type="web_article",
        content="ddg result",
        source="ddg",
        supports_claim_types=["what"],
    )
    call_order: list[str] = []

    def fake_yandex_search(*args, **kwargs):
        call_order.append("yandex")
        return [yandex_item]

    def fake_ddg_search(*args, **kwargs):
        call_order.append("ddg")
        return [ddg_item]

    researcher.yandex_reverse_search.search = fake_yandex_search
    researcher.free_web_search.search = fake_ddg_search
    researcher.gdelt_search.search = lambda *args, **kwargs: []
    researcher.adapters = []

    result = researcher.research(_claim(), ResearchPlan(claim_id="c1"), existing_evidence=existing)

    assert call_order == ["yandex", "ddg"]
    result_ids = {item.evidence_id for item in result}
    assert "yandex_1" in result_ids
    assert "ddg_1" in result_ids


def test_deep_researcher_skips_yandex_when_reverse_search_disallowed(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMV_ENABLE_YANDEX_REVERSE", "true")
    llm_client = MagicMock()
    researcher = DeepResearcher(llm_client)
    query_path = _image(tmp_path / "query.jpg")
    existing = [
        EvidenceItem(
            evidence_id="frame_1",
            source_type="frame_analysis",
            content="frame",
            source="case",
            media_path=str(query_path),
            supports_claim_types=["what"],
        )
    ]
    researcher.yandex_reverse_search.search = MagicMock(side_effect=AssertionError("Yandex must not run"))
    researcher.free_web_search.search = MagicMock(return_value=[])
    researcher.gdelt_search.search = MagicMock(return_value=[])
    researcher.adapters = []

    researcher.research(
        _claim(),
        ResearchPlan(claim_id="c1"),
        existing_evidence=existing,
        allow_reverse_search=False,
    )

    researcher.yandex_reverse_search.search.assert_not_called()
