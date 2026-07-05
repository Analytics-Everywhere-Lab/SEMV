from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.retrieval.deep_researcher import DeepResearcher
from src.retrieval.gdelt_search import GDELTSearch, _is_good_gdelt_query
from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem


def _claim(kind="what"):
    return SubClaim(claim_id="c1", claim_type=kind, statement="What happened in this video?")


def _config(**overrides):
    retrieval = {
        "gdelt_search_enabled": True,
        "gdelt_cache_enabled": False,
        "gdelt_fetch_full_articles": False,
        "gdelt_min_interval_sec": 0,
        "gdelt_max_retries": 0,
        "gdelt_backoff_base_sec": 0,
    }
    retrieval.update(overrides)
    return {"retrieval": retrieval}


def test_gdelt_disabled_returns_empty():
    search = GDELTSearch(config={"retrieval": {"gdelt_search_enabled": False}})
    result = search.search(_claim(), ResearchPlan(claim_id="c1"), queries=["test"])
    assert result == []


def test_gdelt_converts_article_to_evidence_item():
    search = GDELTSearch(config=_config())
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "articles": [
            {
                "url": "https://www.reuters.com/world/example",
                "title": "Example event confirmed",
                "seendate": "20260705120000",
                "domain": "reuters.com",
                "language": "English",
                "sourcecountry": "United States",
                "socialimage": "https://example.com/image.jpg",
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        result = search.search(_claim(), ResearchPlan(claim_id="c1"), queries=["example event"])

    assert len(result) == 1
    item = result[0]
    assert item.source_type == "news_article"
    assert item.provenance.retrieval_method == "gdelt_doc_api_artlist"
    assert item.metadata["adapter"] == "gdelt"
    assert item.metadata["domain"] == "reuters.com"
    assert item.reliability >= 0.75


def test_gdelt_failure_returns_uncertainty_item():
    search = GDELTSearch(config=_config())

    with patch("requests.get", side_effect=TimeoutError("timed out")):
        result = search.search(_claim(), ResearchPlan(claim_id="c1"), queries=["example event"])

    assert len(result) == 1
    assert result[0].source_type == "synthetic_uncertainty"
    assert "gdelt_search_failed" in result[0].uncertainty_flags[0]


def test_generic_single_word_queries_are_rejected():
    assert _is_good_gdelt_query("location") is False
    assert _is_good_gdelt_query("where") is False
    assert _is_good_gdelt_query("") is False
    assert _is_good_gdelt_query("the a of") is False


def test_specific_queries_are_accepted():
    assert _is_good_gdelt_query("protest in Dhaka July 2024") is True
    assert _is_good_gdelt_query("building collapse Taipei") is True
    assert _is_good_gdelt_query("flooded street Valencia") is True


def test_gdelt_skips_request_when_no_good_query_available():
    search = GDELTSearch(config=_config())
    with patch("requests.get") as mock_get:
        result = search.search(_claim(), ResearchPlan(claim_id="c1"), queries=["location"])
    mock_get.assert_not_called()
    assert result == []


def test_gdelt_retries_on_429_then_succeeds():
    search = GDELTSearch(config=_config(gdelt_max_retries=1))

    rate_limited = MagicMock()
    rate_limited.status_code = 429
    rate_limited.headers = {}

    ok_response = MagicMock()
    ok_response.status_code = 200
    ok_response.raise_for_status = MagicMock()
    ok_response.json.return_value = {
        "articles": [
            {
                "url": "https://www.reuters.com/world/example",
                "title": "Protest in Dhaka turns violent",
                "domain": "reuters.com",
            }
        ]
    }

    with patch("requests.get", side_effect=[rate_limited, ok_response]) as mock_get:
        result = search.search(_claim(), ResearchPlan(claim_id="c1"), queries=["protest in Dhaka July 2024"])

    assert mock_get.call_count == 2
    assert len(result) == 1
    assert result[0].source_type == "news_article"


def test_gdelt_opens_circuit_breaker_after_exhausting_retries_on_429():
    search = GDELTSearch(config=_config(gdelt_max_retries=1))

    rate_limited = MagicMock()
    rate_limited.status_code = 429
    rate_limited.headers = {}

    with patch("requests.get", return_value=rate_limited), patch("random.uniform", return_value=0.0):
        first = search.search(_claim(), ResearchPlan(claim_id="c1"), queries=["protest in Dhaka July 2024"])

    assert first[0].source_type == "synthetic_uncertainty"
    assert search._rate_limited_until > 0

    with patch("requests.get") as mock_get:
        second = search.search(_claim(), ResearchPlan(claim_id="c1"), queries=["protest in Dhaka July 2024"])

    mock_get.assert_not_called()
    assert second == []


def test_deep_researcher_calls_gdelt_before_duckduckgo():
    llm_client = MagicMock()
    researcher = DeepResearcher(llm_client)

    gdelt_item = EvidenceItem(
        evidence_id="gdelt_1",
        source_type="news_article",
        content="gdelt result",
        source="gdelt",
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

    def fake_gdelt_search(*args, **kwargs):
        call_order.append("gdelt")
        return [gdelt_item]

    def fake_ddg_search(*args, **kwargs):
        call_order.append("ddg")
        return [ddg_item]

    researcher.gdelt_search.search = fake_gdelt_search
    researcher.free_web_search.search = fake_ddg_search
    researcher.adapters = []

    result = researcher.research(_claim(), ResearchPlan(claim_id="c1"), existing_evidence=[])

    assert call_order == ["gdelt", "ddg"]
    result_ids = {item.evidence_id for item in result}
    assert "gdelt_1" in result_ids
    assert "ddg_1" in result_ids
