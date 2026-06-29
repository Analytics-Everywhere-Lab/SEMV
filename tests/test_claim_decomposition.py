from __future__ import annotations

from src.planning.claim_decomposer import ClaimDecomposer
from src.schemas.case_schema import MultimediaCase

from tests.conftest import FakeLLMClient


def test_claim_decomposition_returns_all_six_dimensions():
    case = MultimediaCase(case_id="case1", claim="A video shows an event yesterday.")
    subclaims = ClaimDecomposer(FakeLLMClient()).decompose(case, [])

    assert {claim.claim_type for claim in subclaims} == {"what", "where", "when", "who", "why", "authenticity"}
    assert all(claim.claim_id.startswith("case1_") for claim in subclaims)
