from __future__ import annotations

import pytest

from src.ingestion.gold_leakage_guard import assert_no_gold_leakage
from src.schemas.case_bundle_schema import CaseBundle, DatasetInfo, GoldAnnotation, InputMetadata, ProvidedEvidence, RunConfig, TaskInfo


def _bundle():
    return CaseBundle(
        case_id="case1",
        dataset=DatasetInfo(dataset_name="mv2026"),
        task=TaskInfo(task_type="multimedia_verification", media_type="image"),
        input=InputMetadata(title="Claim"),
        gold=GoldAnnotation(read_gold_before_prediction=False),
        run_config=RunConfig(),
    )


def test_gold_report_not_read_during_inference():
    bundle = _bundle()
    assert_no_gold_leakage(bundle, mode="inference_only")


def test_gold_evidence_rejected_during_inference():
    bundle = _bundle()
    bundle.provided_evidence.append(
        ProvidedEvidence(evidence_id="g1", source_type="gold", modality="text", content="gold", is_gold_only=True)
    )

    with pytest.raises(ValueError):
        assert_no_gold_leakage(bundle, mode="inference_only")
