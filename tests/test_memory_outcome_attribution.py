from __future__ import annotations

from src.main import _log_supervised_memory_outcomes
from src.schemas.argument_schema import Argument
from src.schemas.case_bundle_schema import CaseBundle, DatasetInfo, InputMetadata, TaskInfo
from tests.memory_test_utils import make_record, make_service


def _bundle(split="train"):
    return CaseBundle(
        case_id="train_case", dataset=DatasetInfo(dataset_name="mv2026", dataset_split=split),
        task=TaskInfo(task_type="multimedia_verification", media_type="image"), input=InputMetadata(),
    )


def _argument():
    return Argument(argument_id="a1", claim_id="c1", stance="support", text="grounded",
                    evidence_ids=["e1"], verifier_valid=True,
                    metadata={"used_memory_ids": ["mem_used"]})


def test_grounded_argument_with_incorrect_prediction_is_not_success(tmp_path):
    service = make_service(tmp_path)
    service.store.append(make_record(memory_id="mem_used"))
    service.log_usage(case_id="train_case", memory_id="mem_used", stage="argument_cited",
                      argument_id="a1", outcome="grounded", dataset_split="train")
    _log_supervised_memory_outcomes(service, _bundle(), [_argument()], "verified", "false_context")
    service.consolidate()
    record = service.store.load_long_term()[0]
    assert record.usage_count == 1
    assert record.successful_usage_count == 0
    assert record.unsuccessful_usage_count == 1


def test_correct_supervised_outcome_increments_success_once(tmp_path):
    service = make_service(tmp_path)
    service.store.append(make_record(memory_id="mem_used"))
    for _ in range(2):
        _log_supervised_memory_outcomes(service, _bundle(), [_argument()], "verified", "verified")
    service.consolidate()
    service.consolidate()
    record = service.store.load_long_term()[0]
    assert record.usage_count == 1
    assert record.successful_usage_count == 1
    assert record.unsuccessful_usage_count == 0
