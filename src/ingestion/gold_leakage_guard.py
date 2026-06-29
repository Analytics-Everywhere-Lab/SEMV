from __future__ import annotations

from src.schemas.case_bundle_schema import CaseBundle


def assert_no_gold_leakage(bundle: CaseBundle, mode: str) -> None:
    if mode in {"inference_only", "test", "static"}:
        if bundle.gold.read_gold_before_prediction:
            raise ValueError("Gold was marked as read before prediction.")
        for evidence in bundle.provided_evidence:
            if evidence.is_gold_only:
                raise ValueError(
                    "Gold evidence found in provided_evidence during inference."
                )
        for media in bundle.media_assets:
            if media.is_gold_only and media.role != "report_attachment":
                raise ValueError("Gold-only media exposed as input media.")


def mark_gold_read_after_prediction(bundle: CaseBundle) -> CaseBundle:
    return bundle.model_copy(
        update={"gold": bundle.gold.model_copy(update={"read_gold_before_prediction": True})}
    )
