from __future__ import annotations

from src.schemas.argument_schema import Argument
from src.schemas.claim_schema import SubClaim
from src.schemas.qbaf_schema import QBAFGraph
from src.utils.llm_client import LLMClient


class ClashResolver:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def should_resolve(self, graph: QBAFGraph, arguments: list[Argument]) -> bool:
        del graph
        support = max((arg.score for arg in arguments if arg.stance == "support"), default=0.0)
        attack = max((arg.score for arg in arguments if arg.stance == "attack"), default=0.0)
        return support >= 0.55 and attack >= 0.55 and abs(support - attack) <= 0.25

    def resolve(
        self,
        claim: SubClaim,
        graph: QBAFGraph,
        arguments: list[Argument],
    ) -> list[Argument]:
        del claim, graph
        return [
            arg.model_copy(
                update={
                    "uncertainty_flags": sorted(
                        set(arg.uncertainty_flags + ["major_factual_clash"])
                    ),
                    "score": arg.score * 0.9,
                    "verification_notes": (
                        (arg.verification_notes or "")
                        + " Clash resolver lowered confidence due to comparable opposing evidence."
                    ).strip(),
                }
            )
            for arg in arguments
        ]
