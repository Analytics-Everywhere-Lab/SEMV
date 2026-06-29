from __future__ import annotations

from src.schemas.argument_schema import Argument
from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceItem
from src.utils.llm_client import LLMClient


class ArgumentVerifier:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def verify(
        self,
        claim: SubClaim,
        argument: Argument,
        evidence: list[EvidenceItem],
    ) -> Argument:
        linked = [item for item in evidence if item.evidence_id in argument.evidence_ids]
        if not linked:
            return argument.model_copy(
                update={
                    "verifier_valid": False,
                    "verification_notes": "Argument has no linked evidence.",
                    "uncertainty_flags": sorted(
                        set(argument.uncertainty_flags + ["argument_without_evidence"])
                    ),
                }
            )

        prompt = (
            "Check whether the argument is grounded in the provided evidence. "
            "Return JSON with valid boolean and notes string.\n"
            f"Sub-claim: {claim.statement}\nArgument: {argument.text}\n"
            f"Evidence: {[item.content[:240] for item in linked]}"
        )
        try:
            data = self.llm_client.generate_json(prompt)
            valid = bool(data.get("valid", True))
            notes = data.get("notes") or "Verifier returned no notes."
            return argument.model_copy(
                update={"verifier_valid": valid, "verification_notes": notes}
            )
        except Exception:
            evidence_text = " ".join(item.content.lower() for item in linked)
            token_overlap = sum(
                1 for token in argument.text.lower().split()[:30] if token in evidence_text
            )
            valid = token_overlap > 0 or any(item.uncertainty_flags for item in linked)
            return argument.model_copy(
                update={
                    "verifier_valid": valid,
                    "verification_notes": "Deterministic verifier fallback used.",
                }
            )

    def verify_all(
        self,
        claim: SubClaim,
        arguments: list[Argument],
        evidence: list[EvidenceItem],
        bundle: object | None = None,
    ) -> list[Argument]:
        del bundle
        return [self.verify(claim, argument, evidence) for argument in arguments]
