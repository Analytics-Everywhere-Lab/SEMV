from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.schemas.argument_schema import Argument
from src.schemas.claim_schema import SubClaim
from src.schemas.evidence_schema import EvidenceItem
from src.utils.env_loader import get_bool_env, get_int_env
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
        if get_bool_env("SEMV_BATCH_ARGUMENT_VERIFICATION", True):
            return self.verify_all_batched(claim, arguments, evidence)

        if not get_bool_env("SEMV_PARALLEL_ARGUMENT_VERIFICATION", False):
            return [self.verify(claim, argument, evidence) for argument in arguments]

        max_workers = self._max_parallel_verify_workers(len(arguments))
        if max_workers <= 1:
            return [self.verify(claim, argument, evidence) for argument in arguments]

        results_by_id: dict[str, Argument] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.verify, claim, argument, evidence): argument
                for argument in arguments
            }
            for future in as_completed(futures):
                argument = futures[future]
                results_by_id[argument.argument_id] = future.result()

        return [results_by_id[argument.argument_id] for argument in arguments]

    def verify_all_batched(
        self,
        claim: SubClaim,
        arguments: list[Argument],
        evidence: list[EvidenceItem],
        bundle: object | None = None,
    ) -> list[Argument]:
        del bundle
        if not arguments:
            return []

        evidence_by_id = {item.evidence_id: item for item in evidence}
        preverified: dict[str, Argument] = {}
        argument_payload = []
        for argument in arguments:
            linked = [
                evidence_by_id[evidence_id].content[:240]
                for evidence_id in argument.evidence_ids
                if evidence_id in evidence_by_id
            ]
            if not linked:
                preverified[argument.argument_id] = self.verify(claim, argument, evidence)
                continue
            argument_payload.append(
                {
                    "argument_id": argument.argument_id,
                    "stance": argument.stance,
                    "text": argument.text,
                    "linked_evidence": linked,
                }
            )

        if not argument_payload:
            return [preverified[argument.argument_id] for argument in arguments]

        prompt = (
            "Verify whether each argument is grounded in its linked evidence. "
            "Return JSON exactly as "
            "{\"results\":[{\"argument_id\":\"...\",\"valid\":true,\"notes\":\"...\"}]}\n"
            f"Sub-claim: {claim.statement}\n"
            f"Arguments: {argument_payload}"
        )

        try:
            data = self.llm_client.generate_json(prompt)
            result_by_id = {
                item.get("argument_id"): item
                for item in data.get("results", [])
            }
            verified_by_id = dict(preverified)
            for argument in arguments:
                if argument.argument_id in verified_by_id:
                    continue
                result = result_by_id.get(argument.argument_id, {})
                verified_by_id[argument.argument_id] = argument.model_copy(
                    update={
                        "verifier_valid": bool(result.get("valid", True)),
                        "verification_notes": (
                            result.get("notes") or "Batched verifier returned no notes."
                        ),
                    }
                )
            return [verified_by_id[argument.argument_id] for argument in arguments]
        except Exception:
            return [self.verify(claim, argument, evidence) for argument in arguments]

    @staticmethod
    def _max_parallel_verify_workers(argument_count: int) -> int:
        if argument_count <= 1:
            return argument_count
        max_workers = get_int_env("SEMV_ARGUMENT_VERIFY_WORKERS", 2)
        return max(1, min(argument_count, max_workers))
