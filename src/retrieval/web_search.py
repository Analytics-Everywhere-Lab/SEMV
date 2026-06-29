from __future__ import annotations

from pathlib import Path

from src.schemas.claim_schema import ResearchPlan, SubClaim
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.io import project_root, read_jsonl


class CachedEvidenceSearch:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or project_root() / "data" / "evidence_cache"

    def search(self, claim: SubClaim, plan: ResearchPlan) -> list[EvidenceItem]:
        rows = self._load_rows()
        terms = " ".join([claim.statement, claim.claim_type, *plan.search_queries]).lower()
        results = []
        for row in rows:
            haystack = " ".join(
                str(row.get(key, "")) for key in ("title", "content", "claim_type", "tags")
            ).lower()
            if claim.claim_type in str(row.get("claim_type", "")) or any(
                token and token in haystack for token in terms.split()
            ):
                evidence_id = row.get("evidence_id") or f"cached_{stable_hash_text(str(row))}"
                results.append(
                    EvidenceItem(
                        evidence_id=evidence_id,
                        source_type=row.get("source_type", "cached_search"),
                        source=row.get("source", "evidence_cache"),
                        title=row.get("title"),
                        content=row.get("content", ""),
                        url=row.get("url"),
                        reliability=float(row.get("reliability", 0.6)),
                        relevance=float(row.get("relevance", 0.6)),
                        metadata=row.get("metadata", {}),
                        supports_claim_types=row.get("supports_claim_types", [claim.claim_type]),
                        provenance=Provenance(
                            source_id=evidence_id,
                            source_type=row.get("source_type", "cached_search"),
                            source=row.get("source", "evidence_cache"),
                            url=row.get("url"),
                            retrieval_method="local_cache",
                            metadata={"cache_file": str(row.get("_cache_file", ""))},
                        ),
                    )
                )
        return results

    def _load_rows(self) -> list[dict]:
        rows = []
        for path in sorted(self.cache_dir.glob("*.jsonl")):
            for row in read_jsonl(path):
                row["_cache_file"] = str(path)
                rows.append(row)
        return rows
