from __future__ import annotations

from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text


class GeolocationCandidateExtractor:
    def extract(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        candidates: list[EvidenceItem] = []
        for item in evidence:
            metadata = item.metadata or {}
            raw = item.raw_output or {}
            text = f"{item.content} {metadata} {raw}"
            if "gps" not in text.lower() and item.source_type not in {"ocr", "asr", "visual_vqa", "frame_analysis", "web_article", "news_article"}:
                continue
            if not any(token in text.lower() for token in ("gps", "street", "road", "city", "province", "oblast", "avenue", "bridge", "station")):
                continue
            evidence_id = f"geo_candidate_{stable_hash_text(item.evidence_id + text[:80])}"
            candidates.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_type="geolocation_candidate",
                    source=item.source,
                    title="Candidate location",
                    content=f"Candidate location clue derived from {item.evidence_id}: {item.content[:300]}",
                    reliability=min(0.75, max(0.45, item.reliability)),
                    relevance=0.90,
                    media_path=item.media_path,
                    metadata={"source_clues": [item.evidence_id], "lat": None, "lon": None},
                    supports_claim_types=["where"],
                    provenance=Provenance(
                        source_id=evidence_id,
                        source_type="geolocation_candidate",
                        source=item.source,
                        retrieval_method="media_clue_extraction",
                        metadata={"source_evidence_id": item.evidence_id},
                    ),
                )
            )
        return candidates
