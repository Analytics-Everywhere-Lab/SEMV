from __future__ import annotations

import re
from pathlib import Path

from src.evaluation.label_normalizer import normalize_mv2026_label
from src.schemas.evaluation_schema import GoldRecord


SECTION_PATTERNS = {
    "case_summary": r"#\s*Case Summary(?P<body>.*?)(?=\n#\s|\Z)",
    "source_details": r"(?:Source Details|#\s*Source Details)(?P<body>.*?)(?=\n#\s|\n-\s*\*\*Where|\Z)",
    "where": r"(?:Where\??\s*\(Location\)|#\s*Where|\*\*Where\??.*?\*\*)(?P<body>.*?)(?=\n#\s|\n-\s*\*\*When|\Z)",
    "when": r"(?:When\??\s*\(Time\)|#\s*When|\*\*When\??.*?\*\*)(?P<body>.*?)(?=\n#\s|\n-\s*\*\*Who|\Z)",
    "who": r"(?:Who\??\s*\(Entities Involved\)|#\s*Who|\*\*Who\??.*?\*\*)(?P<body>.*?)(?=\n#\s|\n-\s*\*\*Why|\Z)",
    "why": r"(?:Why\??\s*\(Motivation or Intent\)|#\s*Why|\*\*Why\??.*?\*\*)(?P<body>.*?)(?=\n#\s|#\s*Forensic|\Z)",
    "forensic": r"#\s*(?:Forensic Analysis|Authenticity.*?)(?P<body>.*?)(?=\n#\s|\Z)",
    "other_evidence": r"#\s*(?:Other Evidence.*?|Supporting Sources|Evidence)(?P<body>.*?)(?=\n#\s|\Z)",
}


URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
COORD_RE = re.compile(r"(?P<lat>-?\d{1,2}\.\d+)\s*[,/]\s*(?P<lon>-?\d{1,3}\.\d+)")


def parse_mv2026_gold_report(path: str | Path, case_id: str | None = None) -> GoldRecord:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    sections = _extract_sections(text)
    status_text = _find_status_text(text, sections)
    subclaim_text = {
        key: value
        for key, value in sections.items()
        if key in {"where", "when", "who", "why", "forensic", "case_summary"}
    }
    subclaim_labels = {
        "where": _section_label(sections.get("where")),
        "when": _section_label(sections.get("when")),
        "who": _section_label(sections.get("who")),
        "why": _section_label(sections.get("why")),
        "authenticity": _section_label(sections.get("forensic")),
        "what": _section_label(sections.get("case_summary")),
    }
    parsed_case_id = case_id or (target.parents[1].name if len(target.parents) > 1 else target.stem)
    return GoldRecord(
        case_id=parsed_case_id,
        dataset_name="mv2026",
        task_type="multimedia_verification",
        gold_final_label=normalize_mv2026_label(status_text or text[:1200]),
        gold_status_text=status_text,
        gold_subclaim_labels=subclaim_labels,
        gold_subclaim_text=subclaim_text,
        gold_coordinates=_extract_coordinates(text),
        gold_source_urls=sorted(set(URL_RE.findall(text))),
        gold_report_path=str(target),
        gold_raw_sections=sections,
    )


def _extract_sections(text: str) -> dict[str, str]:
    sections = {}
    for name, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            sections[name] = match.group("body").strip()
    return sections


def _find_status_text(text: str, sections: dict[str, str]) -> str | None:
    for pattern in [
        r"Final Verification Status\s*[:\-]?\s*(?P<status>.+)",
        r"Status\s*[:\-]\s*(?P<status>.+)",
        r"Final status\s*[:\-]\s*(?P<status>.+)",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group("status").strip()
    return sections.get("case_summary")


def _section_label(text: str | None) -> str:
    label = normalize_mv2026_label(text)
    if label == "verified":
        return "supported"
    if label in {"false_context", "manipulated_or_synthetic"}:
        return "refuted"
    if label == "partially_verified":
        return "partially_supported"
    if label == "uncertain":
        return "uncertain"
    return "insufficient_evidence"


def _extract_coordinates(text: str) -> list[dict]:
    coords = []
    for match in COORD_RE.finditer(text):
        coords.append(
            {
                "latitude": float(match.group("lat")),
                "longitude": float(match.group("lon")),
                "description": "parsed from gold report",
            }
        )
    return coords
