from __future__ import annotations

from urllib.parse import urlparse


def url_f1(pred_urls: set[str], gold_urls: set[str]) -> dict:
    if not pred_urls and not gold_urls:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred_urls & gold_urls)
    precision = tp / len(pred_urls) if pred_urls else 0.0
    recall = tp / len(gold_urls) if gold_urls else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def hallucinated_source_rate(report) -> float:
    evidence_urls = {item.url for item in report.evidence if item.url}
    predicted_urls = set(report.metadata.get("predicted_source_urls", []))
    if not predicted_urls:
        return 0.0
    hallucinated = predicted_urls - evidence_urls
    return len(hallucinated) / len(predicted_urls)


def provenance_coverage(report) -> float:
    arguments = []
    for subclaim in report.subclaim_reports:
        arguments.extend(subclaim.top_support_arguments)
        arguments.extend(subclaim.top_attack_arguments)
    if not arguments:
        return 0.0
    return sum(1 for argument in arguments if argument.evidence_ids) / len(arguments)


def source_diversity(urls: set[str]) -> int:
    domains = {urlparse(url).netloc for url in urls if urlparse(url).netloc}
    return len(domains)
