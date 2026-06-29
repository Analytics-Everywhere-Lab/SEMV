from __future__ import annotations


def expected_calibration_error(y_true, y_prob, n_bins: int = 10):
    if not y_true:
        return None
    bins = [[] for _ in range(n_bins)]
    for true, prob in zip(y_true, y_prob):
        idx = min(int(max(0.0, min(1.0, prob)) * n_bins), n_bins - 1)
        bins[idx].append((true, prob))
    ece = 0.0
    total = len(y_true)
    bin_rows = []
    for index, bucket in enumerate(bins):
        if not bucket:
            bin_rows.append({"bin": index, "count": 0, "accuracy": None, "confidence": None})
            continue
        acc = sum(int(t == (p >= 0.5)) for t, p in bucket) / len(bucket)
        conf = sum(p for _, p in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(acc - conf)
        bin_rows.append({"bin": index, "count": len(bucket), "accuracy": acc, "confidence": conf})
    return {"ece": ece, "bins": bin_rows}


def brier_score(y_true, y_prob):
    if not y_true:
        return None
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true)
