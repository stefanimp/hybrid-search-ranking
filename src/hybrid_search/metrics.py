"""Mean average precision at ten for query rankings."""

from __future__ import annotations

import numpy as np


def map_at_10_standard(preds: dict[str, list[str]], qrels: dict[str, set[str]]) -> float:
    """Average query AP@10, counting a relevant document only on its first occurrence."""
    if not qrels:
        return 0.0
    vals = []
    for qid, rel in qrels.items():
        hit = 0
        ap = 0.0
        seen = set()
        for rank, doc_id in enumerate(preds.get(qid, [])[:10], 1):
            if doc_id in rel and doc_id not in seen:
                hit += 1
                ap += hit / rank
            seen.add(doc_id)
        vals.append(ap / min(10, len(rel)) if rel else 0.0)
    return float(np.mean(vals))
