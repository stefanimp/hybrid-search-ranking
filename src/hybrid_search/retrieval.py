"""Merge candidates from five retrieval caches into per-document signals."""

from __future__ import annotations

import re
from collections.abc import Callable

from .text import norm01, reciprocal_rank


def ensure_cand(cands: dict[str, dict[str, float]], doc_id: str) -> dict[str, float]:
    if doc_id not in cands:
        cands[doc_id] = {}
    return cands[doc_id]


def add_ranked_source(
    cands: dict[str, dict[str, float]],
    source: str,
    items: list,
    valid_docs: set[str],
    score_extractors: list[tuple[str, Callable]],
    max_items: int | None = None,
) -> None:
    if max_items is not None:
        items = items[:max_items]

    raw_scores: dict[str, list[float]] = {name: [] for name, _ in score_extractors}
    kept = []
    for item in items:
        doc_id = str(item.get("document_id")) if isinstance(item, dict) else str(item[0])
        if doc_id not in valid_docs:
            continue
        kept.append((doc_id, item))
        for name, fn in score_extractors:
            try:
                raw_scores[name].append(float(fn(item)))
            except Exception:
                raw_scores[name].append(0.0)

    normalized = {name: norm01(vals) for name, vals in raw_scores.items()}
    seen_rank: dict[str, int] = {}
    for idx, (doc_id, item) in enumerate(kept, 1):
        if doc_id in seen_rank:
            continue
        seen_rank[doc_id] = idx
        feat = ensure_cand(cands, doc_id)
        feat[f"{source}_present"] = 1.0
        feat[f"{source}_rr"] = reciprocal_rank(idx)
        feat[f"{source}_rank_inv"] = 1.0 / idx
        for name, fn in score_extractors:
            raw_name = f"{source}_{name}"
            try:
                raw = float(fn(item))
            except Exception:
                raw = 0.0
            feat[raw_name] = raw
            feat[f"{raw_name}_n"] = (
                float(normalized[name][idx - 1]) if len(normalized[name]) >= idx else 0.0
            )

        if isinstance(item, dict):
            feat["basic_best_rank_inv"] = max(
                feat.get("basic_best_rank_inv", 0.0),
                1.0 / max(1.0, float(item.get("best_rank", 1e9))),
            )
            feat["basic_best_bm25"] = max(
                feat.get("basic_best_bm25", 0.0), float(item.get("best_bm25", 0.0))
            )
            feat["basic_confidence"] = max(
                feat.get("basic_confidence", 0.0), float(item.get("confidence", 0.0))
            )
            for source_name, rank in item.get("sources", {}).items():
                clean = re.sub(r"[^a-zA-Z0-9_]+", "_", str(source_name))
                feat[f"src_{clean}"] = max(feat.get(f"src_{clean}", 0.0), 1.0 / max(1, int(rank)))


def build_candidates_for_query(
    qid: str, caches: dict[str, dict], valid_docs: set[str]
) -> dict[str, dict[str, float]]:
    cands: dict[str, dict[str, float]] = {}

    add_ranked_source(
        cands,
        "basic",
        caches["basic"].get(qid, []),
        valid_docs,
        [
            ("score", lambda x: x.get("basic_score", 0.0)),
            ("bm25", lambda x: x.get("best_bm25", 0.0)),
        ],
        max_items=80,
    )
    add_ranked_source(
        cands,
        "ce100k",
        caches["ce100k"].get(qid, []),
        valid_docs,
        [("dense", lambda x: x[1]), ("ce", lambda x: x[2])],
        max_items=60,
    )
    add_ranked_source(
        cands,
        "cefull",
        caches["cefull"].get(qid, []),
        valid_docs,
        [("dense", lambda x: x[1]), ("ce", lambda x: x[2])],
        max_items=120,
    )
    add_ranked_source(
        cands,
        "dense",
        caches["dense"].get(qid, []),
        valid_docs,
        [("score", lambda x: x[1])],
        max_items=260,
    )
    add_ranked_source(
        cands,
        "tfidf",
        caches["tfidf"].get(qid, []),
        valid_docs,
        [("word", lambda x: x[1]), ("title_char", lambda x: x[2]), ("text_char", lambda x: x[3])],
        max_items=220,
    )
    return cands
