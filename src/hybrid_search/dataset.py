"""Build grouped query-document matrices from cached retrieval signals."""

from __future__ import annotations

import pandas as pd

from .features import content_features
from .retrieval import build_candidates_for_query, ensure_cand


def build_dataset(
    query_ids: list[str],
    query_text: dict[str, str],
    docs: dict[str, tuple[str, str]],
    valid_docs: set[str],
    caches: dict[str, dict],
    qrels: dict[str, set[str]] | None = None,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    rows: list[dict[str, float | str]] = []
    groups: list[int] = []
    doc_ids: list[str] = []

    for qid in query_ids:
        cands = build_candidates_for_query(qid, caches, valid_docs)
        if qrels is not None:
            for pos in qrels.get(qid, set()):
                if pos in valid_docs:
                    ensure_cand(cands, pos)
        groups.append(len(cands))
        query = query_text[qid]
        for doc_id, feats in cands.items():
            title, text = docs.get(doc_id, ("", ""))
            row = dict(feats)
            row.update(content_features(query, doc_id, title, text))
            row["qid"] = qid
            row["doc_id"] = doc_id
            if qrels is not None:
                row["label"] = 1 if doc_id in qrels.get(qid, set()) else 0
            rows.append(row)
            doc_ids.append(doc_id)

    df = pd.DataFrame(rows).fillna(0.0)

    # Preserve the cross-encoder/dense interaction used by the trained ranker.
    if "cefull_dense_n" in df.columns and "cefull_ce_n" in df.columns:
        df["cefull_hybrid"] = 0.3 * df["cefull_dense_n"] + 0.7 * df["cefull_ce_n"]

    return df, doc_ids, groups


def features_only(df: pd.DataFrame) -> list[str]:
    blocked = {"qid", "doc_id", "label"}
    return [c for c in df.columns if c not in blocked]
