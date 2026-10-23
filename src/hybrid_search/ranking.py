"""LambdaRank fitting, score-gap shortlists and reciprocal rank fusion."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .metrics import map_at_10_standard

SHORTLIST_GAPS = (0.18, 0.10, 0.08)
logger = logging.getLogger(__name__)


def train_ranker(X: pd.DataFrame, y: np.ndarray, groups: list[int]):
    from lightgbm import LGBMRanker

    model = LGBMRanker(
        objective="lambdarank",
        metric="map",
        n_estimators=520,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=18,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.20,
        random_state=42,
        n_jobs=-1,
        importance_type="gain",
        label_gain=[0, 1],
        verbosity=-1,
    )
    model.fit(X, y, group=groups)
    return model


def select_shortlist(
    df: pd.DataFrame, scores: np.ndarray, gap1: float = 0.18, gap2: float = 0.10, gap3: float = 0.08
) -> dict[str, list[str]]:
    tmp = df[["qid", "doc_id"]].copy()
    tmp["score"] = scores
    out: dict[str, list[str]] = {}
    for qid, group in tmp.groupby("qid", sort=False):
        g = group.sort_values("score", ascending=False)
        vals = g.score.to_numpy()
        docs = g.doc_id.astype(str).tolist()
        if len(docs) == 1:
            out[str(qid)] = docs
            continue
        pred = [docs[0]]
        if len(docs) > 1 and vals[1] >= vals[0] - gap1:
            pred.append(docs[1])
        if len(docs) > 2 and vals[2] >= vals[0] - gap2 and vals[2] >= vals[1] - gap3:
            pred.append(docs[2])
        out[str(qid)] = pred
    return out


def rank_ce(arr: list, ce_w: float, dense_w: float) -> list[str]:
    if not arr:
        return []
    ds = np.asarray([x[1] for x in arr], dtype=np.float32)
    ce = np.asarray([x[2] for x in arr], dtype=np.float32)
    dense = (ds - ds.min()) / max(1e-6, float(ds.max() - ds.min()))
    rerank = (ce - ce.min()) / max(1e-6, float(ce.max() - ce.min()))
    final = ce_w * rerank + dense_w * dense
    return [str(arr[int(i)][0]) for i in np.argsort(-final)]


def rrf(
    rankings: list[tuple[float, dict[str, list[str]], int]], k: int = 60
) -> dict[str, list[str]]:
    qids = set().union(*(set(r.keys()) for _, r, _ in rankings))
    out = {}
    for qid in qids:
        scores: dict[str, float] = {}
        for weight, ranking_map, depth in rankings:
            for rank, did in enumerate(ranking_map.get(qid, [])[:depth], 1):
                scores[did] = scores.get(did, 0.0) + weight / (k + rank)
        out[qid] = [did for did, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]]
    return out


def cross_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    groups: list[int],
    train_qids: list[str],
    qrels: dict[str, set[str]],
) -> list[float]:
    """Measure query folds with fixed gaps; injected positives favor these scores."""
    from sklearn.model_selection import KFold

    offsets = np.cumsum([0] + groups)
    qid_to_range = {qid: (offsets[i], offsets[i + 1]) for i, qid in enumerate(train_qids)}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    standard_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_qids), 1):
        train_ranges = [qid_to_range[train_qids[i]] for i in train_idx]
        valid_ranges = [qid_to_range[train_qids[i]] for i in valid_idx]
        train_rows = np.concatenate([np.arange(a, b) for a, b in train_ranges])
        valid_rows = np.concatenate([np.arange(a, b) for a, b in valid_ranges])
        train_groups = [b - a for a, b in train_ranges]
        valid_qids = [train_qids[i] for i in valid_idx]

        model = train_ranker(
            df.iloc[train_rows][feature_cols],
            df.iloc[train_rows]["label"].to_numpy(dtype=np.int32),
            train_groups,
        )
        scores = model.predict(df.iloc[valid_rows][feature_cols])
        valid_df = df.iloc[valid_rows].copy()

        short = select_shortlist(valid_df, scores, *SHORTLIST_GAPS)
        standard_scores.append(map_at_10_standard(short, {qid: qrels[qid] for qid in valid_qids}))
        logger.info("Fold %d: internal MAP@10 %.4f", fold, standard_scores[-1])

    logger.info("Mean internal MAP@10 %.4f", np.mean(standard_scores))
    return standard_scores
