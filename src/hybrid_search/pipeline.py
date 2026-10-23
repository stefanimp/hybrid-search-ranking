"""Train, rank and export the tournament pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from .dataset import build_dataset, features_only
from .io import load_caches, load_docs, load_qrels, load_query_sets, write_submission
from .ranking import SHORTLIST_GAPS, cross_validate, rank_ce, rrf, select_shortlist, train_ranker

logger = logging.getLogger(__name__)


def _rank_by_score(items: list, *, basic: bool = False) -> list[str]:
    if not items:
        return []
    scores = np.asarray(
        [float(item.get("basic_score", 0.0)) if basic else item[1] for item in items],
        dtype=np.float32,
    )
    return [
        str(items[int(i)]["document_id"] if basic else items[int(i)][0])
        for i in np.argsort(-scores)
    ]


def fuse_predictions(
    shortlists: dict[str, list[str]],
    query_ids: list[str],
    caches: dict[str, dict],
    valid_docs: set[str],
    *,
    include_variants: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """Apply the submitted fusion weights, preserving shortlist lengths and tie order."""
    rankings = {}
    for source in ["cefull", "ce100k", "dense"]:
        rankings[source] = {}
        for qid in query_ids:
            items = [item for item in caches[source].get(qid, []) if str(item[0]) in valid_docs]
            if source == "dense":
                rankings[source][qid] = _rank_by_score(items)
            else:
                weights = (0.70, 0.30) if source == "cefull" else (0.50, 0.50)
                rankings[source][qid] = rank_ce(items, *weights)
    sources = [
        (2.00, shortlists, 10),
        (1.50, rankings["dense"], 10),
        (1.00, rankings["cefull"], 40),
        (0.65, rankings["ce100k"], 40),
        (0.40, rankings["dense"], 100),
    ]

    def fuse(extra: list) -> dict[str, list[str]]:
        return {
            qid: documents[: len(shortlists.get(qid, documents[:1]))]
            for qid, documents in rrf(sources + extra).items()
        }

    outputs = {"submision_final.csv": fuse([])}
    if include_variants:
        auxiliary = {}
        for source in ["tfidf", "basic"]:
            auxiliary[source] = {}
            for qid in query_ids:
                items = [
                    item
                    for item in caches[source].get(qid, [])
                    if str(item.get("document_id") if source == "basic" else item[0]) in valid_docs
                ]
                auxiliary[source][qid] = _rank_by_score(items, basic=source == "basic")
        for number, source, weight in [(1, "tfidf", 0.20), (2, "tfidf", 0.60), (3, "basic", 0.30)]:
            outputs[f"submision_pruebas_{number}.csv"] = fuse([(weight, auxiliary[source], 100)])
    return outputs


def run(
    data_dir: Path, cache_dir: Path, output_dir: Path, *, include_variants: bool = False
) -> Path:
    """Validate by query, fit the final ranker and export its predictions."""
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    docs, valid_docs = load_docs(data_dir)
    caches = load_caches(cache_dir)
    qrels = load_qrels(data_dir)
    train_queries, test_queries = load_query_sets(data_dir)
    query_text = {
        str(row.query_id): str(row.query)
        for frame in [train_queries, test_queries]
        for row in frame.itertuples(index=False)
    }
    train_ids = train_queries.query_id.astype(str).tolist()
    test_ids = test_queries.query_id.astype(str).tolist()

    logger.info("Building training pairs for %d queries", len(train_ids))
    train_df, _, groups = build_dataset(train_ids, query_text, docs, valid_docs, caches, qrels)
    columns = features_only(train_df)
    logger.info("Training matrix: %d pairs, %d features", len(train_df), len(columns))
    cross_validate(train_df, columns, groups, train_ids, qrels)
    model = train_ranker(train_df[columns], train_df["label"].to_numpy(dtype=np.int32), groups)
    joblib.dump(
        {"model": model, "feature_cols": columns, "avg_gaps": SHORTLIST_GAPS},
        model_dir / "modelo_final.joblib",
        compress=3,
    )

    test_df, _, _ = build_dataset(test_ids, query_text, docs, valid_docs, caches)
    features = test_df.reindex(columns=columns, fill_value=0.0)
    shortlists = select_shortlist(test_df, model.predict(features), *SHORTLIST_GAPS)
    outputs = fuse_predictions(
        shortlists, test_ids, caches, valid_docs, include_variants=include_variants
    )
    for name, predictions in outputs.items():
        write_submission(output_dir / name, predictions, data_dir)
    return output_dir / "submision_final.csv"
