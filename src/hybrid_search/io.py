"""CSV and trusted local cache I/O."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import joblib
import pandas as pd


def cache_files(cache_dir: Path) -> dict[str, Path]:
    return {
        "basic": cache_dir / "basic_precision_candidates.joblib",
        "ce100k": cache_dir / "cross_encoder_dense_100k_top50_scores.joblib",
        "cefull": cache_dir / "cross_encoder_dense_top100_scores.joblib",
        "dense": cache_dir / "dense_candidates.joblib",
        "tfidf": cache_dir / "tfidf_word_titlechar_100k_candidates.joblib",
    }


def check_inputs(data_dir: Path, cache_dir: Path) -> dict[str, int]:
    """Check CSV contracts without deserializing executable joblib caches."""
    schemas = {
        "corpus_100k.csv": ("document_id", "title", "text"),
        "train_queries.csv": ("query_id", "query"),
        "test_queries.csv": ("query_id", "query"),
        "train_qrels.csv": ("Id", "Expected"),
    }
    paths = [data_dir / name for name in schemas] + list(cache_files(cache_dir).values())
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("Missing required files: " + ", ".join(missing))
    counts = {}
    query_ids = {}
    for name, columns in schemas.items():
        frame = pd.read_csv(data_dir / name, usecols=list(columns))
        key = columns[0]
        if frame.empty or frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"{name}: {key} must contain non-empty, unique IDs")
        counts[name] = len(frame)
        if name.endswith("queries.csv"):
            if frame["query"].isna().any():
                raise ValueError(f"{name}: query text is missing")
            query_ids[name] = set(frame[key].astype(str))
    if query_ids["train_queries.csv"] & query_ids["test_queries.csv"]:
        raise ValueError("Training and test query IDs overlap")
    qrels = load_qrels(data_dir)
    if set(qrels) != query_ids["train_queries.csv"] or any(not values for values in qrels.values()):
        raise ValueError("Each training query must have non-empty relevance labels")
    return counts


def load_qrels(data_dir: Path) -> dict[str, set[str]]:
    df = pd.read_csv(data_dir / "train_qrels.csv")
    result = {}
    for row in df.itertuples(index=False):
        try:
            values = ast.literal_eval(str(row.Expected))
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"Invalid relevance list for query {row.Id}") from error
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"Expected a list of document IDs for query {row.Id}")
        result[str(row.Id)] = set(values)
    return result


def load_query_sets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(data_dir / "train_queries.csv"),
        pd.read_csv(data_dir / "test_queries.csv"),
    )


def load_docs(data_dir: Path) -> tuple[dict[str, tuple[str, str]], set[str]]:
    df = pd.read_csv(data_dir / "corpus_100k.csv", usecols=["document_id", "title", "text"])
    docs = {
        str(row.document_id): (str(row.title or ""), str(row.text or ""))
        for row in df.itertuples(index=False)
    }
    return docs, set(docs)


def load_caches(cache_dir: Path) -> dict[str, dict]:
    missing = [str(path) for path in cache_files(cache_dir).values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing retrieval caches: " + ", ".join(missing))
    return {name: joblib.load(path) for name, path in cache_files(cache_dir).items()}


def write_submission(path: Path, preds: dict[str, list[str]], data_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(data_dir / "test_queries.csv")
    rows = [
        {"Id": str(row.query_id), "Predicted": repr(preds.get(str(row.query_id), [])[:10])}
        for row in test.itertuples(index=False)
    ]
    pd.DataFrame(rows, columns=["Id", "Predicted"]).to_csv(path, index=False, encoding="utf-8")
    logging.getLogger(__name__).info("Saved %d predictions to %s", len(rows), path)
