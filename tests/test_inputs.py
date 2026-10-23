import pandas as pd
import pytest

from hybrid_search.io import cache_files, check_inputs


@pytest.fixture
def inputs(tmp_path):
    pd.DataFrame({"document_id": ["d#0"], "title": ["Title"], "text": ["Passage"]}).to_csv(
        tmp_path / "corpus_100k.csv", index=False
    )
    pd.DataFrame({"query_id": ["train"], "query": ["Question"]}).to_csv(
        tmp_path / "train_queries.csv", index=False
    )
    pd.DataFrame({"query_id": ["test"], "query": ["Another question"]}).to_csv(
        tmp_path / "test_queries.csv", index=False
    )
    pd.DataFrame({"Id": ["train"], "Expected": ["['d#0']"]}).to_csv(
        tmp_path / "train_qrels.csv", index=False
    )
    for path in cache_files(tmp_path).values():
        path.write_bytes(b"not a deserializable joblib file")
    return tmp_path


def test_preflight_checks_contract_without_loading_caches(inputs):
    assert check_inputs(inputs, inputs)["train_queries.csv"] == 1


def test_preflight_rejects_split_overlap(inputs):
    pd.DataFrame({"query_id": ["train"], "query": ["Duplicate split"]}).to_csv(
        inputs / "test_queries.csv", index=False
    )
    with pytest.raises(ValueError, match="overlap"):
        check_inputs(inputs, inputs)


def test_preflight_rejects_malformed_relevance_lists(inputs):
    pd.DataFrame({"Id": ["train"], "Expected": ["not a list"]}).to_csv(
        inputs / "train_qrels.csv", index=False
    )
    with pytest.raises(ValueError, match="relevance list"):
        check_inputs(inputs, inputs)
