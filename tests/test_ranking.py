"""Contract tests for ranking semantics and differential legacy parity."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hybrid_search.features import content_features
from hybrid_search.metrics import map_at_10_standard
from hybrid_search.ranking import rrf, select_shortlist
from hybrid_search.retrieval import build_candidates_for_query
from hybrid_search.text import norm01


@pytest.fixture(scope="module")
def reference():
    return json.loads((Path(__file__).parent / "fixtures/ranking.json").read_text(encoding="utf-8"))


def test_reciprocal_fusion_prefers_agreement():
    sources = [(1.0, {"q": ["a", "b"]}, 2), (1.0, {"q": ["b", "c"]}, 2)]
    assert rrf(sources) == {"q": ["b", "a", "c"]}


def test_fusion_respects_depth_and_tie_order():
    assert rrf([(1.0, {"q": ["a", "b"]}, 1), (1.0, {"q": ["c"]}, 1)]) == {"q": ["a", "c"]}


@pytest.mark.parametrize(
    "scores, expected",
    [
        ([1.0, 0.83, 0.81], ["a", "b"]),
        ([1.0, 0.91, 0.90], ["a", "b", "c"]),
        ([1.0, 0.80, 0.79], ["a"]),
    ],
)
def test_shortlist_thresholds(scores, expected):
    frame = pd.DataFrame({"qid": ["q"] * 3, "doc_id": ["a", "b", "c"]})
    assert select_shortlist(frame, np.array(scores)) == {"q": expected}


def test_average_precision_is_rank_sensitive():
    assert map_at_10_standard({"q": ["wrong", "right"]}, {"q": {"right"}}) == 0.5


def test_average_precision_does_not_count_duplicate_hits():
    assert map_at_10_standard({"q": ["right", "right"]}, {"q": {"right"}}) == 1.0
    assert map_at_10_standard({}, {}) == 0.0


def test_constant_scores_have_zero_normalized_features():
    np.testing.assert_array_equal(norm01([5, 5]), [0, 0])


def test_content_features_match_original_fixture(reference):
    for case in reference["features"]:
        assert content_features(*case["args"]) == case["expected"]


def test_candidate_sources_match_original_fixture_and_filter_unknown_documents(reference):
    caches = {
        "basic": {
            "q": [
                {"document_id": "a", "basic_score": 3, "best_bm25": 2, "sources": {"bm25-title": 1}}
            ]
        },
        "ce100k": {"q": [("b", 0.2, 0.8), ("unknown", 0.9, 0.9)]},
        "cefull": {"q": [("a", 0.5, 0.6)]},
        "dense": {"q": [("b", 0.8), ("a", 0.3)]},
        "tfidf": {"q": [("a", 0.5, 0.4, 0.3)]},
    }
    actual = build_candidates_for_query("q", caches, {"a", "b"})
    assert actual == reference["candidates"]["expected"]
    assert set(actual) == {"a", "b"}


def test_fusion_returns_shortlist_length(reference):
    from hybrid_search.pipeline import fuse_predictions

    outputs = fuse_predictions(
        {"q": ["a"]}, ["q"], reference["candidates"]["caches"], {"a", "b"}, include_variants=True
    )
    assert len(outputs) == 4
    assert all(len(ranking["q"]) == 1 for ranking in outputs.values())
