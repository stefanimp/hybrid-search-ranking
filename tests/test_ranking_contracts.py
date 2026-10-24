"""Public metric and fusion contracts, independent of the archived model outputs."""

import pytest

from hybrid_search.metrics import map_at_10_standard
from hybrid_search.ranking import rrf


def test_metric_averages_over_judged_queries_including_missing_predictions():
    # Unjudged queries must not dilute evaluation; a missing judged query must.
    predictions = {"answered": ["relevant"], "unjudged": ["anything"]}
    judgments = {"answered": {"relevant"}, "missing": {"other"}, "empty": set()}
    assert map_at_10_standard(predictions, judgments) == pytest.approx(1 / 3)


def test_metric_ignores_hits_after_rank_ten():
    predictions = {"q": [f"irrelevant-{i}" for i in range(10)] + ["relevant"]}
    assert map_at_10_standard(predictions, {"q": {"relevant"}}) == 0


def test_metric_uses_at_most_ten_relevant_documents_in_denominator():
    relevant = {f"d{i}" for i in range(15)}
    assert map_at_10_standard({"q": [f"d{i}" for i in range(10)]}, {"q": relevant}) == 1


def test_fusion_handles_disjoint_queries_and_empty_sources():
    assert rrf([]) == {}
    sources = [(1.0, {"a": ["x"], "empty": []}, 10), (1.0, {"b": ["y"]}, 10)]
    assert rrf(sources) == {"a": ["x"], "empty": [], "b": ["y"]}


def test_fusion_caps_output_even_when_sources_have_more_candidates():
    documents = [f"d{i}" for i in range(30)]
    assert rrf([(1.0, {"q": documents}, 30)]) == {"q": documents[:10]}


def test_fusion_is_invariant_to_common_positive_weight_scaling():
    sources = [(0.7, {"q": ["a", "b", "c"]}, 3), (0.3, {"q": ["c", "b"]}, 2)]
    scaled = [(weight * 17, ranking, depth) for weight, ranking, depth in sources]
    assert rrf(sources) == rrf(scaled)


def test_fusion_does_not_mutate_caller_rankings():
    ranking = {"q": ["c", "b", "a"]}
    rrf([(1.0, ranking, 2)])
    assert ranking == {"q": ["c", "b", "a"]}
