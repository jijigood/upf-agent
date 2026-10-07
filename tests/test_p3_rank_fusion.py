import pytest

from upf_agent.p3_search import combine_rerank


def test_combined_rank_preserves_strong_multi_route_support():
    items = [
        {"chunk_id": "a", "rrf_score": 3 / 61, "rerank_raw_logit_difference": 1},
        {"chunk_id": "b", "rrf_score": 1 / 61, "rerank_raw_logit_difference": 10},
    ]
    combined = combine_rerank(items, 60)
    assert [item["chunk_id"] for item in combined] == ["a", "b"]
    assert combined[0]["model_rerank_rank"] == 2
    assert combined[0]["combined_score"] == pytest.approx(3 / 61 + 0.5 / 62)


def test_combined_rank_uses_model_to_break_equal_retrieval_support():
    items = [
        {"chunk_id": "a", "rrf_score": 0.02, "rerank_raw_logit_difference": 1},
        {"chunk_id": "b", "rrf_score": 0.02, "rerank_raw_logit_difference": 10},
    ]
    assert [item["chunk_id"] for item in combine_rerank(items, 60)] == ["b", "a"]


def test_combined_rank_is_deterministic_and_validates_k():
    items = [
        {"chunk_id": "b", "rrf_score": 0.02, "rerank_raw_logit_difference": 1},
        {"chunk_id": "a", "rrf_score": 0.02, "rerank_raw_logit_difference": 1},
    ]
    assert [item["chunk_id"] for item in combine_rerank(items, 60)] == ["a", "b"]
    with pytest.raises(ValueError):
        combine_rerank(items, 0)
