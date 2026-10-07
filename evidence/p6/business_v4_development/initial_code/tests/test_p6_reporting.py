from upf_agent.p6_reporting import retrieval_summary


def test_human_acceptance_cannot_skip_failed_retrieval_gates(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import pytest

    import upf_agent.p6_final as final
    from upf_agent.quality_repair import sha

    suite, development, decision, output = [
        tmp_path / name for name in ("suite.jsonl", "dev.json", "decision.json", "frozen.json")
    ]
    suite.write_text("test fixture")
    monkeypatch.setattr(final, "read_suite", lambda *args: [None] * 60)
    monkeypatch.setattr(final, "pipeline_identity", lambda root: "current")
    monkeypatch.setattr(
        final, "load_model_config", lambda path: SimpleNamespace(generation={"baseline": None})
    )
    monkeypatch.setattr(final, "expected_configuration", lambda model: {})
    development.write_text(
        json.dumps(
            {
                "suite_sha256": sha(suite),
                "pipeline_identity": "current",
                "label_mode": "reviewed",
                "split": "dev",
                "cases": 60,
                "execution_complete": True,
                "model_profile": "baseline",
                "server_configuration": {},
                "retrieval": {"retrieval_gate_pass": False},
            }
        )
    )
    decision.write_text(json.dumps({"configuration_accepted": True}))
    with pytest.raises(ValueError, match="Recall/nDCG gates"):
        final.freeze_configuration(tmp_path, suite, development, decision, output)
    assert not output.exists()


def test_failed_retrieval_cannot_disappear_from_formal_denominator():
    rows = [
        {
            "category": "knowledge",
            "retrieval_metrics": {
                "reranked": {"recall_at_20": 1.0, "ndcg_at_8": 1.0},
                "rrf": {"recall_at_20": 1.0, "ndcg_at_8": 0.5},
            },
        },
        {"category": "command"},
        {"category": "missing"},
    ]
    result = retrieval_summary(rows)
    assert result["routes"]["reranked"] == {
        "n": 2,
        "failed_or_unscored": 1,
        "recall_at_20": 0.5,
        "ndcg_at_8": 0.5,
    }
    assert result["retrieval_gate_pass"] is False


def test_reranker_regression_blocks_configuration_freezing():
    rows = [
        {
            "category": "knowledge",
            "retrieval_metrics": {
                "reranked": {"recall_at_20": 1.0, "ndcg_at_8": 0.5},
                "rrf": {"recall_at_20": 1.0, "ndcg_at_8": 1.0},
            },
        }
    ]
    assert retrieval_summary(rows)["retrieval_gate_pass"] is False
    rows[0]["retrieval_metrics"]["reranked"]["ndcg_at_8"] = 1.0
    assert retrieval_summary(rows)["retrieval_gate_pass"] is True
