import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from upf_agent.p6_contracts import (
    EvalCase,
    percentiles,
    ranking_metrics,
    read_suite,
    validate_partition,
)
from upf_agent.p6_evaluation import reserve_heldout
from upf_agent.quality_repair import sha


def case(**changes):
    return EvalCase(
        **{
            "id": "test-1",
            "category": "knowledge",
            "split": "dev",
            "question": "测试问题",
            "source_groups": ["section-1"],
            "index_generation": "a" * 20,
            **changes,
        }
    )


def test_hit_is_not_recall_and_zero_relevance_not_accuracy():
    metrics = ranking_metrics(["a"], {"a": 1, "b": 1, "c": 1, "d": 1})
    assert metrics["hit_at_20"] == 1
    assert metrics["recall_at_20"] == 0.25
    assert ranking_metrics([], {})["recall_at_20"] is None


@pytest.mark.parametrize("error", ["duplicate_id", "duplicate_question", "group_leak"])
def test_partition_rejects_leakage(error):
    first = case()
    second = case(id="test-2", question="另一个问题", source_groups=["section-2"])
    if error == "duplicate_id":
        second.id = first.id
    elif error == "duplicate_question":
        second.question = "测 试问 题"
    else:
        second.source_groups, second.split = first.source_groups, "heldout"
    with pytest.raises(ValueError):
        validate_partition([first, second])


def test_draft_development_is_explicit_and_heldout_not_allowed(tmp_path):
    path = tmp_path / "suite.jsonl"
    path.write_text(case().model_dump_json() + "\n")
    with pytest.raises(ValueError, match="Human truth"):
        read_suite(path)
    assert len(read_suite(path, allow_drafts=True)) == 1
    path.write_text(case(split="heldout").model_dump_json() + "\n")
    with pytest.raises(ValueError, match="Heldout"):
        read_suite(path, split="heldout", allow_drafts=True)


def test_reviewed_truth_requires_provenance_and_real_topology_not_faked():
    with pytest.raises(ValidationError):
        case(label_status="reviewed")
    with pytest.raises(ValidationError):
        case(category="topology", synthetic=False)


@pytest.mark.parametrize("reviewer", ["AI辅助复核（待人工签名）", "待签名", "   "])
def test_reviewed_truth_rejects_pending_human_attribution(reviewer):
    with pytest.raises(ValidationError, match="reviewer"):
        case(label_status="reviewed", reviewer=reviewer, review_date="2026-09-28")


def test_reviewed_truth_rejects_invalid_calendar_date():
    with pytest.raises(ValidationError, match="date"):
        case(label_status="reviewed", reviewer="实际复核人", review_date="2026-02-30")


def test_nearest_rank_percentiles_report_actual_denominator():
    values = list(range(1, 21))
    assert percentiles(values) == {"n": 20, "p50": 10, "p95": 19}
    assert percentiles([]) == {"n": 0, "p50": None, "p95": None}


def test_heldout_reservation_is_atomic_and_cannot_be_retried(tmp_path, monkeypatch):
    import upf_agent.p6_evaluation as evaluation

    monkeypatch.setattr(evaluation, "pipeline_identity", lambda root: "current")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        case(
            split="heldout", label_status="reviewed", reviewer="用户", review_date="2026-09-28"
        ).model_dump_json()
        + "\n"
    )
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "status": "frozen_after_reviewed_development",
                "suite_sha256": sha(suite),
                "pipeline_identity": "current",
            }
        )
    )

    def run():
        try:
            reserve_heldout(tmp_path, suite, frozen, Path("output"))
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(lambda _: run(), range(2))) == 1
    marker = json.loads((tmp_path / "eval/p6/heldout_consumed.json").read_text())
    assert marker["consumed_before_execution"]


def test_unfrozen_or_changed_runtime_never_consumes_heldout(tmp_path, monkeypatch):
    import upf_agent.p6_evaluation as evaluation

    monkeypatch.setattr(evaluation, "pipeline_identity", lambda root: "current")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        case(
            split="heldout", label_status="reviewed", reviewer="用户", review_date="2026-09-28"
        ).model_dump_json()
        + "\n"
    )
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            {"status": "provisional", "suite_sha256": sha(suite), "pipeline_identity": "current"}
        )
    )
    with pytest.raises(ValueError, match="reviewed development"):
        reserve_heldout(tmp_path, suite, frozen, Path("output"))
    assert not (tmp_path / "eval/p6/heldout_consumed.json").exists()


def test_new_suite_gets_new_one_use_marker_without_overwriting_old(tmp_path, monkeypatch):
    import upf_agent.p6_evaluation as evaluation

    monkeypatch.setattr(evaluation, "pipeline_identity", lambda root: "current")
    marker = tmp_path / "eval/p6/heldout_consumed.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"suite_sha256": "old-suite"}))
    suite = tmp_path / "new_suite.jsonl"
    suite.write_text(
        case(
            split="heldout", label_status="reviewed", reviewer="用户", review_date="2026-10-06"
        ).model_dump_json()
        + "\n"
    )
    frozen = tmp_path / "new_frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "status": "frozen_after_reviewed_development",
                "suite_sha256": sha(suite),
                "pipeline_identity": "current",
            }
        )
    )
    assert len(reserve_heldout(tmp_path, suite, frozen, Path("new-output"))) == 1
    assert json.loads(marker.read_text()) == {"suite_sha256": "old-suite"}
    new_marker = marker.with_name(f"heldout_consumed_{sha(suite)[:16]}.json")
    assert json.loads(new_marker.read_text())["suite_sha256"] == sha(suite)
    with pytest.raises(FileExistsError):
        reserve_heldout(tmp_path, suite, frozen, Path("another-output"))


def test_machine_gate_triages_command_publication_before_human_review(tmp_path):
    from upf_agent.p6_final import machine_review_gate

    command = case(category="command", expected_extracts=["章节\nset-redis-start 1"])
    rows = [{"case": command.id, "expected_status": True, "failure": None}]
    result = tmp_path / "test-1.json"
    source = {"source_verified": True}
    result.write_text(
        json.dumps(
            {
                "result": {
                    "answer": {"claims": [{"text": "set-redis-start 1"}], "citations": [source]}
                }
            }
        )
    )
    retrieval = {"retrieval_gate_pass": True}
    sources = {"actual_source_partition_pass": True}

    assert machine_review_gate([command], rows, tmp_path, retrieval, sources, True)["pass"]
    result.write_text(
        json.dumps(
            {
                "result": {
                    "answer": {"claims": [{"text": "set-redis-start 0"}], "citations": [source]}
                }
            }
        )
    )
    failed = machine_review_gate([command], rows, tmp_path, retrieval, sources, True)
    assert failed["pass"] is False
    assert failed["human_review_action"] == "defer_until_repaired"
    assert failed["failures"] == [{"case": "test-1", "reason": "expected_command_literal_absent"}]
    rows[0]["expected_status"] = False
    assert {
        item["reason"]
        for item in machine_review_gate([command], rows, tmp_path, retrieval, sources, True)[
            "failures"
        ]
    } == {"status_or_execution_failure"}


def test_failed_development_run_writes_triage_without_blank_review(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import upf_agent.p6_final as final

    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        case(
            category="missing",
            label_status="reviewed",
            reviewer="测试复核人",
            review_date="2026-10-07",
        ).model_dump_json()
        + "\n"
    )
    output = tmp_path / "run"

    class Worker:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(final, "pipeline_identity", lambda root: "fixed")
    monkeypatch.setattr(final, "generation_preflight", lambda root, profile: {})
    monkeypatch.setattr(
        final,
        "validate_reviewed_sources",
        lambda root, cases: {"actual_source_partition_pass": True},
    )
    monkeypatch.setattr(
        final,
        "load_model_config",
        lambda path: SimpleNamespace(
            embedding=SimpleNamespace(path=tmp_path), reranker=SimpleNamespace(path=tmp_path)
        ),
    )
    monkeypatch.setattr(final, "RetrievalWorker", Worker)
    monkeypatch.setattr(final, "retrieval_summary", lambda rows: {"retrieval_gate_pass": True})
    monkeypatch.setattr(
        final,
        "run_agent",
        lambda root, task: (_ for _ in ()).throw(ValueError("simulated tool failure")),
    )
    report = final.evaluate_reviewed(tmp_path, suite, "baseline", "dev", output)
    assert report["machine_review_gate"]["pass"] is False
    assert (output / "machine_failure_triage.json").is_file()
    assert not (output / "answer_review.json").exists()


def test_pipeline_identity_includes_nested_replacement_topology(tmp_path):
    from upf_agent.p6_evaluation import pipeline_identity

    fixture = tmp_path / "eval/p6/business_v5_replacement/topology_synthetic_10.csv"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("first fixture")
    before = pipeline_identity(tmp_path)
    fixture.write_text("changed fixture")
    assert pipeline_identity(tmp_path) != before


def test_parameter_scoring_rejects_invented_optional_filters():
    from upf_agent.agent_contracts import AgentTask, ToolCall
    from upf_agent.p6_evaluation import score_parameter_response

    task = AgentTask(
        run_id="a" * 32, question="日志测试", task_type="diagnose", log_generation="a" * 20
    )
    call = ToolCall(name="find_log_events", arguments={"log_id": "a" * 20, "limit": 3})
    assert score_parameter_response(task, call, call.model_dump_json())[0]
    guessed = ToolCall(name=call.name, arguments=call.arguments | {"event_type": "guessed"})
    assert score_parameter_response(task, call, guessed.model_dump_json()) == (
        False,
        "requested_parameters_changed",
    )
    wrong_limit = ToolCall(name=call.name, arguments=call.arguments | {"limit": 4})
    assert not score_parameter_response(task, call, wrong_limit.model_dump_json())[0]


def test_human_review_import_rejects_pending_and_untranslated_corrections(tmp_path):
    from upf_agent.p6_review import import_review

    suite, review, out = (tmp_path / name for name in ("suite.jsonl", "review.md", "gold.jsonl"))
    example = case()
    suite.write_text(example.model_dump_json() + "\n")
    base = (
        "## test-1 · dev · knowledge\n"
        "- [{check}] 问题及预期事实/相关证据已核对\n"
        "- 问题：测试问题\n- 结论：{decision}\n"
        "- 复核人：用户\n- 日期：2026-09-28\n- 修正意见：{opinion}\n"
    )
    review.write_text(base.format(check=" ", decision="待复核", opinion=""))
    with pytest.raises(ValueError, match="checkbox"):
        import_review(suite, review, out)
    assert not out.exists()
    review.write_text(base.format(check="x", decision="通过", opinion="改证据"))
    with pytest.raises(ValueError, match="overrides"):
        import_review(suite, review, out)
    review.write_text(base.format(check="x", decision="通过", opinion="无"))
    receipt = import_review(suite, review, out)
    assert receipt["reviewed"] == 1
    assert read_suite(out)[0].reviewer == "用户"
    assert case().label_status == "draft"


def test_review_import_carries_previously_reviewed_development_without_reapproval(tmp_path):
    from upf_agent.p6_review import import_review

    suite = tmp_path / "combined.jsonl"
    reviewed = case(
        id="old",
        question="旧题",
        label_status="reviewed",
        reviewer="用户",
        review_date="2026-09-28",
    )
    draft = case(id="new", question="新题", split="heldout", source_groups=["section-new"])
    suite.write_text(reviewed.model_dump_json() + "\n" + draft.model_dump_json() + "\n")
    review = tmp_path / "new_review.md"
    review.write_text(
        "## new · heldout · knowledge\n"
        "- [x] 问题及预期事实/相关证据已核对\n"
        "- 问题：新题\n- 结论：通过\n- 复核人：用户\n"
        "- 日期：2026-10-06\n- 修正意见：无\n"
    )
    out = tmp_path / "reviewed.jsonl"
    receipt = import_review(suite, review, out)
    assert (receipt["carried_reviewed"], receipt["newly_reviewed"]) == (1, 1)
    assert read_suite(out, "dev") == [reviewed]
    assert read_suite(out, "heldout")[0].reviewer == "用户"


def test_current_topology_fixture_is_valid_but_not_real(tmp_path):
    from datetime import datetime

    from upf_agent.topology import import_topology, lookup_topology

    root = Path(__file__).resolve().parents[1]
    source = root / "eval/p6/topology_synthetic_0.csv"
    result = import_topology(source, tmp_path, synthetic=True)
    at = datetime.fromisoformat("2025-06-01T00:00:00+00:00")
    directory = tmp_path / result["generation"]
    present = lookup_topology(directory, "SYN-UPF-0", "N3", at)
    assert len(present.data["edges"]) == 1
    assert any("Synthetic fixture" in warning for warning in present.warnings)
    assert not lookup_topology(directory, "SYN-UPF-0", "N30", at).data.get("edges")
    after = datetime.fromisoformat("2026-06-01T00:00:00+00:00")
    expired = lookup_topology(directory, "SYN-UPF-0", "N3", after)
    assert not expired.data.get("edges")
    assert expired.status.value == "missing_source"
    assert len(expired.data["excluded_edges"]) == 1
    assert expired.data["excluded_edges"][0]["exclusion_reason"] == "at_or_after_valid_to"
    assert expired.data["excluded_edges"][0]["interface_type"] == "N3"
    assert expired.data["excluded_edges"][0]["synthetic"] is True
    assert not lookup_topology(directory, "SYN-UPF-0", "N30", after).data["excluded_edges"]


def test_answer_adjudication_requires_complete_run_bound_human_decisions(tmp_path):
    from upf_agent.p6_review import adjudicate_answers

    suite, summary, review, output = (
        tmp_path / name for name in ("suite.jsonl", "summary.json", "review.json", "result.json")
    )
    examples = [
        case(category="command"),
        case(
            id="missing-1", category="missing", question="缺资料吗", source_groups=["missing-group"]
        ),
    ]
    suite.write_text("\n".join(item.model_dump_json() for item in examples) + "\n")
    summary.write_text(
        json.dumps(
            {
                "suite_sha256": sha(suite),
                "split": "dev",
                "rows": [{"case": item.id} for item in examples],
            }
        )
    )
    rows = [
        {
            "case": item.id,
            "supported": True,
            "complete": True,
            "command_exact": True,
            "unsupported_critical_parameter": False,
            "missing_source_correct": True,
            "reviewer": "用户",
            "date": "2026-09-28",
        }
        for item in examples
    ]
    review.write_text(json.dumps({"summary_sha256": "changed", "rows": rows}))
    with pytest.raises(ValueError, match="does not match"):
        adjudicate_answers(suite, summary, review, output)
    rows[0]["supported"] = None
    review.write_text(json.dumps({"summary_sha256": sha(summary), "rows": rows}))
    with pytest.raises(ValueError, match="true/false"):
        adjudicate_answers(suite, summary, review, output)
    rows[0]["supported"] = True
    review.write_text(json.dumps({"summary_sha256": sha(summary), "rows": rows}))
    result = adjudicate_answers(suite, summary, review, output)
    assert result["semantic_gate_pass"]
    assert result["support"] == {"n": 2, "pass": 2, "rate": 1.0}
    assert result["M6_pass"] is False


def test_failed_heldout_cannot_trigger_full_human_scoring(tmp_path):
    from upf_agent.p6_review import adjudicate_answers

    suite, summary, review, output = (
        tmp_path / name for name in ("suite.jsonl", "summary.json", "review.json", "result.json")
    )
    example = case(split="heldout")
    suite.write_text(example.model_dump_json() + "\n")
    summary.write_text(
        json.dumps(
            {
                "suite_sha256": sha(suite),
                "split": "heldout",
                "rows": [{"case": example.id, "expected_status": False}],
            }
        )
    )
    review.write_text(json.dumps({"summary_sha256": sha(summary), "rows": []}))
    with pytest.raises(ValueError, match="machine gate failed"):
        adjudicate_answers(suite, summary, review, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["ai_assisted", None])
def test_ai_assisted_scores_do_not_grant_human_acceptance(tmp_path, kind):
    from upf_agent.p6_review import adjudicate_answers

    suite, summary, review, output = (
        tmp_path / name for name in ("suite.jsonl", "summary.json", "review.json", "result.json")
    )
    example = case()
    suite.write_text(example.model_dump_json() + "\n")
    summary.write_text(
        json.dumps({"suite_sha256": sha(suite), "split": "dev", "rows": [{"case": example.id}]})
    )
    decision = {
        "summary_sha256": sha(summary),
        "rows": [
            {
                "case": example.id,
                "supported": True,
                "complete": True,
                "reviewer": "豆包（AI辅助复核）",
                "date": "2026-09-29",
            }
        ],
    }
    if kind:
        decision["review_kind"] = kind
    review.write_text(json.dumps(decision))
    result = adjudicate_answers(suite, summary, review, output)
    assert result["review_kind"] == "ai_assisted"
    assert result["review_complete"] is True
    assert result["human_review_complete"] is False
    assert result["support"] == {"n": 1, "pass": 1, "rate": 1.0}
    assert result["M6_pass"] is False


@pytest.mark.parametrize(
    "profile,structured,expected",
    [
        ("baseline", True, None),
        ("target", True, {"enable_thinking": False}),
        ("target", False, {"reasoning_effort": "low"}),
    ],
)
def test_short_structured_nodes_use_the_correct_model_template(
    monkeypatch, profile, structured, expected
):
    import httpx

    from upf_agent.agent_contracts import AgentData, AgentTask
    from upf_agent.agent_worker import CountedClient
    from upf_agent.models import LocalChatClient

    observed = []

    def handle(request):
        observed.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    original = LocalChatClient.__init__

    def init(self, model, **kwargs):
        original(self, model, transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(LocalChatClient, "__init__", init)
    root = Path(__file__).resolve().parents[1]
    task = AgentTask(
        run_id="a" * 32, question="JSON调用", task_type="manual", model_profile=profile
    )
    with CountedClient(root, AgentData(task=task), 10, 1) as client:
        kwargs = {"response_format": {"type": "json_object"}} if structured else {}
        client.chat([{"role": "user", "content": "输出JSON"}], max_tokens=256, **kwargs)
        assert client.stats["requests"] == 1
    assert observed[0].get("chat_template_kwargs") == expected
    assert observed[0]["max_tokens"] == 256


@pytest.mark.parametrize(
    "profile,policy,expected",
    [
        ("baseline", "auto", None),
        ("target", "auto", {"reasoning_effort": "low"}),
        ("target", "off", {"enable_thinking": False}),
    ],
)
def test_generation_benchmark_preserves_controls_and_records_thinking_policy(
    monkeypatch, profile, policy, expected
):
    import httpx

    from upf_agent.p6_evaluation import stream_probe

    observed = []

    def handle(request):
        observed.append(json.loads(request.content))
        event = {"choices": [{"delta": {"content": "已连接"}, "finish_reason": "stop"}]}
        return httpx.Response(200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n")

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs)
    )
    root = Path(__file__).resolve().parents[1]
    result = stream_probe(root, profile, "本地测试", thinking_policy=policy)
    assert observed[0].get("chat_template_kwargs") == expected
    assert observed[0]["max_tokens"] == 256
    assert observed[0]["stream"] is True
    assert result["finish_reason"] == "stop"
    assert result["content"] == "已连接"
    assert result["first_visible_ms"] is not None


@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
def test_reasoning_stream_is_observed_but_never_counted_as_visible_answer(
    monkeypatch, reasoning_field
):
    import httpx

    from upf_agent.p6_evaluation import stream_probe

    event = {"choices": [{"delta": {reasoning_field: "内部思考"}, "finish_reason": "length"}]}
    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text="data: " + json.dumps(event) + "\n")
            ),
            **kwargs,
        ),
    )
    result = stream_probe(Path(__file__).resolve().parents[1], "target", "测试")
    assert result["first_model_event_ms"] is not None
    assert result["first_visible_ms"] is None
    assert not result["content"]
    assert result["finish_reason"] == "length"
