import json
from pathlib import Path

import httpx
import pytest

from upf_agent.v8.phase2 import METHODS, run_phase2
from upf_agent.v8.plan_generator import HEAD_VERSION, ObservedPlanClient
from upf_agent.v8.plan_ir import PlanIR, PlanStep

PROJECT = Path(__file__).resolve().parents[1]


def response(request):
    body = json.loads(request.content)
    context = json.loads(body["messages"][1]["content"])
    if "trusted_header" not in context:
        calls = {c["name"] for c in context["calls"]}
        actions = (
            []
            if "search_manual" in calls
            else [
                {"name": "search_manual", "arguments": {"query": context["question"]}},
                {
                    "name": "lookup_topology",
                    "arguments": {
                        "topology_id": context["confirmed_task"]["topology_generation"],
                        "device_id": "UPF-A",
                    },
                },
            ]
        )
        content = json.dumps(
            {"actions": actions, "done": not actions, "questions": [], "candidates": []}
        )
    else:
        header = context["trusted_header"]
        steps = []
        for index, obj in enumerate(header["target_objects"]):
            steps.append(
                PlanStep(
                    step_id=f"id{index}", text="核对身份", action_type="CHECK_IDENTITY", target=obj
                )
            )
            steps.append(
                PlanStep(
                    step_id=f"check{index}",
                    text="检查N4接口",
                    action_type="CHECK_INTERFACE",
                    target=obj,
                    parameters={"interface": "N4"},
                    depends_on=[f"id{index}"],
                )
            )
        content = PlanIR(**header, steps=steps, stop_conditions=["缺证则停止"]).model_dump_json()
    return httpx.Response(
        200,
        json={
            "id": "test-request",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        },
    )


def fake_client(model, collector, budget, **kwargs):
    return ObservedPlanClient(
        model, collector, budget, transport=httpx.MockTransport(response), **kwargs
    )


def test_four_method_pipeline_preserves_sources_costs_raw_outputs_and_no_graph(tmp_path):
    output = tmp_path / "pilot"
    summary = run_phase2(PROJECT, output, 1, client_factory=fake_client)
    assert summary["execution_pass"] and summary["method_runs"] == 4
    assert summary["expert_review_pending"] and not summary["reviewed"]
    assert all(
        s["vpr"] is None and s["vpr_denominator"] == 0 for s in summary["summaries"].values()
    )
    for method in range(1, 5):
        folder = output / "runs/pilot-01" / f"method-{method}"
        raw = json.loads((folder / "raw-output.json").read_text())
        assert (folder / "raw-plan.txt").read_text() == raw["content"]
        assert (folder / "translation.json").exists() and (folder / "published-plan.json").exists()
        metrics = json.loads((folder / "metrics.json").read_text())
        assert metrics["token_usage"]["input"] is not None
        assert metrics["invocations"]["tool"]["attempted"] <= 8
        assert metrics["invocations"]["model"]["attempted"] <= 8
    p5 = output / "runs/pilot-01/method-2"
    native = json.loads((p5 / "p5-native.json").read_text())
    assert native["rounds"] == 2
    assert any(r["call"]["name"] == "search_manual" for r in native["receipts"])
    p5_trace = [json.loads(line) for line in (p5 / "model-traces.jsonl").read_text().splitlines()]
    assert [t["operation"] for t in p5_trace] == ["p5_plan", "p5_plan", "common_plan_head"]
    for entry in p5_trace:
        assert "case-reference" not in json.dumps(entry["request"])
        assert "required_actions" not in json.dumps(entry["request"])
    no_graph = output / "runs/pilot-01/method-3"
    full = output / "runs/pilot-01/method-4"
    assert not list(no_graph.glob("runtime/runs/v8/*/subgraph-*.json"))
    assert list(full.glob("runtime/runs/v8/*/subgraph-*.json"))
    bindings = json.loads((full / "bindings.json").read_text())
    graph = [b for b in bindings if ":graph:" in b["evidence_id"]]
    assert graph and all(b["knowledge_applicability"]["status"] == "mapped" for b in graph)
    assert all(not b["knowledge_applicability"]["reviewed"] for b in graph)
    for folder in [no_graph, full]:
        native = json.loads((folder / "v8-native.json").read_text())
        assert "scope_denied" not in native["query_plan"]["gaps"]
        actual = {(b["object_id"], b["category"]) for b in native["bindings"]}
        assert {(obj, kind) for obj in ["UPF-A", "SMF-B"] for kind in ["K", "R", "P"]} <= actual
        traces = [
            json.loads(line) for line in (folder / "model-traces.jsonl").read_text().splitlines()
        ]
        head = next(t for t in traces if t["operation"] == "common_plan_head")
        request = json.loads(head["request"]["messages"][1]["content"])
        assert request["generation_capabilities"]["unresolved"] == []
        assert all(s["code"] == "satisfied" for s in native["states"].values())
    from upf_agent.v8.plan_cases import SPECS

    assert len(SPECS) == 24


def test_http_failure_kept_as_generation_error(tmp_path):
    def failing(model, collector, budget, **kwargs):
        return ObservedPlanClient(
            model,
            collector,
            budget,
            transport=httpx.MockTransport(lambda r: httpx.Response(503)),
            **kwargs,
        )

    summary = run_phase2(PROJECT, tmp_path / "failed", 1, client_factory=failing)
    assert not summary["execution_pass"]
    assert summary["generation_errors"] == 4
    assert summary["summaries"][METHODS[0]]["parse_failure_count"] == 1
    assert len(json.loads((tmp_path / "failed/failures.json").read_text())) == 4


def test_output_collision_does_not_override_history(tmp_path):
    with pytest.raises(ValueError, match="output exists"):
        run_phase2(PROJECT, tmp_path, 1, client_factory=fake_client)


def test_selected_development_cases_keep_identity_calibration_and_frozen_head(tmp_path):
    output = tmp_path / "selected"
    summary = run_phase2(PROJECT, output, case_ids=[15, 1, 13, 5, 23], client_factory=fake_client)
    assert summary["method_runs"] == 20 and summary["execution_pass"]
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["selected_case_ids"] == [1, 5, 13, 15, 23]
    assert protocol["common_head"] == HEAD_VERSION
    calibration = json.loads((output / "calibration-status.json").read_text())
    assert calibration["candidate_cases"] == ["pilot-01", "pilot-05"]
    assert calibration["reviews"]["paired_cases"] == 0
    assert {p.name for p in (output / "cases").iterdir()} == {
        "pilot-01",
        "pilot-05",
        "pilot-13",
        "pilot-15",
        "pilot-23",
    }
    manifest = json.loads((output / "common-head.json").read_text())
    for trace_file in (output / "runs").glob("*/*/model-traces.jsonl"):
        traces = [json.loads(line) for line in trace_file.read_text().splitlines()]
        head = next(t for t in traces if t["operation"] == "common_plan_head")
        assert head["request"]["messages"][0]["content"] == manifest["instructions"]
        context = json.loads(head["request"]["messages"][1]["content"])
        schema = head["request"]["response_format"]["json_schema"]["schema"]
        from upf_agent.v8.models import digest

        assert "plan_schema" not in context
        assert context["generation_schema_sha256"] == digest(schema)
        assert head["request"]["response_format"]["type"] == "json_schema"
        for branch in schema["$defs"]["PlanStep"]["anyOf"]:
            assert {"depends_on", "evidence", "permission_kind", "effect_kind"} <= set(
                branch["required"]
            )
            action = branch["properties"]["action_type"]["const"]
            spec = context["action_registry"]["actions"][action]
            assert branch["properties"]["permission_kind"] == {"const": spec["permission_kind"]}
            assert branch["properties"]["effect_kind"] == {"const": spec["effect_kind"]}
        for native in traces:
            if native["operation"] != "common_plan_head":
                assert native["request"]["response_format"]["type"] == "json_object"
        assert "case-reference" not in json.dumps(context)
        for entry in context["evidence"]:
            assert entry["citation"] == {
                "evidence_id": entry["evidence_id"],
                "revision": entry["revision"],
                "source_sha256": entry["source"]["sha256"],
                "locator": entry["source"]["locator"],
            }


@pytest.mark.parametrize("case_ids", [[], [1, 1], [0], [25], [True]])
def test_invalid_case_selection_fails_before_creating_experiment(tmp_path, case_ids):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="case_ids"):
        run_phase2(PROJECT, output, case_ids=case_ids, client_factory=fake_client)
    assert not output.exists()


def test_case_selection_cannot_be_combined_with_count(tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="cannot be combined"):
        run_phase2(PROJECT, output, 2, case_ids=[1], client_factory=fake_client)
    assert not output.exists()


def test_grounded_schema_preserves_unknown_evidence_without_certifying_eligibility(tmp_path):
    from test_v8_rules import prepared

    from upf_agent.v8.plan_generator import generation_schema

    task, _, bindings = prepared(tmp_path)
    unknown = next(b for b in bindings if b.category == "K").model_copy(deep=True)
    unknown.evidence_id = "unknown-purpose-input"
    unknown.purposes = []
    schema = generation_schema([unknown], {"target_objects": task.objects})
    variants = schema["$defs"]["EvidenceReference"]["anyOf"]
    assert len(variants) == 1
    values = {key: prop["const"] for key, prop in variants[0]["properties"].items()}
    assert values == {
        "evidence_id": unknown.evidence_id,
        "revision": unknown.revision,
        "source_sha256": unknown.source.sha256,
        "locator": unknown.source.locator,
    }
    assert all(
        branch["properties"]["target"] == {"enum": task.objects}
        for branch in schema["$defs"]["PlanStep"]["anyOf"]
    )


def test_empty_evidence_schema_forbids_invented_references():
    from upf_agent.v8.plan_generator import generation_schema

    schema = generation_schema([])
    for branch in schema["$defs"]["PlanStep"]["anyOf"]:
        assert branch["properties"]["evidence"]["maxItems"] == 0
    assert schema["properties"]["steps"]["maxItems"] == 8


def test_budget_reserves_common_head_and_blocks_without_dispatch(tmp_path):
    from upf_agent.config import load_model_config
    from upf_agent.v8.experiment_metrics import MetricCollector
    from upf_agent.v8.fixtures import example
    from upf_agent.v8.plan_generator import PlanBudget

    task = example(tmp_path)
    task.budget.max_model_calls = 2
    collector = MetricCollector(task.run_id)
    budget = PlanBudget(task, collector)
    model = load_model_config(PROJECT / "config/models.yaml").generation["baseline"]
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )
    )
    with ObservedPlanClient(model, collector, budget, transport=transport) as client:
        client.complete([], operation="p5_plan", reserve_head=True)
        with pytest.raises(TimeoutError):
            client.complete([], operation="p5_plan", reserve_head=True)
        client.complete([], operation="common_plan_head")
        with pytest.raises(TimeoutError):
            client.complete([], operation="common_plan_head")
    assert collector.summary()["invocations"]["model"]["attempted"] == 2
    assert collector.summary()["invocations"]["model"]["blocked"] == 2
    assert collector.summary()["token_usage"]["input"] == 6


def test_all_24_case_references_frozen_unreviewed_and_grouped(tmp_path):
    from upf_agent.v8.plan_cases import prepare_case

    cases = [prepare_case(tmp_path / f"case-{i}", i) for i in range(24)]
    assert len({gold.case_id for _, gold in cases}) == 24
    assert all(gold.split == "dev" and gold.synthetic and not gold.reviewed for _, gold in cases)
    assert all(not task.model_enabled for task, _ in cases)
    assert all(gold.root_cause_truth == "unknown" for _, gold in cases)


def test_elapsed_preserves_deadline_overrun_while_allowance_blocks_dispatch(tmp_path, monkeypatch):
    from upf_agent.v8 import plan_generator
    from upf_agent.v8.experiment_metrics import MetricCollector
    from upf_agent.v8.fixtures import example

    now = {"seconds": 0.0}
    monkeypatch.setattr(plan_generator, "perf_counter", lambda: now["seconds"])
    task = example(tmp_path)
    collector = MetricCollector(task.run_id)
    budget = plan_generator.PlanBudget(task, collector)
    now["seconds"] = task.budget.timeout_seconds + 0.5
    assert budget.remaining() == 0
    assert budget.elapsed() == task.budget.timeout_seconds + 0.5
    with pytest.raises(TimeoutError):
        budget.check("model", {"operation": "head"})
    assert collector.summary()["invocations"]["model"]["attempted"] == 0
    assert collector.summary()["invocations"]["model"]["blocked"] == 1


def test_p5_cache_reuse_is_recorded_without_another_backend_dispatch(tmp_path):
    def repeat_response(request):
        body = json.loads(request.content)
        context = json.loads(body["messages"][1]["content"])
        if "trusted_header" in context:
            return response(request)
        content = json.dumps(
            {
                "actions": [{"name": "search_manual", "arguments": {"query": context["question"]}}],
                "done": False,
                "questions": [],
                "candidates": [],
            }
        )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )

    def repeat_client(model, collector, budget, **kwargs):
        return ObservedPlanClient(
            model, collector, budget, transport=httpx.MockTransport(repeat_response), **kwargs
        )

    output = tmp_path / "cache"
    run_phase2(PROJECT, output, 1, client_factory=repeat_client)
    metrics = json.loads((output / "runs/pilot-01/method-2/metrics.json").read_text())
    assert metrics["invocations"]["tool"]["attempted"] == 2
    assert metrics["invocations"]["tool"]["reused"] == 1


def test_graph_knowledge_reference_is_checked_against_the_actual_source(tmp_path):
    from upf_agent.v8.compiler import compile_contract
    from upf_agent.v8.models import EvaluationContext, EvidenceBinding, OrchestrationTask, digest
    from upf_agent.v8.plan_ir import EvidenceReference
    from upf_agent.v8.plan_validator import validate_plan

    output = tmp_path / "source"
    run_phase2(PROJECT, output, 1, client_factory=fake_client)
    folder = output / "runs/pilot-01/method-4"
    bindings = [
        EvidenceBinding.model_validate(b)
        for b in json.loads((folder / "bindings.json").read_text())
    ]
    task = OrchestrationTask.model_validate_json((folder / "task.json").read_text())
    context = EvaluationContext.model_validate_json((folder / "context.json").read_text())
    knowledge = next(b for b in bindings if ":graph:" in b.evidence_id and b.object_id == "UPF-A")
    plan = PlanIR(
        context_fingerprint=context.fingerprint,
        scope_fingerprint=digest(task.permission),
        batch_id=context.batch_id,
        evaluation_as_of=context.as_of,
        target_objects=task.objects,
        steps=[
            PlanStep(
                step_id="config",
                text="只读检查配置",
                action_type="CHECK_CONFIGURATION",
                target="UPF-A",
                parameters={"interface": "N4", "field": "n4_enabled"},
                evidence=[
                    EvidenceReference(
                        evidence_id=knowledge.evidence_id,
                        revision=1,
                        source_sha256=knowledge.source.sha256,
                        locator=knowledge.source.locator,
                    )
                ],
            )
        ],
        stop_conditions=["未知则停止"],
    )
    assert validate_plan(
        plan, bindings, task, compile_contract(task), context, folder / "runtime"
    ).raw_plan_validity


def test_no_graph_never_invokes_the_graph_node(tmp_path, monkeypatch):
    from upf_agent.v8.engine import OrchestrationEngine
    from upf_agent.v8.phase1 import prepare_synthetic_root

    task = prepare_synthetic_root(tmp_path / "root")
    task.graph_enabled = False
    engine = OrchestrationEngine(tmp_path / "root")
    monkeypatch.setattr(
        engine, "_graph", lambda state: (_ for _ in ()).throw(AssertionError("graph called"))
    )
    try:
        result = engine.run(task)
        assert result["status"] == "completed"
        assert engine.ledger.metrics(task.run_id)["invocations"]["tool"]["attempted"] == 2
    finally:
        engine.close()
