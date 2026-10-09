from copy import deepcopy
from datetime import timedelta

import pytest

from upf_agent.v8.actions import action_templates, full_actions, gate
from upf_agent.v8.adapters import collect
from upf_agent.v8.compiler import compile_contract, compile_query_plan
from upf_agent.v8.dependency import LocalEvaluator, semantic_results
from upf_agent.v8.evaluator import EvidenceSnapshot, full_evaluate, route_gaps
from upf_agent.v8.fixtures import example
from upf_agent.v8.models import Code, EvidenceBinding


def prepared(root, template="upf_n4_fault"):
    task = example(root, template=template)
    contract = compile_contract(task)
    plan = compile_query_plan(task, contract)
    bindings = []
    for step in plan.steps:
        result = collect(root, task, contract, step, step.step_id)
        bindings.extend(EvidenceBinding.model_validate(b) for b in result["bindings"])
    return task, contract, bindings


@pytest.mark.parametrize(
    "template", ["upf_n4_fault", "upf_n3_fault", "manual_operation", "log_observation"]
)
def test_four_complete_templates(tmp_path, template):
    task, contract, bindings = prepared(tmp_path, template)
    states = full_evaluate(task, contract, bindings, "a")
    assert all(s.code == Code.SATISFIED for s in states.values())
    assert all(
        a.outcome == "suggest_allowed"
        for a in full_actions(task, contract, bindings, states, "a").values()
    )


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("wrong_version", Code.NOT_APPLICABLE),
        ("unknown_version", Code.UNKNOWN),
        ("no_identity", Code.MISSING),
        ("no_window", Code.UNKNOWN),
        ("stale", Code.STALE),
        ("bad_endpoint", Code.CONFLICT),
        ("bad_service", Code.CONFLICT),
        ("bad_direction", Code.UNKNOWN),
        ("bad_config", Code.CONFLICT),
        ("single_side", Code.MISSING),
        ("touch_windows", Code.CONFLICT),
        ("unverified", Code.UNKNOWN),
        ("false", Code.SATISFIED),
        ("zero", Code.SATISFIED),
        ("none", Code.MISSING),
        ("cross_task", Code.UNKNOWN),
    ],
)
def test_rules_and_pairing(tmp_path, mutation, expected):
    task, contract, bindings = prepared(tmp_path)
    target = next(
        b for b in bindings if b.category == ("R" if mutation in {"false", "zero", "none"} else "P")
    )
    rid = target.requirement_id
    if mutation == "wrong_version":
        target.version = "4.0"
    if mutation == "unknown_version":
        target.version = None
    if mutation == "no_identity":
        target.object_id = None
    if mutation == "no_window":
        target.observed = None
    if mutation == "stale":
        target.observed = target.observed.model_copy(
            update={
                "start": task.as_of - timedelta(hours=2),
                "end": task.as_of - timedelta(hours=1),
            }
        )
    if mutation == "bad_endpoint":
        target.fields["peer_endpoint"] = "192.0.2.99"
    if mutation == "bad_service":
        target.fields["service_id"] = "other"
    if mutation == "bad_direction":
        target.fields["direction"] = "?"
    if mutation == "bad_config":
        target.config_version = "cfg-2"
    if mutation == "single_side":
        bindings.remove(target)
    if mutation == "touch_windows":
        target.observed = target.observed.model_copy(
            update={
                "start": task.as_of - timedelta(seconds=120),
                "end": task.as_of - timedelta(seconds=60),
            }
        )
    if mutation == "unverified":
        target.source_verified = False
    if mutation == "false":
        target.fields["n4_enabled"] = False
    if mutation == "zero":
        target.fields["n4_enabled"] = 0
    if mutation == "none":
        target.fields["n4_enabled"] = None
    if mutation == "cross_task":
        target.task_id = "f" * 32
    states = full_evaluate(task, contract, bindings, "a")
    assert states[rid].code == expected
    if mutation not in {"zero"}:
        decisions = full_actions(task, contract, bindings, states, "a")
        if mutation in {"false", "none"}:
            assert decisions["action:UPF-A"].outcome != "suggest_allowed"
        elif expected != Code.SATISFIED:
            assert all(a.outcome != "suggest_allowed" for a in decisions.values())


def test_direction_rules_conflicts_priority_and_routing(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    req = next(r for r in contract.requirements if r.kind == "P")
    req.direction_map = {}
    states = full_evaluate(task, contract, bindings, "a")
    assert states[req.requirement_id].code == Code.UNKNOWN
    assert (
        next(
            g for g in route_gaps(contract, states) if g.requirement_id == req.requirement_id
        ).route
        == "human"
    )
    req.direction_map = {"tx": "forward", "rx": "reverse"}
    original = next(b for b in bindings if b.category == "R")
    alternative = original.model_copy(deep=True)
    alternative.evidence_id += "-opposing"
    alternative.source.source_id = "preferred"
    alternative.fields["n4_enabled"] = False
    bindings.append(alternative)
    assert (
        full_evaluate(task, contract, bindings, "a")[original.requirement_id].code == Code.CONFLICT
    )
    next(
        r for r in contract.requirements if r.requirement_id == original.requirement_id
    ).source_priority = ["preferred"]
    states = full_evaluate(task, contract, bindings, "b")
    assert states[original.requirement_id].code == Code.SATISFIED
    assert (
        full_actions(task, contract, bindings, states, "b")["action:UPF-A"].outcome == "re_evidence"
    )


def test_local_value_change_add_delete_time_and_missing_edge(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    local = LocalEvaluator(task, contract, bindings)
    for index, operation in enumerate(
        ["value", "unrelated", "add", "delete", "null", "time", "edge"]
    ):
        changed = deepcopy(bindings)
        if operation == "value":
            changed[1].fields["n4_enabled"] = False
        if operation == "unrelated":
            changed[1].fields["unused"] = 5
        if operation == "add":
            b = changed[1].model_copy(deep=True)
            b.evidence_id += "new"
            changed.append(b)
        if operation == "delete":
            changed.pop(1)
        if operation == "null":
            changed[1].fields["n4_enabled"] = None
        now = task.as_of + timedelta(hours=1) if operation == "time" else task.as_of
        if operation == "edge":
            local.index.graph.remove_edge(*next(iter(local.index.graph.edges)))
        batch = str(index)
        receipt = local.update(changed, batch, now)
        states = full_evaluate(task, contract, changed, batch, now)
        actions = full_actions(task, contract, changed, states, batch, now=now)
        assert semantic_results(local.states, local.actions) == semantic_results(states, actions)
        if operation == "edge":
            assert receipt["mode"] == "full_fallback"


def test_pending_budget_cycle_and_scope(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    local = LocalEvaluator(task, contract, bindings)
    changed = deepcopy(bindings)
    changed[1].fields["n4_enabled"] = False
    result = local.update(changed, "b", max_recomputations=0)
    assert result["pending_states"]
    assert all(a.outcome != "suggest_allowed" for a in local.actions.values())
    local.index.graph.add_edge("state:C:UPF-A", "state:R:UPF-A")
    result = local.update(changed, "c")
    assert result["reason"] == "dependency_cycle"
    assert all(a.outcome != "suggest_allowed" for a in local.actions.values())
    states = full_evaluate(task, contract, bindings, "a")
    action = action_templates(task, contract)[0]
    action.prerequisite_objects = ["outside"]
    assert (
        gate(action, task, contract, states, EvidenceSnapshot(bindings), "a").outcome == "blocked"
    )


def test_scope_split_and_capability_gap(tmp_path):
    task, contract, _ = prepared(tmp_path)
    plan = compile_query_plan(task, contract)
    assert len(plan.steps) == 2
    assert {s.object_id for s in plan.steps} == set(task.objects)
    task.observation_files = []
    assert compile_query_plan(task, contract).gaps
    task.permission.objects = ["UPF-A"]
    states = full_evaluate(task, contract, [], "a")
    assert states["A:SMF-B"].code == Code.DENIED
