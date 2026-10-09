import json
from copy import deepcopy
from datetime import timedelta

import networkx as nx
import pytest
from test_v8_rules import prepared

from upf_agent.v8.actions import action_templates, gate, impact_scope
from upf_agent.v8.compiler import compile_gap_plan, slot
from upf_agent.v8.dependency import LocalEvaluator, semantic_results
from upf_agent.v8.engine import OrchestrationEngine
from upf_agent.v8.evaluator import EvidenceSnapshot, full_evaluate, route_gaps
from upf_agent.v8.graph import MemoryGraphStore
from upf_agent.v8.models import Code, FieldPredicate, PathPosition


@pytest.mark.parametrize(
    "change", ["purpose", "interface", "mapping", "validity", "real_synthetic"]
)
def test_applicability_cannot_be_bypassed(tmp_path, change):
    task, contract, bindings = prepared(tmp_path)
    target = next(
        b
        for b in bindings
        if b.category == ("K" if change == "purpose" else "P" if change == "mapping" else "R")
    )
    if change == "purpose":
        target.purposes = ["log_observation"]
    if change == "interface":
        target.fields["interface"] = "N3"
    if change == "mapping":
        for b in bindings:
            if b.category == "P":
                b.fields["mapping_version"] = "map-obsolete"
    if change == "validity":
        target.valid = target.observed.model_copy(
            update={"start": task.as_of - timedelta(seconds=5), "end": task.as_of}
        )
    if change == "real_synthetic":
        # Evaluate the record restriction independently of path admission.
        task.path.synthetic = False
        task.synthetic = False
    state = full_evaluate(task, contract, bindings, "b")[target.requirement_id]
    assert state.code != Code.SATISFIED


def test_previous_batch_frozen_even_if_caller_mutates(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    local = LocalEvaluator(task, contract, bindings)
    bindings[1].fields["n4_enabled"] = False
    receipt = local.update(bindings, "b")
    assert receipt["recomputed_states"] > 0
    assert local.actions["action:UPF-A"].outcome == "re_evidence"


def test_unchanged_upstream_clears_only_stable_readers(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    local = LocalEvaluator(task, contract, bindings)
    unused = deepcopy(bindings)
    unused[1].fields["unused"] = "irrelevant"
    receipt = local.update(unused, "b")
    assert receipt["recomputed_states"] == 0
    assert not receipt["pending_states"]
    # Invalidating a logical field without changing its normalized value safely clears C.
    changed = deepcopy(unused)
    changed[1].revision += 1
    receipt = local.update(changed, "c")
    assert not receipt["pending_states"]


def test_declared_actual_read_omission_falls_back(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    local = LocalEvaluator(task, contract, bindings)
    edge = (slot("UPF-A", "R", "n4_enabled"), "state:R:UPF-A")
    local.index.graph.remove_edge(*edge)
    # Simulate a corrupt producer that falsely certifies its reduced edge set.
    local.index.expected_edges = frozenset(local.index.graph.edges)
    changed = deepcopy(bindings)
    changed[1].revision += 1
    receipt = local.update(changed, "b")
    assert receipt["mode"] == "full_fallback"
    # Independently captured initial reads catch the omission before scheduling.
    assert receipt["reason"] == "incomplete_read_index"


def test_directed_missing_side_plan_does_not_expand_scope(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    bindings = [b for b in bindings if not (b.category == "P" and b.object_id == "SMF-B")]
    gaps = route_gaps(contract, full_evaluate(task, contract, bindings, "b"))
    plan = compile_gap_plan(task, contract, gaps, set())
    pair_steps = [s for s in plan.steps if any(r.startswith("P:") for r in s.requirement_ids)]
    assert [s.object_id for s in pair_steps] == ["SMF-B"]


def test_disjoint_impact_intervals_and_cutoff(tmp_path):
    task, contract, _ = prepared(tmp_path)
    task.path.positions = [PathPosition(object_id=str(i), domain="d") for i in range(5)]
    action = action_templates(task, contract)[0]
    action.target = "0"
    graph = nx.DiGraph()
    graph.add_edge("0", "1", relation="CARRIES")
    graph.add_edge("1", "4", relation="DEPENDS_ON")
    graph.add_edge("1", "2", relation="ADMIN_BOUNDARY")
    objects, intervals = impact_scope(action, task, graph)
    assert objects == ["0", "1", "4"]
    assert intervals == [[0, 2], [4, 5]]


def test_committed_receipts_survive_node_crash(tmp_path, monkeypatch):
    monkeypatch.setattr("upf_agent.v8.engine.load_manual_graph", lambda _: MemoryGraphStore())
    task, _, _ = prepared(tmp_path)
    engine = OrchestrationEngine(tmp_path)
    original = engine._collect

    def crash(state):
        original(state)
        raise RuntimeError("simulated crash after evidence transaction")

    engine._collect = crash
    with pytest.raises(RuntimeError):
        engine.run(task)
    before = engine.ledger.inspect(task.run_id)["receipts"]
    engine.close()
    recovered = OrchestrationEngine(tmp_path)
    try:
        result = recovered.resume(task.run_id)
        assert result["status"] == "completed"
        after = recovered.ledger.inspect(task.run_id)["receipts"]
        assert all(r in after for r in before)
        assert len(after) == len(before) + 1  # the unstarted graph query only
    finally:
        recovered.close()


def test_source_changed_after_receipt_is_not_trusted(tmp_path, monkeypatch):
    monkeypatch.setattr("upf_agent.v8.engine.load_manual_graph", lambda _: MemoryGraphStore())
    task, _, _ = prepared(tmp_path)
    engine = OrchestrationEngine(tmp_path)
    original = engine._collect

    def corrupt(state):
        result = original(state)
        path = tmp_path / task.observation_files[0]
        body = json.loads(path.read_text())
        body["changed_after_receipt"] = True
        path.write_text(json.dumps(body))
        return result

    engine._collect = corrupt
    try:
        result = engine.run(task)
        assert all(a["outcome"] != "suggest_allowed" for a in result["actions"])
    finally:
        engine.close()


def test_gate_rejects_current_binding_with_old_satisfied_basis(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    states = full_evaluate(task, contract, bindings, "same-batch")
    changed = deepcopy(bindings)
    next(b for b in changed if b.category == "K").version = "4.0"
    action = action_templates(task, contract)[0]
    decision = gate(action, task, contract, states, EvidenceSnapshot(changed), "same-batch")
    assert decision.outcome == "human_review"
    assert any("basis_mismatch" in r for r in decision.reasons)
    task.permission.revision = "2"
    decision = gate(action, task, contract, states, EvidenceSnapshot(bindings), "same-batch")
    assert decision.outcome == "human_review"
    assert any("context_basis" in r for r in decision.reasons)


def test_unrelated_field_has_no_semantic_effect_and_direct_predicate_is_tracked(tmp_path):
    from upf_agent.v8.actions import full_actions

    task, contract, bindings = prepared(tmp_path)
    local = LocalEvaluator(task, contract, bindings)
    changed = deepcopy(bindings)
    changed[1].fields["unused"] = "ignored"
    receipt = local.update(changed, "b")
    states = full_evaluate(task, contract, changed, "b")
    assert receipt["recomputed_states"] == 0
    assert semantic_results(local.states, local.actions) == semantic_results(
        states, full_actions(task, contract, changed, states, "b")
    )
    action = action_templates(task, contract)[0]
    action.predicates = [FieldPredicate(object_id="UPF-A", field="mtu", operator="ge", value=1280)]
    local = LocalEvaluator(task, contract, bindings, actions=[action])
    changed = deepcopy(bindings)
    changed[1].fields["mtu"] = 900
    receipt = local.update(changed, "c")
    assert receipt["recomputed_states"] == 0
    assert local.actions[action.action_id].outcome == "re_evidence"


def test_active_task_deadline_includes_processing(tmp_path, monkeypatch):
    from time import sleep

    monkeypatch.setattr("upf_agent.v8.engine.load_manual_graph", lambda _: MemoryGraphStore())
    task, _, _ = prepared(tmp_path)
    task.budget.timeout_seconds = 0.01
    engine = OrchestrationEngine(tmp_path)
    original = engine._collect

    def delayed(state):
        sleep(0.02)
        return original(state)

    engine._collect = delayed
    try:
        value = engine.run(task)
        assert value["stop_reason"] == "task_timeout"
        assert all(a["outcome"] != "suggest_allowed" for a in value["actions"])
    finally:
        engine.close()


def test_concurrent_run_is_excluded_and_lock_released(tmp_path):
    from upf_agent.v8.store import Ledger

    task, _, _ = prepared(tmp_path)
    first, second = Ledger(tmp_path), Ledger(tmp_path)
    try:
        with first.execution_lock(task.run_id):
            with pytest.raises(ValueError, match="already active"):
                with second.execution_lock(task.run_id):
                    pytest.fail("the same run cannot enter twice")
            with second.execution_lock("f" * 32):
                pass
        with second.execution_lock(task.run_id):
            pass
    finally:
        first.close()
        second.close()


def test_object_identifiers_and_field_slots_cannot_collide(tmp_path):
    from upf_agent.v8.compiler import compile_contract

    task, _, _ = prepared(tmp_path)
    task.objects = ["A:B", "A", "B:C"]
    task.path.positions = [PathPosition(object_id=o, domain="d") for o in task.objects]
    contract = compile_contract(task)
    identifiers = [r.requirement_id for r in contract.requirements]
    assert len(set(identifiers)) == len(identifiers)
    assert "P:A%3AB:A" in identifiers
    assert slot("A:B", "R", "C") != slot("A", "B:R", "C")


def test_actual_state_read_omission_falls_back(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    local = LocalEvaluator(task, contract, bindings)
    edge = ("state:R:UPF-A", "state:C:UPF-A")
    local.index.graph.remove_edge(*edge)
    local.index.expected_edges = frozenset(local.index.graph.edges)
    changed = deepcopy(bindings)
    changed[0].revision += 1
    changed[1].revision += 1
    receipt = local.update(changed, "b")
    assert receipt["mode"] == "full_fallback"
    assert receipt["reason"] == "incomplete_read_index"


def test_different_windows_need_declared_normalization(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    first = next(b for b in bindings if b.category == "R")
    second = first.model_copy(deep=True)
    second.evidence_id += "-different-window"
    second.observed.start += timedelta(seconds=5)
    bindings.append(second)
    states = full_evaluate(task, contract, bindings, "b")
    assert states[first.requirement_id].code == Code.UNKNOWN
    assert "window_normalization_unspecified" in states[first.requirement_id].reasons
    assert len(EvidenceSnapshot(bindings).records(first.object_id, "R", first.requirement_id)) == 2


def test_unresolved_cycle_stops_with_pending_states(tmp_path, monkeypatch):
    task, contract, bindings = prepared(tmp_path)
    local = LocalEvaluator(task, contract, bindings)
    local.index.graph.add_edge("state:C:UPF-A", "state:R:UPF-A")
    evaluate = local._full
    calls = []

    def oscillate(*args):
        result = evaluate(*args)
        calls.append(1)
        local.states["C:UPF-A"].basis["fault_injection"] = len(calls)
        return result

    monkeypatch.setattr(local, "_full", oscillate)
    receipt = local.update(bindings, "b")
    assert receipt["cycle_iterations"] == 3
    assert not receipt["cycle_converged"]
    assert all(s.pending for s in local.states.values())
    assert all(a.outcome == "human_review" for a in local.actions.values())


def test_missing_timezone_observation_cannot_release(tmp_path, monkeypatch):
    monkeypatch.setattr("upf_agent.v8.engine.load_manual_graph", lambda _: MemoryGraphStore())
    task, _, _ = prepared(tmp_path)
    path = tmp_path / task.observation_files[0]
    body = json.loads(path.read_text())
    for row in body["observations"]:
        for edge in ("start", "end"):
            row["observed"][edge] = row["observed"][edge].removesuffix("+00:00")
    path.write_text(json.dumps(body))
    engine = OrchestrationEngine(tmp_path)
    try:
        result = engine.run(task)
        assert result["status"] == "needs_evidence"
        assert all(a["outcome"] != "suggest_allowed" for a in result["actions"])
    finally:
        engine.close()


def test_actual_neo4j_unavailable_produces_terminal_review(tmp_path, monkeypatch):
    import socket

    monkeypatch.setattr("upf_agent.v8.engine.load_manual_graph", lambda _: MemoryGraphStore())
    task, _, _ = prepared(tmp_path)
    task.graph_backend = "neo4j"
    config = tmp_path / "config/v8-neo4j.json"
    config.parent.mkdir(parents=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        # A bound, non-listening socket guarantees connection refusal during this test.
        config.write_text(
            json.dumps(
                {"uri": f"bolt://127.0.0.1:{port}", "username": "test", "password": "test-fixture"}
            )
        )
        engine = OrchestrationEngine(tmp_path)
        try:
            result = engine.run(task)
            assert result["stop_reason"].startswith("graph_unavailable:")
            assert all(a["outcome"] == "human_review" for a in result["actions"])
        finally:
            engine.close()
