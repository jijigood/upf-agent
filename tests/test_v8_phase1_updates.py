"""Regression cases from the independent October 8 feasibility audit."""

from copy import deepcopy
from datetime import timedelta

import pytest
from test_v8_rules import prepared

from upf_agent.v8.actions import action_templates, full_actions, gate
from upf_agent.v8.compiler import slot
from upf_agent.v8.dependency import DependencyIndex, LocalEvaluator, complete_results
from upf_agent.v8.evaluator import EvidenceSnapshot, full_evaluate


def expiring_conflict(task, bindings):
    current = next(b for b in bindings if b.category == "R" and b.object_id == "UPF-A")
    conflict = current.model_copy(deep=True)
    conflict.evidence_id += ":expiring-conflict"
    conflict.fields["n4_enabled"] = False
    conflict.valid = current.observed.model_copy(update={"end": task.as_of + timedelta(seconds=5)})
    return [*bindings, conflict]


def test_noop_update_certifies_every_action_in_current_batch(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    local = LocalEvaluator(task, contract, bindings, "before")
    receipt = local.update(deepcopy(bindings), "after")
    assert receipt["recomputed_states"] == 0
    assert receipt["recomputed_actions"] == 0
    assert all(s.batch_id == "after" for s in local.states.values())
    assert all(a.batch_id == "after" for a in local.actions.values())
    assert all(a.computed_batch_id == "before" for a in local.actions.values())
    states = full_evaluate(task, contract, bindings, "after")
    actions = full_actions(task, contract, bindings, states, "after")
    assert complete_results(local.states, local.actions) == complete_results(states, actions)


def test_partial_update_certifies_unaffected_actions(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    local = LocalEvaluator(task, contract, bindings, "before")
    changed = deepcopy(bindings)
    next(b for b in changed if b.category == "R" and b.object_id == "UPF-A").fields[
        "n4_enabled"
    ] = False
    local.update(changed, "after")
    assert local.actions["action:UPF-A"].outcome == "re_evidence"
    assert local.actions["action:SMF-B"].outcome == "suggest_allowed"
    assert all(a.batch_id == "after" for a in local.actions.values())


def test_expired_conflict_uses_same_time_in_state_and_gate(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    augmented = expiring_conflict(task, bindings)
    local = LocalEvaluator(task, contract, augmented, "before")
    later = task.as_of + timedelta(seconds=10)
    local.update(augmented, "after", now=later)
    assert all(s.code.value == "satisfied" for s in local.states.values())
    assert all(a.outcome == "suggest_allowed" for a in local.actions.values())


def test_expired_conflict_full_fallback_also_uses_current_time(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    augmented = expiring_conflict(task, bindings)
    local = LocalEvaluator(task, contract, augmented, "before")
    local.index.graph.remove_edge(*next(iter(local.index.graph.edges)))
    receipt = local.update(augmented, "after", now=task.as_of + timedelta(seconds=10))
    assert receipt["mode"] == "full_fallback"
    assert all(a.outcome == "suggest_allowed" for a in local.actions.values())


def test_initial_missing_read_group_cannot_self_certify(tmp_path, monkeypatch):
    task, contract, bindings = prepared(tmp_path)
    runtime = next(b for b in bindings if b.category == "R" and b.object_id == "UPF-A")
    original = DependencyIndex.__init__

    def faulty(index, *args, **kwargs):
        original(index, *args, **kwargs)
        for field in [f"__record__:{runtime.requirement_id}", *runtime.fields]:
            source = slot(runtime.object_id, runtime.category, field)
            index.graph.remove_edges_from(list(index.graph.out_edges(source)))
        index.expected_edges = frozenset(index.graph.edges)

    monkeypatch.setattr(DependencyIndex, "__init__", faulty)
    local = LocalEvaluator(task, contract, bindings, "before")
    changed = deepcopy(bindings)
    current = next(b for b in changed if b.evidence_id == runtime.evidence_id)
    current.fields["n4_enabled"] = False
    current.revision += 1
    receipt = local.update(changed, "after")
    assert receipt["mode"] == "full_fallback"
    assert local.actions["action:UPF-A"].outcome == "re_evidence"
    assert not local.index.audit()


def test_naive_evaluation_time_is_rejected(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    local = LocalEvaluator(task, contract, bindings)
    with pytest.raises(ValueError, match="timezone"):
        local.update(bindings, "after", now=task.as_of.replace(tzinfo=None))


def test_current_time_full_actions_contract(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    augmented = expiring_conflict(task, bindings)
    later = task.as_of + timedelta(seconds=10)
    states = full_evaluate(task, contract, augmented, "after", now=later)
    decisions = full_actions(task, contract, augmented, states, "after", now=later)
    assert all(a.outcome == "suggest_allowed" for a in decisions.values())


def test_complete_comparison_detects_stale_certification(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    states = full_evaluate(task, contract, bindings, "current")
    actions = full_actions(task, contract, bindings, states, "current")
    stale = deepcopy(actions)
    next(iter(stale.values())).batch_id = "old"
    assert complete_results(states, stale) != complete_results(states, actions)


def test_gate_rejects_states_certified_for_a_different_time(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    states = full_evaluate(task, contract, bindings, "same-batch")
    decisions = full_actions(
        task, contract, bindings, states, "same-batch", now=task.as_of + timedelta(seconds=1)
    )
    assert all(a.outcome == "human_review" for a in decisions.values())


def test_context_rebuild_certifies_initial_reads(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    local = LocalEvaluator(task, contract, bindings, "before")
    task.permission.revision = "new-permission-context"
    receipt = local.update(bindings, "after")
    assert receipt["reason"] == "context_changed"
    assert receipt["read_audit"]["trusted"]
    assert all(a.batch_id == "after" for a in local.actions.values())
    assert all(a.context_fingerprint == local.context.fingerprint for a in local.actions.values())


def test_eligibility_cache_cannot_cross_evaluation_contexts(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    states = full_evaluate(task, contract, bindings, "before")
    snap = EvidenceSnapshot(bindings)
    action = action_templates(task, contract)[0]
    assert gate(action, task, contract, states, snap, "before").outcome == "suggest_allowed"
    later = task.as_of + timedelta(seconds=1)
    new_states = full_evaluate(task, contract, bindings, "after", now=later)
    with pytest.raises(ValueError, match="cannot be reused"):
        gate(action, task, contract, new_states, snap, "after", now=later)
