"""E4 counterexamples distinguish propagation, auditing, and recovery from Full traversal."""

from copy import deepcopy

import pytest

from upf_agent.v8 import dependency
from upf_agent.v8.compiler import slot
from upf_agent.v8.dependency import LocalEvaluator, complete_results
from upf_agent.v8.e4 import MODES, FullEvaluator, make_evaluator, matrix_groups, run_matrix
from upf_agent.v8.evaluation import scaling_fixture
from upf_agent.v8.models import Requirement
from upf_agent.v8.phase1 import _boundaries, prepared


@pytest.mark.parametrize("mode", list(MODES)[1:])
def test_thirty_certification_boundaries(tmp_path, mode):
    assert _boundaries(tmp_path, MODES[mode]) == {"cases": 30, "complete_equal": 30}


def test_full_reference_does_not_construct_dependency_index(tmp_path, monkeypatch):
    task, contract, bindings, actions, _ = scaling_fixture(tmp_path, "S", 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Full must not depend on tested index")

    monkeypatch.setattr(dependency.DependencyIndex, "__init__", forbidden)
    reference = FullEvaluator(task, contract, bindings, actions)
    assert reference.update(bindings, "next")["recomputed_states"] == 30


def test_cutoff_actually_skips_transitive_reader(tmp_path):
    task, contract, bindings, _, _ = scaling_fixture(tmp_path, "S", 0)
    contract.requirements.append(
        Requirement(
            requirement_id="C:child",
            kind="C",
            objects=["object-0"],
            upstream=["R:0"],
        )
    )
    bindings[0].version = "4.0"
    no_cutoff = LocalEvaluator(
        task, contract, bindings, actions=[], cutoff=False, audit_reads=False
    )
    cutoff = LocalEvaluator(task, contract, bindings, actions=[], cutoff=True, audit_reads=False)
    changed = deepcopy(bindings)
    # This record is excluded for wrong version. Its changed scalar input cannot change the
    # rejection; only L1 must propagate to C:child. This is also an independent negative oracle.
    changed[0].fields["metric_0"] = 2
    left = no_cutoff.update(changed, "next")
    right = cutoff.update(changed, "next")
    assert left["recomputed_states"] == 2
    assert right["recomputed_states"] == 1
    assert right["reused_states"] == 1
    assert cutoff.states["R:0"].code.value == "not_applicable"
    assert cutoff.states["C:child"].code.value == "missing"
    reference = FullEvaluator(task, contract, changed, [], "next")
    assert (
        complete_results(no_cutoff.states, no_cutoff.actions)
        == complete_results(cutoff.states, cutoff.actions)
        == complete_results(reference.states, reference.actions)
    )


@pytest.mark.parametrize("mode", list(MODES)[1:])
def test_budget_interruption_then_identical_input_recovers(tmp_path, mode):
    task, contract, bindings = prepared(tmp_path / "fixture")
    local = make_evaluator(mode, task, contract, bindings, None)
    changed = deepcopy(bindings)
    next(b for b in changed if b.category == "R").fields["n4_enabled"] = False
    interrupted = local.update(changed, "interrupted", max_recomputations=0)
    assert interrupted["reason"] == "recompute_budget_exhausted"
    assert interrupted["pending_states"]
    assert all(a.outcome == "human_review" for a in local.actions.values())
    recovered = local.update(changed, "recovered")
    assert recovered["reason"] == "pending_recovery"
    assert not recovered["pending_states"]
    reference = FullEvaluator(task, contract, changed, None, "recovered")
    assert complete_results(reference.states, reference.actions) == complete_results(
        local.states, local.actions
    )


def test_action_only_interruption_can_recover(tmp_path):
    task, contract, bindings, actions, _ = scaling_fixture(tmp_path, "S", 0)
    actions[0].predicates[0].field = "extra"
    task.permission.readable_fields.append("extra")
    bindings[0].fields["extra"] = 1
    local = LocalEvaluator(task, contract, bindings, actions=actions)
    changed = deepcopy(bindings)
    changed[0].fields["extra"] = 0
    interrupted = local.update(changed, "stop", max_recomputations=0)
    assert interrupted["reason"] == "recompute_budget_exhausted"
    assert not interrupted["pending_states"]
    repeated = local.update(changed, "still-stop", max_recomputations=0)
    assert repeated["reason"] == "recompute_budget_exhausted"
    assert all(a.outcome == "human_review" for a in local.actions.values())
    assert local.update(changed, "resume")["reason"] == "pending_recovery"
    full = FullEvaluator(task, contract, changed, actions, "resume")
    assert complete_results(full.states, full.actions) == complete_results(
        local.states, local.actions
    )


@pytest.mark.parametrize("mode", list(MODES)[1:])
def test_index_corruption_remains_guarded_in_audit_ablation(tmp_path, mode):
    task, contract, bindings = prepared(tmp_path / "fixture")
    local = make_evaluator(mode, task, contract, bindings, None)
    local.index.graph.remove_edge(*next(iter(local.index.graph.edges)))
    changed = deepcopy(bindings)
    next(b for b in changed if b.category == "R").fields["n4_enabled"] = False
    receipt = local.update(changed, "corrupt")
    assert receipt["reason"] == "incomplete_read_index"
    assert local.actions["action:UPF-A"].outcome == "re_evidence"


def test_audited_mode_detects_new_branch_read(tmp_path, monkeypatch):
    task, contract, bindings = prepared(tmp_path / "fixture")
    local = make_evaluator("L3-Local-audited-cutoff", task, contract, bindings, None)
    evaluate = dependency.evaluate_requirement

    def branch(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        args[3].reads.add(slot("UPF-A", "R", "new_branch_input"))
        return result

    monkeypatch.setattr(dependency, "evaluate_requirement", branch)
    changed = deepcopy(bindings)
    next(b for b in changed if b.category == "R").fields["n4_enabled"] = False
    receipt = local.update(changed, "new-branch")
    assert receipt["reason"] == "undeclared_actual_read"
    assert receipt["cost"]["observed_read_audits"] == 1
    assert local.actions["action:UPF-A"].outcome == "re_evidence"


@pytest.mark.parametrize("mode", list(MODES)[1:])
def test_cycle_never_leaves_allowed_suggestion(tmp_path, mode):
    task, contract, bindings = prepared(tmp_path / "fixture")
    local = make_evaluator(mode, task, contract, bindings, None)
    local.index.graph.add_edge("state:C:UPF-A", "state:C:UPF-A")
    receipt = local.update(bindings, "cycle")
    assert receipt["reason"] == "dependency_cycle"
    assert all(a.outcome != "suggest_allowed" for a in local.actions.values())


def test_full_and_small_protocol_counts():
    assert sum(s[2] * s[3] * 4 for s in matrix_groups("small")) == 720
    assert sum(s[2] * s[3] * 4 for s in matrix_groups("full")) == 64000


def test_local_validates_context_once_per_batch(tmp_path, monkeypatch):
    from upf_agent.v8 import actions as action_module

    task, contract, bindings = prepared(tmp_path / "fixture")
    local = LocalEvaluator(task, contract, bindings)
    changed = deepcopy(bindings)
    next(b for b in changed if b.category == "R").revision += 1
    calls = []
    original = dependency.resolve_context

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(dependency, "resolve_context", counted)
    monkeypatch.setattr(action_module, "resolve_context", counted)
    local.update(changed, "one-context")
    assert len(calls) == 1
    reference = FullEvaluator(task, contract, changed, None, "one-context")
    assert complete_results(local.states, local.actions) == complete_results(
        reference.states, reference.actions
    )


def test_expansion_without_correctness_evidence_is_rejected(tmp_path):
    from pathlib import Path

    with pytest.raises(ValueError, match="correctness summary"):
        run_matrix(Path(dependency.__file__).resolve().parents[3], tmp_path / "expanded", "full")
    assert not (tmp_path / "expanded").exists()
