"""Persist the 30 boundary scenarios in each Local mode and independent safety controls."""

import argparse
from copy import deepcopy
from pathlib import Path

from upf_agent.v8.dependency import complete_results
from upf_agent.v8.e4 import MODES, FullEvaluator, make_evaluator, write_json
from upf_agent.v8.experiment_manifest import source_manifest
from upf_agent.v8.phase1 import _boundaries, prepared


def run(project, output):
    if output.exists():
        raise ValueError("use a new experiment ID")
    output.mkdir(parents=True)
    source = source_manifest(project)
    boundaries, controls = {}, []
    for mode in list(MODES)[1:]:
        folder = output / mode
        folder.mkdir()
        boundaries[mode] = _boundaries(folder, MODES[mode])
        task, contract, bindings = prepared(folder / "controls")
        local = make_evaluator(mode, task, contract, bindings, None)
        assert local.actions["action:UPF-A"].outcome == "suggest_allowed"
        controls.append({"method": mode, "control": "legal-positive", "pass": True})
        changed = deepcopy(bindings)
        next(b for b in changed if b.category == "R").fields["n4_enabled"] = False
        receipt = local.update(changed, "field-change")
        denied = local.actions["action:UPF-A"].outcome == "re_evidence"
        controls.append(
            {
                "method": mode,
                "control": "value-sensitive-negative",
                "pass": denied,
                "receipt": receipt,
            }
        )
        # Restore, then interrupt a new change; same-input replay must settle every pending reader.
        local.update(bindings, "restored")
        receipt = local.update(changed, "interrupted", max_recomputations=0)
        controls.append(
            {
                "method": mode,
                "control": "budget-stop",
                "pass": all(a.outcome == "human_review" for a in local.actions.values()),
                "receipt": receipt,
            }
        )
        receipt = local.update(changed, "recovered")
        full = FullEvaluator(task, contract, changed, None, "recovered")
        controls.append(
            {
                "method": mode,
                "control": "same-input-recovery",
                "pass": complete_results(full.states, full.actions)
                == complete_results(local.states, local.actions),
                "receipt": receipt,
            }
        )
        # Initial observed reads remain mandatory even for the two audit ablation modes.
        local.index.graph.remove_edge(*next(iter(local.index.actual_edges)))
        local.index.expected_edges = frozenset(local.index.graph.edges)
        receipt = local.update(bindings, "broken-read-index")
        controls.append(
            {
                "method": mode,
                "control": "observed-edge-omission",
                "pass": receipt["reason"] == "incomplete_read_index",
                "receipt": receipt,
            }
        )
        local = make_evaluator(mode, task, contract, bindings, None)
        local.index.graph.add_edge("state:C:UPF-A", "state:C:UPF-A")
        receipt = local.update(bindings, "cycle")
        controls.append(
            {
                "method": mode,
                "control": "cycle-safe-stop",
                "pass": receipt["reason"] == "dependency_cycle"
                and all(a.outcome != "suggest_allowed" for a in local.actions.values()),
                "receipt": receipt,
            }
        )
    result = {
        "source": source,
        "synthetic": True,
        "reviewed": False,
        "common_rules": True,
        "expert_gold": False,
        "heldout_consumed": False,
        "boundaries": boundaries,
        "controls": controls,
        "model_calls": 0,
        "source_unchanged": source_manifest(project) == source,
    }
    result["pass"] = (
        result["source_unchanged"]
        and all(b["complete_equal"] == b["cases"] for b in boundaries.values())
        and all(c["pass"] for c in controls)
    )
    write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.project.resolve(), args.output.resolve())
    print({"pass": result["pass"], "boundaries": 90, "controls": len(result["controls"])})
    raise SystemExit(0 if result["pass"] else 1)
