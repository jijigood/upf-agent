"""Measure semantic cutoff on an explicit stable-output fanout control, outside V2 matrix."""

import argparse
import random
from pathlib import Path
from time import perf_counter

from upf_agent.v8.dependency import complete_results
from upf_agent.v8.e4 import MODES, make_evaluator, write_json
from upf_agent.v8.evaluation import SIZES, scaling_fixture
from upf_agent.v8.experiment_manifest import source_manifest
from upf_agent.v8.models import Requirement


def run(project, output):
    if output.exists():
        raise ValueError("choose a new experiment ID")
    output.mkdir(parents=True)
    source, rows = source_manifest(project), []
    write_json(
        output / "protocol.json",
        {
            "synthetic": True,
            "reviewed": False,
            "expert_gold": False,
            "model_calls": 0,
            "matrix": "S/M/L/XL x 3 seeds x 3 batches x 4 modes = 144 method batches",
            "intervention": (
                "R:0 wrong-version record is always excluded; change its scalar/revision"
            ),
            "expected": "R:0 not_applicable; fanout C missing; L1 recomputes C; L2/L3 reuse C",
            "timing": (
                "single worker, randomized mode order, shared machine; excludes comparison IO"
            ),
            "source": source,
        },
    )
    for size in SIZES:
        for seed in range(3):
            task, contract, bindings, actions, _ = scaling_fixture(output / "sources", size, seed)
            bindings[0].version = "4.0"
            for i, action in enumerate(actions):
                rid = f"C:stable:{i}"
                contract.requirements.append(
                    Requirement(
                        requirement_id=rid,
                        kind="C",
                        objects=[action.target],
                        upstream=["R:0"],
                    )
                )
                action.required_states.append(rid)
            write_json(
                output / f"fixture-{size}-{seed}.json",
                {
                    "task": task.model_dump(mode="json"),
                    "contract": contract.model_dump(mode="json"),
                    "bindings": [b.model_dump(mode="json") for b in bindings],
                    "actions": [a.model_dump(mode="json") for a in actions],
                },
            )
            methods = {m: make_evaluator(m, task, contract, bindings, actions) for m in MODES}
            order, rng = list(MODES), random.Random(seed)
            field = next(iter(bindings[0].fields))
            for batch in range(3):
                changed = list(bindings)
                changed[0] = bindings[0].model_copy(deep=True)
                changed[0].fields[field] = batch + 2
                changed[0].revision += 1
                rng.shuffle(order)
                results, current_rows = {}, []
                for mode in order:
                    evaluator = methods[mode]
                    started = perf_counter()
                    receipt = evaluator.update(changed, f"stable-{size}-{seed}-{batch}")
                    elapsed = perf_counter() - started
                    results[mode] = complete_results(evaluator.states, evaluator.actions)
                    expected_count = (
                        len(contract.requirements)
                        if mode == "L0-Full"
                        else len(actions) + 1
                        if mode == "L1-Local-no-cutoff"
                        else 1
                    )
                    current_rows.append(
                        {
                            "size": size,
                            "seed": seed,
                            "batch": batch,
                            "method": mode,
                            "seconds": elapsed,
                            "receipt": receipt,
                            "independent_control_pass": evaluator.states["R:0"].code.value
                            == "not_applicable"
                            and all(
                                evaluator.states[f"C:stable:{i}"].code.value == "missing"
                                for i in range(len(actions))
                            )
                            and receipt["recomputed_states"] == expected_count,
                        }
                    )
                for row in current_rows:
                    row["equal"] = results[row["method"]] == results["L0-Full"]
                rows.extend(current_rows)
                write_json(output / "rows.json", rows)
                if not all(r["equal"] and r["independent_control_pass"] for r in current_rows):
                    write_json(output / "failure.json", results)
                    raise RuntimeError("stable fanout control failed")
                bindings = changed
    result = {
        "pass": len(rows) == 144 and source_manifest(project) == source,
        "method_batches": len(rows),
        "source": source,
        "rows": rows,
    }
    write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.project.resolve(), args.output.resolve())
    print({"pass": result["pass"], "method_batches": result["method_batches"]})
    raise SystemExit(0 if result["pass"] else 1)
