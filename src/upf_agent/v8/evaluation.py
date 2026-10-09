"""Frozen synthetic correctness suites and measured full/local scalability experiments."""

from __future__ import annotations

import csv
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

from upf_agent.v8.actions import full_actions
from upf_agent.v8.dependency import LocalEvaluator, semantic_results
from upf_agent.v8.evaluator import full_evaluate
from upf_agent.v8.models import (
    ActionTemplate,
    BusinessPath,
    EvidenceBinding,
    EvidenceContract,
    FieldPredicate,
    OrchestrationTask,
    PathPosition,
    PermissionScope,
    Requirement,
    SourcePointer,
    digest,
)

SIZES = {"S": (100, 30, 10), "M": (500, 150, 30), "L": (2000, 600, 100), "XL": (10000, 3000, 500)}
UPDATES = [1, 5, 10, 50]
SEEDS = list(range(20))
BATCHES = 50


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def percentiles(values):
    values = sorted(values)
    return (
        {"p50": values[int((len(values) - 1) * 0.50)], "p95": values[int((len(values) - 1) * 0.95)]}
        if values
        else {}
    )


def scaling_fixture(root: Path, size: str, seed: int):
    field_count, state_count, action_count = SIZES[size]
    objects = [f"object-{i}" for i in range(action_count)]
    field_names = [f"metric_{i}" for i in range(field_count)]
    now = datetime(2026, 10, 7, tzinfo=UTC)
    identity = {
        "size": size,
        "seed": seed,
        "updates": UPDATES,
        "batches": BATCHES,
        "generator": "v8-scaling-1",
        "synthetic": True,
    }
    source_path = root / f"data/fixtures/v8-scaling/{size}-{seed}.json"
    write_json(source_path, identity)
    source = SourcePointer(
        source_id=f"scaling-{size}-{seed}",
        sha256=digest(identity),
        locator="/deterministic-generator",
        relative_path=str(source_path.relative_to(root)),
    )
    # The archive points to the generator and seed, not a real device observation.
    import hashlib

    source.sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    task = OrchestrationTask(
        run_id=digest(identity)[:32],
        question="Synthetic scalar-rule scaling experiment",
        template="log_observation",
        objects=objects,
        synthetic=True,
        product="UPF",
        version="5.0",
        as_of=now,
        path=BusinessPath(
            positions=[PathPosition(object_id=o, domain=str(i)) for i, o in enumerate(objects)],
            synthetic=True,
        ),
        permission=PermissionScope(objects=objects, readable_fields=field_names),
    )
    reqs, bindings, field_map = [], [], {}
    runtime_states = state_count - action_count
    for i in range(runtime_states):
        obj = objects[i % action_count]
        fields = field_names[i::runtime_states]
        req = Requirement(requirement_id=f"R:{i}", kind="R", objects=[obj], fields=fields)
        reqs.append(req)
        binding = EvidenceBinding(
            evidence_id=f"e:{i}",
            native_id=f"e:{i}",
            task_id=task.run_id,
            contract_id="scaling",
            requirement_id=req.requirement_id,
            object_id=obj,
            category="R",
            fields=dict.fromkeys(fields, 1),
            source=source,
            version="5.0",
            synthetic=True,
            source_verified=True,
            receipt_id="synthetic-generator",
            observed={
                "start": now - timedelta(seconds=60),
                "end": now,
                "timezone_source": "UTC fixture",
            },
        )
        bindings.append(binding)
        for f in fields:
            field_map[f] = i
    reqs.extend(
        Requirement(
            requirement_id=f"A:{o}",
            kind="A",
            objects=[o],
            version_required=False,
            freshness_seconds=None,
        )
        for o in objects
    )
    contract = EvidenceContract(
        contract_id="scaling",
        task_id=task.run_id,
        template="scalar_scaling",
        task_fingerprint=digest(task),
        requirements=reqs,
    )
    actions = [
        ActionTemplate(
            action_id=f"action:{i}",
            name="CHECK_INTERFACE_STATUS",
            target=o,
            parameters={"mode": "read_only"},
            required_states=[f"R:{i}", f"A:{o}"],
            allowed_relations=[],
            predicates=[FieldPredicate(object_id=o, field=field_names[i], operator="ge", value=1)],
        )
        for i, o in enumerate(objects)
    ]
    return task, contract, bindings, actions, field_map


def local_experiment(root: Path) -> dict:
    output = root / "experiments/v8/local"
    output.mkdir(parents=True, exist_ok=True)
    matrix = {
        "sizes": SIZES,
        "updates": UPDATES,
        "seeds": SEEDS,
        "batches": BATCHES,
        "comparison": "independent full evaluator vs tracked dependency evaluator",
        "synthetic": True,
        "model_calls": 0,
        "tool_calls": 0,
    }
    write_json(output / "matrix.json", matrix)
    totals = {"batches": 0, "consistent": 0, "mismatches": [], "groups": {}, "synthetic": True}
    with (output / "batches.csv").open("w", newline="") as stream:
        columns = [
            "size",
            "updates",
            "seed",
            "batch",
            "full_seconds",
            "local_seconds",
            "full_states",
            "local_states",
            "local_actions",
            "consistent",
            "fallback",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for size in SIZES:
            for updates in UPDATES:
                full_times, local_times, cold = [], [], []
                full_counts, local_counts = [], []
                for seed in SEEDS:
                    task, contract, bindings, actions, field_map = scaling_fixture(root, size, seed)
                    started = perf_counter()
                    local = LocalEvaluator(task, contract, bindings, actions=actions)
                    cold.append(perf_counter() - started)
                    randomizer = random.Random(seed)
                    names = sorted(field_map)
                    receipts = []
                    for batch in range(BATCHES):
                        changed = list(bindings)
                        selected = randomizer.sample(names, updates)
                        grouped = {}
                        for field in selected:
                            i = field_map[field]
                            if i not in grouped:
                                grouped[i] = bindings[i].model_copy(deep=True)
                            grouped[i].fields[field] = randomizer.randrange(0, 4)
                        for i, b in grouped.items():
                            b.revision += 1
                            changed[i] = b
                        batch_id = f"{size}-{updates}-{seed}-{batch}"
                        started = perf_counter()
                        reference = full_evaluate(task, contract, changed, batch_id)
                        reference_actions = full_actions(
                            task, contract, changed, reference, batch_id, actions
                        )
                        full_duration = perf_counter() - started
                        started = perf_counter()
                        receipt = local.update(changed, batch_id)
                        local_duration = perf_counter() - started
                        consistent = semantic_results(
                            reference, reference_actions
                        ) == semantic_results(local.states, local.actions)
                        totals["batches"] += 1
                        totals["consistent"] += int(consistent)
                        if not consistent:
                            totals["mismatches"].append(batch_id)
                            write_json(
                                output / f"failure-{batch_id}.json",
                                {
                                    "full": semantic_results(reference, reference_actions),
                                    "local": semantic_results(local.states, local.actions),
                                },
                            )
                        writer.writerow(
                            dict(
                                zip(
                                    columns,
                                    [
                                        size,
                                        updates,
                                        seed,
                                        batch,
                                        full_duration,
                                        local_duration,
                                        len(reference),
                                        receipt["recomputed_states"],
                                        receipt.get("recomputed_actions", 0),
                                        consistent,
                                        receipt["mode"],
                                    ],
                                    strict=True,
                                )
                            )
                        )
                        full_times.append(full_duration)
                        local_times.append(local_duration)
                        full_counts.append(len(reference))
                        local_counts.append(receipt["recomputed_states"])
                        receipts.append(
                            {
                                **receipt,
                                "selected": selected,
                                "values": {f: changed[field_map[f]].fields[f] for f in selected},
                            }
                        )
                        bindings = changed
                    write_json(output / f"updates-{size}-{updates}-{seed}.json", receipts)
                    stream.flush()
                group = {
                    "full_seconds": percentiles(full_times),
                    "local_seconds": percentiles(local_times),
                    "cold_seconds": percentiles(cold),
                    "batches": len(full_times),
                    "mean_full_states": sum(full_counts) / len(full_counts),
                    "mean_local_states": sum(local_counts) / len(local_counts),
                }
                totals["groups"][f"{size}-{updates}"] = group
                write_json(output / "summary.json", totals)
                print(
                    json.dumps(
                        {
                            "local_group": f"{size}-{updates}",
                            "completed": totals["batches"],
                            "consistent": totals["consistent"],
                            **group,
                        }
                    ),
                    flush=True,
                )
    totals["pass"] = totals["consistent"] == 16000 and not totals["mismatches"]
    write_json(output / "summary.json", totals)
    return totals


def evaluate_suite(root: Path, suite: str, batch: str | None = None):
    if suite == "local":
        return local_experiment(root)
    if suite in {"dev", "agent", "heldout"}:
        from upf_agent.v8.scenarios import scenario_experiment

        return scenario_experiment(root, suite, batch)
    if suite == "graph":
        from upf_agent.v8.graph_experiment import graph_experiment

        return graph_experiment(root)
    if suite == "all":
        return {
            name: evaluate_suite(root, name, batch)
            for name in ["dev", "local", "graph", "agent", "heldout"]
        }
    raise ValueError("suite must be dev, local, graph, agent, heldout or all")
