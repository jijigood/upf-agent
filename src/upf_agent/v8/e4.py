"""Four-mode, CPU-only E4 experiment with independent Full traversal and frozen deltas."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import resource
from collections import Counter
from pathlib import Path
from time import perf_counter, process_time

from upf_agent.v8.actions import full_actions
from upf_agent.v8.dependency import LocalEvaluator, complete_results
from upf_agent.v8.evaluation import BATCHES, SEEDS, SIZES, UPDATES, percentiles, scaling_fixture
from upf_agent.v8.evaluator import full_evaluate
from upf_agent.v8.experiment_manifest import source_manifest
from upf_agent.v8.models import Requirement, digest

MODES = {
    "L0-Full": None,
    "L1-Local-no-cutoff": {"cutoff": False, "audit_reads": False},
    "L2-Local-cutoff": {"cutoff": True, "audit_reads": False},
    "L3-Local-audited-cutoff": {"cutoff": True, "audit_reads": True},
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class FullEvaluator:
    """No DependencyIndex is constructed or consulted, including during initialization."""

    def __init__(self, task, contract, bindings, actions, batch="initial"):
        self.task, self.contract, self.templates = task, contract, actions
        started = perf_counter()
        self.update(bindings, batch)
        self.initial_cost = {"total_seconds": perf_counter() - started, "index_seconds": 0.0}

    def update(self, bindings, batch, now=None):
        started = perf_counter()
        self.states = full_evaluate(self.task, self.contract, bindings, batch, now)
        self.actions = full_actions(
            self.task, self.contract, bindings, self.states, batch, self.templates, now=now
        )
        elapsed = perf_counter() - started
        return {
            "mode": "full",
            "reason": None,
            "recomputed_states": len(self.states),
            "recomputed_actions": len(self.actions),
            "cost": {"full_evaluation_seconds": elapsed},
        }


def make_evaluator(mode, task, contract, bindings, actions):
    flags = MODES[mode]
    if flags is None:
        return FullEvaluator(task, contract, bindings, actions)
    return LocalEvaluator(task, contract, bindings, actions=actions, **flags)


def delta(bindings, field_map, count, rng):
    """Copy only mutated records; persisted deltas plus the seeded fixture fully replay inputs."""
    selected = rng.sample(sorted(field_map), count)
    grouped = {}
    for field in selected:
        position = field_map[field]
        if position not in grouped:
            grouped[position] = bindings[position].model_copy(deep=True)
        grouped[position].fields[field] = rng.randrange(0, 4)
    changed = list(bindings)
    for position, binding in grouped.items():
        binding.revision += 1
        changed[position] = binding
    return changed, {
        "fields": selected,
        "records": {
            str(i): {"revision": b.revision, "fields": b.fields} for i, b in grouped.items()
        },
    }


def matrix_groups(profile):
    if profile == "small":
        return [(s, str(u), 3, 10, False) for s in ("S", "M") for u in (1, 5, 10)]
    if profile == "full":
        return [(s, str(u), len(SEEDS), BATCHES, False) for s in SIZES for u in UPDATES]
    if profile == "stress":
        return [
            (s, u, 3, 3, fanout)
            for s in SIZES
            for u, fanout in [("10%", False), ("50%", False), ("100%", False), ("1", True)]
        ]
    raise ValueError("unknown matrix profile")


def run_group(output, spec):
    size, update, seeds, batches, fanout = spec
    name = f"{size}-{update}" + ("-fanout" if fanout else "")
    folder = output / name
    folder.mkdir()
    rows, initial, failures = [], [], []
    matched = Counter()
    columns = [
        "size",
        "updates",
        "seed",
        "batch",
        "order",
        "method",
        "seconds",
        "cpu_seconds",
        "rss_high_water_kib",
        "recomputed_states",
        "recomputed_actions",
        "reused_states",
        "invalidated_readers",
        "consistent",
        "fallback",
        "reason",
        "fingerprint_seconds",
        "index_integrity_audit_seconds",
        "observed_read_audit_seconds",
        "observed_read_audits",
        "propagation_recompute_seconds",
        "fallback_seconds",
        "full_evaluation_seconds",
    ]
    with (
        (folder / "batches.csv").open("w", newline="") as stream,
        (folder / "changes.jsonl").open("w") as changes,
    ):
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for seed in range(seeds):
            task, contract, bindings, actions, field_map = scaling_fixture(
                folder / "sources", size, seed
            )
            if fanout:
                contract.requirements.extend(
                    Requirement(
                        requirement_id=f"C:fanout:{i}",
                        kind="C",
                        objects=[actions[i].target],
                        upstream=["R:0"],
                        version_required=False,
                        freshness_seconds=None,
                    )
                    for i in range(len(actions))
                )
                for i, action in enumerate(actions):
                    action.required_states.append(f"C:fanout:{i}")
            count = (
                max(1, len(field_map) * int(update[:-1]) // 100)
                if update.endswith("%")
                else int(update)
            )
            write_json(
                folder / f"fixture-{seed}.json",
                {
                    "task": task.model_dump(mode="json"),
                    "contract": contract.model_dump(mode="json"),
                    "bindings": [b.model_dump(mode="json") for b in bindings],
                    "actions": [a.model_dump(mode="json") for a in actions],
                    "field_map": field_map,
                },
            )
            methods = {}
            order_rng, rng = random.Random(10000 + seed), random.Random(seed)
            order = list(MODES)
            order_rng.shuffle(order)
            for mode in order:
                cpu = process_time()
                methods[mode] = make_evaluator(mode, task, contract, bindings, actions)
                initial.append(
                    {
                        "seed": seed,
                        "method": mode,
                        "cpu_seconds": process_time() - cpu,
                        **methods[mode].initial_cost,
                    }
                )
            for batch in range(batches):
                # Fanout updates always hit the common root; ordinary sequences preserve V2 RNG.
                mapping = {f: i for f, i in field_map.items() if i == 0} if fanout else field_map
                changed, receipt = delta(bindings, mapping, count, rng)
                batch_id = f"{name}-{seed}-{batch}"
                changes.write(json.dumps({"batch": batch_id, **receipt}, ensure_ascii=False) + "\n")
                order_rng.shuffle(order)
                results, costs = {}, {}
                for position, mode in enumerate(order):
                    evaluator = methods[mode]
                    cpu, started = process_time(), perf_counter()
                    costs[mode] = evaluator.update(changed, batch_id)
                    seconds, cpu_seconds = perf_counter() - started, process_time() - cpu
                    # Serialization, oracle comparison and persistence are outside latency timing.
                    results[mode] = complete_results(evaluator.states, evaluator.actions)
                    row = {
                        "size": size,
                        "updates": update,
                        "seed": seed,
                        "batch": batch_id,
                        "order": position,
                        "method": mode,
                        "seconds": seconds,
                        "cpu_seconds": cpu_seconds,
                        "rss_high_water_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        "recomputed_states": costs[mode]["recomputed_states"],
                        "recomputed_actions": costs[mode]["recomputed_actions"],
                        "reused_states": costs[mode].get("reused_states", 0),
                        "invalidated_readers": costs[mode].get("invalidated_readers", 0),
                        "fallback": costs[mode]["mode"],
                        "reason": costs[mode]["reason"],
                        **costs[mode]["cost"],
                    }
                    rows.append(row)
                reference = results["L0-Full"]
                for row in rows[-4:]:
                    mode = row["method"]
                    row["consistent"] = results[mode] == reference
                    matched[mode] += int(row["consistent"])
                    writer.writerow({k: row.get(k, 0) for k in columns})
                    if not row["consistent"]:
                        failures.append({"batch": batch_id, "method": mode})
                        write_json(
                            folder / "failures" / f"{batch_id}-{mode}.json",
                            {
                                "full": reference,
                                "actual": results[mode],
                                "delta": receipt,
                                "receipt": costs[mode],
                            },
                        )
                bindings = changed
                stream.flush()
                changes.flush()
                write_json(
                    folder / "progress.json",
                    {
                        "completed_method_batches": len(rows),
                        "failures": failures,
                        "expected_method_batches": seeds * batches * len(MODES),
                    },
                )
                if failures:
                    raise RuntimeError(f"unexpected E4 mismatch; retained snapshot in {folder}")
    summaries = {}
    for mode in MODES:
        subset = [r for r in rows if r["method"] == mode]
        summaries[mode] = {
            "batches": len(subset),
            "consistent": matched[mode],
            "seconds": percentiles([r["seconds"] for r in subset]),
            "total_seconds": sum(r["seconds"] for r in subset),
            "cpu_seconds": sum(r["cpu_seconds"] for r in subset),
            "mean_recomputed_states": sum(r["recomputed_states"] for r in subset) / len(subset),
            "mean_recomputed_actions": sum(r["recomputed_actions"] for r in subset) / len(subset),
            "fallbacks": sum(r["fallback"] == "full_fallback" for r in subset),
            "cold_seconds": percentiles(
                [r["total_seconds"] for r in initial if r["method"] == mode]
            ),
            "cost_totals": {
                k: sum(r.get(k, 0) for r in subset)
                for k in columns
                if k.endswith("_seconds") and k not in {"cpu_seconds"}
            },
        }
    full_time = summaries["L0-Full"]["total_seconds"]
    for summary in summaries.values():
        summary["paired_total_full_over_method"] = full_time / summary["total_seconds"]
    result = {
        "spec": spec,
        "methods": summaries,
        "failures": failures,
        "initialization": initial,
        "pass": not failures,
    }
    write_json(folder / "summary.json", result)
    return name, result


def run_matrix(project: Path, output: Path, profile="small", workers=1, correctness=None):
    project, output = project.resolve(), output.resolve()
    if output.exists():
        raise ValueError("output already exists; use a new experiment ID")
    if project != Path(__file__).resolve().parents[3]:
        raise ValueError("declared project must match imported source root")
    if workers not in (1, 4):
        raise ValueError("workers must be 1 (latency) or 4 (throughput only)")
    source = source_manifest(project)
    if profile != "small":
        if correctness is None:
            raise ValueError("small correctness summary is required before expansion")
        gate = json.loads(correctness.read_text())
        if not (
            gate.get("pass")
            and gate.get("profile") == "small"
            and gate.get("method_batches") == 720
            and gate.get("source") == source
        ):
            raise ValueError("expansion requires 720 passing method batches from identical source")
    specs = matrix_groups(profile)
    output.mkdir(parents=True)
    write_json(output / "source_manifest.json", source)
    protocol = {
        "schema": "v8-e4-four-mode-1",
        "profile": profile,
        "groups": specs,
        "workers": workers,
        "measurement": "latency" if workers == 1 else "throughput-only",
        "synthetic": True,
        "reviewed": False,
        "heldout_consumed": False,
        "model_calls": 0,
        "real_network_writes": 0,
        "modes": MODES,
        "initial_read_certification_and_index_integrity": "mandatory in all Local modes",
        "audit_ablation": "L3 additionally checks each recomputed reader's actual reads",
        "comparison": "complete certification and semantics, excluding computed_batch_id",
        "timing": "per-method update; excludes serialization/comparison/input generation",
        "order": "seeded random order for every initialization and batch",
        "warm": "initialized evaluators; no artificial cache flush; not OS cold-cache latency",
        "rss": "Linux process cumulative high water (KiB), not per-method allocation",
        "fallback_seconds": "subset of propagation_recompute_seconds; do not add twice",
        "common_rules": True,
        "python": platform.python_version(),
        "host": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "load_average_at_start": os.getloadavg(),
            "exclusive_machine": False,
        },
        "correctness_summary_sha256": (
            hashlib.sha256(correctness.read_bytes()).hexdigest() if correctness else None
        ),
        "expected_method_batches": sum(s[2] * s[3] * len(MODES) for s in specs),
    }
    write_json(output / "protocol.json", protocol)
    started, cpu = perf_counter(), process_time()
    results = {}

    def completed(name, result):
        results[name] = result
        write_json(
            output / "progress.json",
            {
                "completed_groups": sorted(results),
                "completed_method_batches": sum(
                    sum(m["batches"] for m in g["methods"].values()) for g in results.values()
                ),
                "expected_method_batches": protocol["expected_method_batches"],
            },
        )
        print(
            f"completed {name}: {sum(m['batches'] for m in result['methods'].values())}", flush=True
        )

    if workers == 1:
        for spec in specs:
            completed(*run_group(output, spec))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_group, output, spec) for spec in specs]
            for future in as_completed(futures):
                completed(*future.result())
    count = sum(sum(m["batches"] for m in g["methods"].values()) for g in results.values())
    unchanged = source_manifest(project) == source
    summary = {
        **protocol,
        "groups": results,
        "method_batches": count,
        "source": source,
        "source_unchanged": unchanged,
        "elapsed_seconds": perf_counter() - started,
        "parent_cpu_seconds": process_time() - cpu,
        "pass": unchanged
        and count == protocol["expected_method_batches"]
        and all(g["pass"] for g in results.values()),
    }
    summary["result_digest"] = digest(summary)
    write_json(output / "summary.json", summary)
    return summary
