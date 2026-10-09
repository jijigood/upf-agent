"""CPU-only phase-one acceptance; no private corpus, model service or heldout."""

import csv
import json
import random
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

from upf_agent.v8.actions import full_actions
from upf_agent.v8.adapters import collect
from upf_agent.v8.compiler import compile_contract, compile_query_plan
from upf_agent.v8.dependency import LocalEvaluator, complete_results
from upf_agent.v8.engine import OrchestrationEngine
from upf_agent.v8.evaluation import scaling_fixture
from upf_agent.v8.evaluator import full_evaluate
from upf_agent.v8.experiment_manifest import source_manifest
from upf_agent.v8.fixtures import example
from upf_agent.v8.graph import MemoryGraphStore
from upf_agent.v8.models import EvidenceBinding


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare_synthetic_root(root: Path, template="upf_n4_fault"):
    if root.exists():
        raise ValueError("synthetic root already exists; use a new run directory")
    root.mkdir(parents=True)
    task = example(root, template=template)
    write_json(root / "data/v8/manual_graph.json", MemoryGraphStore().snapshot())
    return task


def prepared(root):
    task = prepare_synthetic_root(root)
    contract = compile_contract(task)
    bindings = []
    for step in compile_query_plan(task, contract).steps:
        packet = collect(root, task, contract, step, step.step_id)
        bindings.extend(EvidenceBinding.model_validate(b) for b in packet["bindings"])
    return task, contract, bindings


def _compare(task, contract, bindings, local, batch, now=None):
    now = task.as_of if now is None else now
    states = full_evaluate(task, contract, bindings, batch, now)
    actions = full_actions(
        task, contract, bindings, states, batch, list(local.index.actions.values()), now=now
    )
    update = local.update(bindings, batch, now)
    return {
        "batch": batch,
        "equal": complete_results(states, actions) == complete_results(local.states, local.actions),
        "mode": update["mode"],
        "stale_action_batches": sum(a.batch_id != batch for a in local.actions.values()),
        "recomputed_states": update["recomputed_states"],
        "recomputed_actions": update.get("recomputed_actions", 0),
    }


def _matrix(output, seeds, batches):
    rows = []
    for size in ["S", "M"]:
        for count in [1, 5, 10]:
            for seed in range(seeds):
                task, contract, bindings, actions, field_map = scaling_fixture(
                    output / "sources" / f"{size}-{count}-{seed}", size, seed
                )
                local = LocalEvaluator(task, contract, bindings, actions=actions)
                rng = random.Random(seed)
                for batch in range(batches):
                    changed = deepcopy(bindings)
                    selected = rng.sample(sorted(field_map), count)
                    positions = set()
                    for name in selected:
                        position = field_map[name]
                        changed[position].fields[name] = rng.randrange(0, 4)
                        positions.add(position)
                    for position in positions:
                        changed[position].revision += 1
                    row = _compare(task, contract, changed, local, f"{size}-{count}-{seed}-{batch}")
                    rows.append({"size": size, "updates": count, "seed": seed, **row})
                    if not row["equal"]:
                        write_json(
                            output / "failures" / f"{row['batch']}.json",
                            {
                                "local": complete_results(local.states, local.actions),
                                "inputs": [b.model_dump(mode="json") for b in changed],
                            },
                        )
                    bindings = changed
    with (output / "batches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "matrix": f"S/M x 1/5/10 fields x {seeds} seeds x {batches} batches",
        "batches": len(rows),
        "complete_equal": sum(r["equal"] for r in rows),
        "stale_action_batches": sum(r["stale_action_batches"] for r in rows),
        "performance_claim": False,
    }


MUTATIONS = [
    "noop",
    "unrelated",
    "n4_disabled",
    "mtu",
    "revision_only",
    "purpose_missing",
    "purpose_wrong",
    "version_wrong",
    "version_missing",
    "product_wrong",
    "knowledge_unverified",
    "knowledge_deleted",
    "runtime_interface",
    "window_missing",
    "future_window",
    "stale_window",
    "validity_expired",
    "runtime_unverified",
    "runtime_deleted",
    "pair_side_missing",
    "service_wrong",
    "peer_wrong",
    "direction_wrong",
    "config_wrong",
    "mapping_wrong",
    "pair_deleted",
    "cross_task",
    "as_of_expiry",
    "permission_denied",
    "contract_revision",
]


def _boundaries(output, local_options=None):
    rows = []
    for name in MUTATIONS:
        task, contract, original = prepared(output / "sources" / name)
        local = LocalEvaluator(task, contract, original, "initial", **(local_options or {}))
        bindings = deepcopy(original)
        k, r, p = [next(b for b in bindings if b.category == kind) for kind in ("K", "R", "P")]
        now = task.as_of
        if name == "unrelated":
            r.fields["unused"] = "irrelevant"
        elif name == "n4_disabled":
            r.fields["n4_enabled"] = False
        elif name == "mtu":
            r.fields["mtu"] = 900
        elif name == "revision_only":
            r.revision += 1
        elif name == "purpose_missing":
            k.purposes = []
        elif name == "purpose_wrong":
            k.purposes = ["log_observation"]
        elif name == "version_wrong":
            k.version = "4.0"
        elif name == "version_missing":
            k.version = None
        elif name == "product_wrong":
            k.product = "other-product"
        elif name == "knowledge_unverified":
            k.source_verified = False
        elif name == "knowledge_deleted":
            k.deleted = True
        elif name == "runtime_interface":
            r.fields["interface"] = "N3"
        elif name == "window_missing":
            r.observed = None
        elif name == "future_window":
            r.observed = r.observed.model_copy(
                update={
                    "start": now + timedelta(seconds=10),
                    "end": now + timedelta(seconds=70),
                }
            )
        elif name == "stale_window":
            r.observed = r.observed.model_copy(
                update={
                    "start": now - timedelta(hours=2),
                    "end": now - timedelta(hours=1),
                }
            )
        elif name == "validity_expired":
            r.valid = r.observed.model_copy()
        elif name == "runtime_unverified":
            r.source_verified = False
        elif name == "runtime_deleted":
            r.deleted = True
        elif name == "pair_side_missing":
            bindings = [b for b in bindings if not (b.category == "P" and b.object_id == "SMF-B")]
        elif name == "service_wrong":
            p.fields["service_id"] = "other-service"
        elif name == "peer_wrong":
            p.fields["peer_endpoint"] = "192.0.2.99"
        elif name == "direction_wrong":
            p.fields["direction"] = "unknown"
        elif name == "config_wrong":
            p.config_version = "other-config"
        elif name == "mapping_wrong":
            p.fields["mapping_version"] = "other-map"
        elif name == "pair_deleted":
            p.deleted = True
        elif name == "cross_task":
            r.task_id = "f" * 32
        elif name == "as_of_expiry":
            now += timedelta(hours=1)
        elif name == "permission_denied":
            task.permission.objects = []
        elif name == "contract_revision":
            contract.version = "2"
        row = _compare(task, contract, bindings, local, name, now)
        row["synthetic"] = True
        rows.append(row)
    write_json(output / "boundary-cases.json", rows)
    return {"cases": len(rows), "complete_equal": sum(r["equal"] for r in rows)}


def run_phase1(project: Path, output: Path, seeds=3, batches=10):
    project, output = project.resolve(), output.resolve()
    if project != Path(__file__).resolve().parents[3]:
        raise ValueError("declared project differs from the imported code root")
    if output.exists():
        raise ValueError("experiment output already exists; choose a new ID")
    if seeds < 1 or batches < 1:
        raise ValueError("seeds and batches must be positive")
    output.mkdir(parents=True)
    manifest = source_manifest(project)
    write_json(output / "source_manifest.json", manifest)
    write_json(
        output / "protocol.json",
        {
            "stage": "phase1-E0-foundation-and-E4-correctness",
            "synthetic": True,
            "reviewed": False,
            "heldout_consumed": False,
            "models_enabled": False,
            "seeds": seeds,
            "batches": batches,
            "comparison": (
                "semantics + batch + evaluation time + context (excluding computation origin)"
            ),
        },
    )
    write_json(
        output / "method_manifest.json",
        {
            "executed": [
                "Full deterministic rules",
                "Local with observed-read audit",
                "v8 synthetic engine",
            ],
            "not_executed": [
                "P5 live planner",
                "natural-language plan evaluation",
                "PPR",
                "Open5GS",
            ],
            "metric_boundary": "dispatcher",
            "unmeasured": ["physical source reads", "retrieval cache hits"],
        },
    )
    engines = []
    for template in ("upf_n4_fault", "upf_n3_fault", "manual_operation", "log_observation"):
        root = output / "engine-runs" / template
        task = prepare_synthetic_root(root, template)
        engine = OrchestrationEngine(root)
        try:
            result = engine.run(task)
            metrics = engine.ledger.metrics(task.run_id)
            engines.append(
                {
                    "template": template,
                    "run_id": task.run_id,
                    "status": result["status"],
                    "metrics": metrics,
                    "execute_authorized": any(a["execute_authorized"] for a in result["actions"]),
                }
            )
        finally:
            engine.close()
    matrix = _matrix(output, seeds, batches)
    boundaries = _boundaries(output)
    unchanged = source_manifest(project) == manifest
    summary = {
        "stage": "phase1",
        "synthetic": True,
        "reviewed": False,
        "engines": engines,
        "differential": matrix,
        "boundaries": boundaries,
        "source_unchanged_during_run": unchanged,
        "model_calls": 0,
        "real_network_writes": 0,
        "new_heldout_consumed": False,
    }
    summary["pass"] = (
        unchanged
        and matrix["complete_equal"] == matrix["batches"]
        and matrix["stale_action_batches"] == 0
        and boundaries["complete_equal"] == boundaries["cases"]
        and all(e["status"] == "completed" and not e["execute_authorized"] for e in engines)
    )
    write_json(output / "summary.json", summary)
    return summary
