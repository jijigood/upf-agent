"""Run the exact 16,000-batch matrix on the final kernel, in independent CPU lanes."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

from upf_agent.v8.actions import full_actions
from upf_agent.v8.dependency import LocalEvaluator, semantic_results
from upf_agent.v8.evaluation import (
    BATCHES,
    SEEDS,
    SIZES,
    UPDATES,
    percentiles,
    scaling_fixture,
    write_json,
)
from upf_agent.v8.evaluator import full_evaluate


def group(argument):
    import random

    root_text, output_text, size, updates = argument
    root, output = Path(root_text), Path(output_text)
    folder = output / f"{size}-{updates}"
    folder.mkdir(parents=True)
    full_times, local_times, cold, counts = [], [], [], []
    matches = 0
    with (folder / "batches.csv").open("w", newline="") as stream:
        columns = [
            "size",
            "updates",
            "seed",
            "batch",
            "full_seconds",
            "local_seconds",
            "local_states",
            "local_actions",
            "consistent",
            "mode",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for seed in SEEDS:
            # Each lane owns its source files; no concurrent source mutation is possible.
            source_root = root / f"data/v8/scaling-lanes/{size}-{updates}"
            task, contract, bindings, actions, field_map = scaling_fixture(source_root, size, seed)
            started = perf_counter()
            local = LocalEvaluator(task, contract, bindings, actions=actions)
            cold.append(perf_counter() - started)
            rng = random.Random(seed)
            names = sorted(field_map)
            receipts = []
            for batch in range(BATCHES):
                changed = list(bindings)
                selected = rng.sample(names, updates)
                grouped = {}
                for field in selected:
                    i = field_map[field]
                    if i not in grouped:
                        grouped[i] = bindings[i].model_copy(deep=True)
                    grouped[i].fields[field] = rng.randrange(0, 4)
                for i, b in grouped.items():
                    b.revision += 1
                    changed[i] = b
                batch_id = f"{size}-{updates}-{seed}-{batch}"
                started = perf_counter()
                states = full_evaluate(task, contract, changed, batch_id)
                decisions = full_actions(task, contract, changed, states, batch_id, actions)
                full_time = perf_counter() - started
                started = perf_counter()
                receipt = local.update(changed, batch_id)
                local_time = perf_counter() - started
                full = semantic_results(states, decisions)
                partial = semantic_results(local.states, local.actions)
                equal = full == partial
                matches += int(equal)
                if not equal:
                    write_json(
                        folder / f"failure-{batch_id}.json", {"full": full, "local": partial}
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
                                full_time,
                                local_time,
                                receipt["recomputed_states"],
                                receipt.get("recomputed_actions", 0),
                                equal,
                                receipt["mode"],
                            ],
                            strict=True,
                        )
                    )
                )
                receipts.append(
                    {
                        **receipt,
                        "selected": selected,
                        "values": {f: changed[field_map[f]].fields[f] for f in selected},
                    }
                )
                full_times.append(full_time)
                local_times.append(local_time)
                counts.append(receipt["recomputed_states"])
                bindings = changed
            write_json(folder / f"updates-{seed}.json", receipts)
            stream.flush()
    result = {
        "size": size,
        "updates": updates,
        "batches": len(full_times),
        "consistent": matches,
        "full_seconds": percentiles(full_times),
        "local_seconds": percentiles(local_times),
        "cold_seconds": percentiles(cold),
        "full_states": SIZES[size][1],
        "mean_local_states": sum(counts) / len(counts),
        "pass": matches == 1000,
    }
    write_json(folder / "summary.json", result)
    return result


def run(root: Path, output: Path, workers: int):
    if output.exists():
        raise ValueError("output already exists; use a new batch directory")
    output.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    files = [
        *sorted((project / "src/upf_agent").rglob("*.py")),
        Path(__file__),
        project / "uv.lock",
    ]
    manifest = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    write_json(
        output / "freeze.json",
        {
            "code": manifest,
            "sizes": SIZES,
            "updates": UPDATES,
            "seeds": SEEDS,
            "batches": BATCHES,
            "workers": workers,
            "synthetic": True,
        },
    )
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(group, (str(root), str(output), size, updates))
            for size in SIZES
            for updates in UPDATES
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    changed = [
        name
        for name, sha in manifest.items()
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha
    ]
    summary = {
        "batches": sum(r["batches"] for r in results),
        "consistent": sum(r["consistent"] for r in results),
        "groups": sorted(results, key=lambda r: (list(SIZES).index(r["size"]), r["updates"])),
        "workers": workers,
        "frozen_code_unchanged": not changed,
        "changed_files": changed,
        "synthetic": True,
        "model_calls": 0,
        "tool_calls": 0,
    }
    summary["pass"] = summary["batches"] == 16000 and summary["consistent"] == 16000 and not changed
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.root, args.output, args.workers)
