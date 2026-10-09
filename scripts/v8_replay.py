"""Replay consumed, frozen inputs; this is never a new blind acceptance batch."""

import argparse
import hashlib
import json
from pathlib import Path

from upf_agent.v8.actions import full_actions
from upf_agent.v8.dependency import semantic_results
from upf_agent.v8.engine import OrchestrationEngine
from upf_agent.v8.evaluation import write_json
from upf_agent.v8.evaluator import full_evaluate
from upf_agent.v8.models import OrchestrationData, OrchestrationTask, digest


def replay(root: Path, frozen_file: Path, output: Path):
    if output.exists():
        raise ValueError("replay destination exists")
    frozen = json.loads(frozen_file.read_text())
    original_root = next(
        Path(name).parent.parent for name in frozen["files"] if name.endswith("/config/models.yaml")
    )
    for name, expected in frozen["files"].items():
        path = Path(name)
        if path.is_relative_to(original_root):
            path = root / path.relative_to(original_root)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen input/code hash mismatch: {name}")
    output.mkdir(parents=True)
    rows = []
    for case in frozen["cases"]:
        task = OrchestrationTask.model_validate(case["task"])
        task.run_id = digest(["known-input-replay", task.run_id])[:32]
        engine = OrchestrationEngine(root)
        try:
            result = engine.run(task)
        finally:
            engine.close()
        data = OrchestrationData.model_validate(result)
        states = full_evaluate(data.task, data.contract, data.bindings, data.batch_id)
        actions = full_actions(data.task, data.contract, data.bindings, states, data.batch_id)
        identical = semantic_results(states, actions) == semantic_results(
            data.states, {a.action_id: a for a in data.actions}
        )
        allowed = sorted(a.action_id for a in data.actions if a.outcome == "suggest_allowed")
        oracle_equal = allowed == case["oracle"]["allowed_action_ids"]
        row = {
            "case": case["oracle"]["case_id"],
            "full_local_equal": identical,
            "oracle_equal": oracle_equal,
            "allowed_action_ids": allowed,
            "run_id": task.run_id,
        }
        rows.append(row)
        write_json(output / f"{row['case']}.json", row)
    recorded = json.loads((frozen_file.parent / "summary.json").read_text())
    oracle_by_case = {c["oracle"]["case_id"]: c["oracle"] for c in frozen["cases"]}
    archived_decisions_equal = all(
        sorted(set(row["suggested"]) & set(oracle_by_case[row["case"]]["blocked_action_ids"]))
        == row["unsafe_releases"]
        and sorted(set(row["suggested"]) & set(oracle_by_case[row["case"]]["allowed_action_ids"]))
        == row["legal_accepted"]
        for row in recorded["results"]
    )
    summary = {
        "mode": "known-input replay; no new blind acceptance",
        "freeze_sha256": digest(frozen),
        "cases": rows,
        "recorded_model_decisions_rechecked": archived_decisions_equal,
        "model_calls": 0,
        "pass": archived_decisions_equal
        and all(r["full_local_equal"] and r["oracle_equal"] for r in rows),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = replay(args.root, args.freeze, args.output)
    raise SystemExit(0 if result["pass"] else 2)
