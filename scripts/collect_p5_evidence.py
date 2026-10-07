"""Archive selected real CLI runs, preserve failed trials, and verify published citations."""

import argparse
import json
import shutil
from pathlib import Path

from upf_agent.log_parser import file_sha
from upf_agent.log_tools import write_json
from upf_agent.p3_answer import source_lookup

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="label:run_id")
    args = parser.parse_args()
    directory = ROOT / "evidence/p5"
    cases = []
    for spec in args.run:
        label, run_id = spec.split(":", 1)
        if not label.replace("_", "").isalnum() or len(run_id) != 32:
            raise ValueError("invalid evidence label or run_id")
        source = ROOT / "runs/p5" / run_id
        report = json.loads((source / "report.json").read_text())
        manifest = json.loads((source / "manifest.json").read_text())
        metrics = json.loads((source / "metrics.json").read_text())
        evidence = {entry["evidence_id"]: entry for entry in report["evidence"]}
        assert all(eid.startswith(run_id + ":") for eid in evidence)
        for fact in report["fact_records"]:
            assert set(fact["evidence_ids"]) <= evidence.keys()
        manual_generation = manifest["task"]["manual_generation"]
        for entry in evidence.values():
            if entry["kind"] == "manual":
                original = source_lookup(ROOT, manual_generation, entry["native_id"])
                assert original["source_sha256"] == entry["source_sha256"]
                assert entry["excerpt"] in original["text"]
        for fact in report["fact_records"]:
            refs = [evidence[eid] for eid in fact["evidence_ids"]]
            if all(entry["kind"] == "manual" for entry in refs):
                assert all(
                    line in "\n".join(entry["excerpt"] for entry in refs)
                    for line in fact["text"].splitlines()
                )
        assert metrics["tool_attempts"] <= 8 and metrics["rounds"] <= 3
        assert metrics["parameter_repairs"] <= 1 and metrics["model_requests_reserved"] <= 8
        destination = directory / label
        if destination.exists():
            raise ValueError("evidence label already exists; use a new label")
        shutil.copytree(source, destination)
        cases.append(
            {
                "label": label,
                "run_id": run_id,
                "status": report["status"],
                "degraded": report["degraded"],
                "facts": len(report["facts"]),
                "evidence": len(evidence),
                "metrics": metrics,
                "pending_nodes": manifest["pending_nodes"],
                "report_sha256": file_sha(destination / "report.json"),
                "manifest_sha256": file_sha(destination / "manifest.json"),
            }
        )
    tests = ROOT / "runs/p5-tests.log"
    shutil.copy2(tests, directory / "tests.log")
    write_json(
        directory / "regression.json",
        {
            "kind": "development_functional_validation_not_business_accuracy",
            "cases": cases,
            "tests_output": tests.read_text(),
            "fixed_contract_requests": 100,
            "contract_suite_scope": "code_schema_and_scope_validation_not_model_parameter_accuracy",
            "human_business_signoff": False,
            "formal_business_accuracy_claim": False,
            "pipeline_files": {
                str(path.relative_to(ROOT)): file_sha(path)
                for path in (ROOT / "src/upf_agent").glob("agent_*.py")
            },
            "uv_lock_sha256": file_sha(ROOT / "uv.lock"),
        },
    )
    archive = directory / "archive"
    for run_id in ("5b8cf78d6fae4eb99564d793a02f0813", "54a0f162b8df485e89d54064e6239772"):
        source = ROOT / "runs/p5" / run_id
        if source.exists() and not (archive / run_id).exists():
            shutil.copytree(source, archive / run_id)
    print(
        json.dumps(
            {"cases": cases, "report": str(directory / "regression.json")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
