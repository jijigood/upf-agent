"""Recompute a frozen phase-two batch from original text, sources and dispatcher receipts."""

import argparse
import hashlib
import json
from pathlib import Path

from upf_agent.v8.compiler import compile_contract
from upf_agent.v8.experiment_metrics import summarize_events
from upf_agent.v8.models import EvaluationContext, EvidenceBinding, OrchestrationTask, digest
from upf_agent.v8.phase1 import write_json
from upf_agent.v8.plan_capabilities import compile_capabilities
from upf_agent.v8.plan_generator import generation_schema
from upf_agent.v8.plan_ir import CaseGold
from upf_agent.v8.plan_metrics import score_case, summarize_cases
from upf_agent.v8.plan_translator import translate_plan
from upf_agent.v8.plan_validator import validate_plan


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in (path / "metrics-by-case.jsonl").read_text().splitlines()]


def audit(pilot, baseline, output):
    if output.exists():
        raise ValueError("audit exists; choose a new audit ID")
    current, old = rows(pilot), rows(baseline)
    protocol = read(pilot / "protocol.json")
    assert len(current) == 4 * protocol["cases"]
    hashes = {}
    for batch in [baseline, pilot]:
        manifest = read(batch / "sha256_manifest.json")
        for name, sha in manifest.items():
            assert hashlib.sha256((batch / name).read_bytes()).hexdigest() == sha, name
        hashes[batch.name] = len(manifest)
    identical_references = []
    for case in protocol["selected_case_ids"]:
        name = f"pilot-{case:02d}"
        assert (pilot / "cases" / name / "case-reference.json").read_bytes() == (
            baseline / "cases" / name / "case-reference.json"
        ).read_bytes()
        identical_references.append(name)
    instructions = read(pilot / "common-head.json")["instructions"]
    old_index = {(r["case_id"], r["method"]): r for r in old}
    changes, costs, supplementary, collection, p5_native = [], {}, [], [], []
    for row in current:
        method = protocol["methods"].index(row["method"]) + 1
        folder = pilot / "runs" / row["case_id"] / f"method-{method}"
        task = OrchestrationTask.model_validate(read(folder / "task.json"))
        context = EvaluationContext.model_validate(read(folder / "context.json"))
        bindings = [EvidenceBinding.model_validate(b) for b in read(folder / "bindings.json")]
        gold = CaseGold.model_validate(
            read(pilot / "cases" / row["case_id"] / "case-reference.json")
        )
        text = (folder / "raw-plan.txt").read_text()
        raw = read(folder / "raw-output.json")
        assert text == raw["content"]
        assert hashlib.sha256(text.encode()).hexdigest() == raw["content_sha256"]
        translated = translate_plan(text)
        assert translated.model_dump(mode="json") == read(folder / "translation.json")
        validation = validate_plan(
            translated.plan, bindings, task, compile_contract(task), context, folder / "runtime"
        )
        assert validation.model_dump(mode="json") == read(folder / "validation.json")
        metrics = score_case(translated.plan, validation, gold)
        if row["method_failure"] or row["generation_error"] or row["finish_reason"] == "length":
            metrics["reference_pass"] = False
            if metrics["vpr"] is not None:
                metrics["vpr"] = False
        assert all(row[key] == value for key, value in metrics.items())
        events = [json.loads(line) for line in (folder / "traces.jsonl").read_text().splitlines()]
        cost = summarize_events(events)
        assert cost == read(folder / "metrics.json")
        assert cost["invocations"]["tool"]["attempted"] <= 8
        assert cost["invocations"]["model"]["attempted"] <= 8
        traces = [
            json.loads(line) for line in (folder / "model-traces.jsonl").read_text().splitlines()
        ]
        for trace in traces:
            serialized = json.dumps(trace["request"], ensure_ascii=False)
            assert "case-reference" not in serialized and "required_actions" not in serialized
        head = next(t for t in traces if t["operation"] == "common_plan_head")
        assert head["request"]["messages"][0]["content"] == instructions
        assert digest(head["request"]["messages"]) == raw["request_sha256"]
        request = json.loads(head["request"]["messages"][1]["content"])
        caps = compile_capabilities(task, context, bindings, folder / "runtime")
        assert request["generation_capabilities"] == caps
        schema = generation_schema(
            bindings, request["trusted_header"], capabilities=caps, task=task
        )
        assert schema == head["request"]["response_format"]["json_schema"]["schema"]
        assert request["generation_schema_sha256"] == digest(schema)
        assert "plan_schema" not in request
        assert not caps["uses_gold_or_contract_states"]
        assert validation.rewritten_steps == 0 and not validation.execute_authorized
        if method == 3:
            assert not list(folder.glob("runtime/runs/v8/*/subgraph-*.json"))
        if method == 4:
            assert list(folder.glob("runtime/runs/v8/*/subgraph-*.json"))
        if method in {3, 4}:
            native = read(folder / "v8-native.json")
            gaps = native["query_plan"]["gaps"]
            assert "scope_denied" not in gaps, (row["case_id"], row["method"], gaps)
            coverage = sorted({(b["object_id"], b["category"]) for b in native["bindings"]})
            if row["case_id"] == "pilot-01":
                assert {(obj, kind) for obj in task.objects for kind in ["K", "R", "P"]} <= {
                    tuple(pair) for pair in coverage
                }
                assert all(s["code"] == "satisfied" for s in native["states"].values())
                assert not caps["unresolved"]
            collection.append(
                {
                    "case_id": row["case_id"],
                    "method": row["method"],
                    "actual_object_kind_coverage": coverage,
                    "query_gaps": gaps,
                    "state_codes": {key: value["code"] for key, value in native["states"].items()},
                }
            )
        if method == 2:
            native = read(folder / "p5-native.json")
            p5_native.append(
                {
                    "case_id": row["case_id"],
                    "gaps": native["gaps"],
                    "planning_failure_retained": any("规划失败" in gap for gap in native["gaps"]),
                    "actual_object_kind_coverage": sorted(
                        {(b.object_id, b.category) for b in bindings}
                    ),
                }
            )
        prior = old_index[(row["case_id"], row["method"])]
        changes.append(
            {
                "case_id": row["case_id"],
                "method": row["method"],
                "old_validator": prior["validator_pass"],
                "new_validator": row["validator_pass"],
                "old_reference": prior["reference_pass"],
                "new_reference": row["reference_pass"],
                "issues": row["issues"],
            }
        )
        if translated.plan:
            requests = [s for s in translated.plan.steps if s.action_type == "REQUEST_EVIDENCE"]
            assert not requests or caps["unresolved"]
            assert all(s.parameters["missing"] == caps["unresolved"] for s in requests)
            supplementary.append(
                {
                    "case_id": row["case_id"],
                    "method": row["method"],
                    "unresolved": caps["unresolved"],
                    "request_steps": len(requests),
                    "active_outlet": any(
                        s.action_type in {"REQUEST_EVIDENCE", "HUMAN_REVIEW"}
                        for s in translated.plan.steps
                    ),
                    "outlet_texts": [
                        s.text
                        for s in translated.plan.steps
                        if s.action_type in {"REQUEST_EVIDENCE", "HUMAN_REVIEW"}
                    ],
                }
            )
        aggregate = costs.setdefault(
            row["method"],
            {
                "tool_attempts": 0,
                "model_attempts": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "elapsed_seconds": [],
                "unknown_token_receipts": 0,
            },
        )
        aggregate["tool_attempts"] += cost["invocations"]["tool"]["attempted"]
        aggregate["model_attempts"] += cost["invocations"]["model"]["attempted"]
        aggregate["elapsed_seconds"].append(row["elapsed_seconds"])
        if cost["token_usage"]["input"] is None:
            aggregate["unknown_token_receipts"] += 1
        else:
            aggregate["input_tokens"] += cost["token_usage"]["input"]
            aggregate["output_tokens"] += cost["token_usage"]["output"]
    result = {
        "schema_version": "v8-phase2-full-audit-1",
        "runs_recomputed": len(current),
        "cost_records_recomputed": len(current),
        "new_model_calls": 0,
        "verified_hashes": hashes,
        "identical_references": identical_references,
        "synthetic": True,
        "reviewed": False,
        "vpr": None,
        "new_heldout_consumed": False,
        "per_method": {
            m: {
                "old": summarize_cases(
                    [r for r in old if r["method"] == m and r["case_id"] in identical_references]
                ),
                "new": summarize_cases([r for r in current if r["method"] == m]),
            }
            for m in protocol["methods"]
        },
        "changes": changes,
        "costs": costs,
        "supplementation": supplementary,
        "collection": collection,
        "scope_denied_count": 0,
        "normal_case_source_coverage_verified": True,
        "p5_native": p5_native,
        "p5_native_planning_failures": sum(r["planning_failure_retained"] for r in p5_native),
        "limitations": [
            "joint shared generation prior and planner prompt revision",
            "same synthetic development cases; no independent task count claim",
            "physical source-read counts remain unknown; verification IO is in elapsed time",
            "no causal, latency superiority or expert VPR claim",
            "safe common-head plans do not certify native P5 collection/planning success",
            "legacy P5 registered-device checks differ from v8 multi-object Contract dispatch",
            "elapsed starts at PlanBudget; fixture copying and initial compilation are setup",
        ],
    }
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.pilot, args.baseline, args.output)
    print(
        json.dumps(
            {
                "runs_recomputed": report["runs_recomputed"],
                "verified_hashes": report["verified_hashes"],
            }
        )
    )
