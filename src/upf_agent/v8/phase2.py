"""Versioned four-method development pilot: preserves every original and unreviewed status."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

from upf_agent.config import load_model_config
from upf_agent.v8.common_tools import CommonEvidenceTools
from upf_agent.v8.compiler import compile_contract
from upf_agent.v8.experiment_manifest import source_manifest
from upf_agent.v8.experiment_metrics import MetricCollector
from upf_agent.v8.models import (
    EvaluationContext,
    EvidenceBinding,
    OrchestrationTask,
    canonical,
    digest,
)
from upf_agent.v8.p5_plan_runner import run_p5
from upf_agent.v8.phase1 import write_json
from upf_agent.v8.plan_cases import SPECS, prepare_case
from upf_agent.v8.plan_generator import (
    HEAD_MAX_TOKENS,
    HEAD_VERSION,
    ObservedPlanClient,
    PlanBudget,
    generate_plan,
    generation_manifest,
)
from upf_agent.v8.plan_metrics import score_case, summarize_cases
from upf_agent.v8.plan_review import calibration_report, validate_splits
from upf_agent.v8.plan_translator import translate_plan
from upf_agent.v8.plan_validator import validate_plan
from upf_agent.v8.store import execution_fingerprint
from upf_agent.v8.v8_plan_runner import run_v8

METHODS = [
    "DirectLLM+CommonPlanHead",
    "P5+CommonPlanHead",
    "v8-no-graph+CommonPlanHead",
    "v8-full+CommonPlanHead",
]


def append_jsonl(path, value):
    with path.open("a") as stream:
        stream.write(canonical(value) + "\n")


def _direct(task, tools):
    from upf_agent.agent_contracts import AgentTask, ToolCall
    from upf_agent.contracts import TaskType

    generation = digest([task.observation_files, "public-synthetic-generation"])[:20]
    agent = AgentTask(
        run_id=task.run_id,
        question=task.question,
        task_type=TaskType.DIAGNOSE,
        device_ids=[],
        product=task.product,
        version=task.version,
        log_generation=generation,
        topology_generation=generation,
        manual_generation=generation,
    )
    # Shared context-preparation costs are explicitly charged to DirectLLM.
    calls = [
        ToolCall(name="search_manual", arguments={"query": task.question}),
        ToolCall(name="analyze_log_window", arguments={"log_id": generation}),
    ]
    calls += [
        ToolCall(name="lookup_topology", arguments={"topology_id": generation, "device_id": obj})
        for obj in task.objects
    ]
    known = []
    for call in calls:
        packet = tools.tool(agent, call, known, tools.budget.remaining())
        known.extend(packet.evidence)
    return {
        "context_preparation_tools": len(calls),
        "information_scope": "shared public sources; no graph or Contract states",
    }


def run_method(
    project, case_root, output, case_task, gold, method, client_factory=ObservedPlanClient
):
    run_root = output / "runtime"
    run_root.mkdir(parents=True)
    shutil.copytree(case_root / "data", run_root / "data", dirs_exist_ok=True)
    (run_root / "config").mkdir()
    shutil.copy2(project / "config/models.yaml", run_root / "config/models.yaml")
    task = case_task.model_copy(deep=True)
    task.run_id = digest([gold.case_id, method, str(output.resolve())])[:32]
    task.graph_enabled = method == "v8-full+CommonPlanHead"
    task.execution_fingerprint = execution_fingerprint(run_root)
    contract = compile_contract(task)
    collector = MetricCollector(
        task.run_id, sink=lambda event: append_jsonl(output / "traces.jsonl", event)
    )
    budget = PlanBudget(task, collector)
    tools = CommonEvidenceTools(run_root, task, contract, budget)
    config = load_model_config(project / "config/models.yaml")
    failure = None
    native = None
    with client_factory(
        config.generation["baseline"],
        collector,
        budget,
        sink=lambda trace: append_jsonl(output / "model-traces.jsonl", trace),
    ) as client:
        try:
            if method == METHODS[0]:
                method_context = _direct(task, tools)
            elif method == METHODS[1]:
                native = run_p5(run_root, task, tools, client)
                method_context = {
                    "native_report": native.report.model_dump(mode="json"),
                    "rounds": native.rounds,
                    "gaps": native.gaps,
                }
                write_json(output / "p5-native.json", native.model_dump(mode="json"))
                for line in (
                    (run_root / "runs/p5" / task.run_id / "events.jsonl").read_text().splitlines()
                ):
                    event = json.loads(line)
                    if event["kind"] == "cache_hit":
                        payload = event
                        collector.emit(
                            payload["key"],
                            "tool",
                            payload["call"]["name"],
                            "reused",
                            payload["call"],
                        )

            else:
                native = run_v8(run_root, task, tools)
                task = OrchestrationTask.model_validate(native["task"])
                contract = compile_contract(task)
                method_context = {
                    "states": {
                        rid: {
                            "code": state["code"],
                            "reasons": state["reasons"],
                            "supporting": state["supporting"],
                        }
                        for rid, state in native["states"].items()
                    },
                    "gaps": native["gaps"],
                    "actions": [
                        {
                            "action_id": a["action_id"],
                            "outcome": a["outcome"],
                            "reasons": a["reasons"],
                            "execute_authorized": False,
                        }
                        for a in native["actions"]
                    ],
                }
                write_json(output / "v8-native.json", native)
        except Exception as error:
            failure = f"{type(error).__name__}:{error}"
            method_context = {"method_failure": failure}
        if method.startswith("v8") and isinstance(native, dict):
            bindings = [EvidenceBinding.model_validate(b) for b in native["bindings"]]
        else:
            bindings = list(tools.bindings.values())
        context = EvaluationContext.from_task(task, contract, f"{gold.case_id}-plan-1")
        raw = generate_plan(
            method,
            task,
            context,
            bindings,
            client,
            method_context=method_context,
            source_root=run_root,
        )
    translation = translate_plan(raw.content)
    validation = validate_plan(translation.plan, bindings, task, contract, context, run_root)
    metrics = score_case(translation.plan, validation, gold)
    if failure or raw.error or raw.finish_reason == "length":
        metrics["reference_pass"] = False
        if metrics["vpr"] is not None:
            metrics["vpr"] = False
    metrics.update(
        method=method,
        run_id=task.run_id,
        method_failure=failure,
        generation_error=raw.error,
        finish_reason=raw.finish_reason,
        elapsed_seconds=budget.elapsed(),
        context_preparation_included=True,
        calibration_candidate=int(gold.case_id.split("-")[1]) <= 10,
    )
    for name, value in [
        ("task", task),
        ("context", context),
        ("raw-output", raw),
        ("translation", translation),
        ("validation", validation),
    ]:
        write_json(output / f"{name}.json", value.model_dump(mode="json"))
    (output / "raw-plan.txt").write_text(raw.content)
    write_json(output / "bindings.json", [b.model_dump(mode="json") for b in bindings])
    write_json(
        output / "published-plan.json",
        validation.published_plan.model_dump(mode="json") if validation.published_plan else None,
    )
    write_json(output / "metrics.json", collector.summary())
    write_json(output / "case-metrics.json", metrics)
    return metrics


def _blind_export(output, rows):
    directory = output / "review-package"
    directory.mkdir()
    mapping = []
    for row in rows:
        token = digest([row["run_id"], "blind-export"])[:20]
        original = output / "runs" / row["case_id"] / f"method-{METHODS.index(row['method']) + 1}"
        folder = directory / token
        folder.mkdir()
        shutil.copy2(original / "raw-plan.txt", folder / "raw-plan.txt")
        # Method and machine judgements omitted. Facts/reference remain explicitly unreviewed.
        shutil.copy2(original / "bindings.json", folder / "visible-evidence.json")
        shutil.copy2(
            output / "cases" / row["case_id"] / "case-reference.json",
            folder / "unreviewed-reference.json",
        )
        write_json(
            folder / "review-form.json",
            {
                "reviewer": None,
                "review_version": None,
                "raw_plan_validity": None,
                "goal_applicability": None,
                "notes": [],
                "reviewed": False,
            },
        )
        mapping.append(
            {
                "token": token,
                "case_id": row["case_id"],
                "method": row["method"],
                "run_id": row["run_id"],
            }
        )
    write_json(output / "review-key.private.json", mapping)
    write_json(
        directory / "instructions.json",
        {
            "synthetic": True,
            "reviewed": False,
            "reference_status": "draft synthetic reference, not expert Gold",
            "order": (
                "review raw text and frozen facts first; "
                "machine results available separately after review"
            ),
            "warning": (
                "Identifiers or style may reveal method; "
                "anonymized export does not mean completed blind review."
            ),
        },
    )


def run_phase2(
    project: Path, output: Path, cases=24, *, case_ids=None, client_factory=ObservedPlanClient
):
    project, output = project.resolve(), output.resolve()
    if project != Path(__file__).resolve().parents[3]:
        raise ValueError("project and imported source differ")
    if output.exists():
        raise ValueError("output exists; use a new experiment ID")
    if not 1 <= cases <= len(SPECS):
        raise ValueError("case count must be between 1 and 24")
    if case_ids is None:
        selected = list(range(1, cases + 1))
    else:
        if cases != 24:
            raise ValueError("case_ids and a custom case count cannot be combined")
        selected = list(case_ids)
        if (
            not selected
            or any(type(i) is not int or not 1 <= i <= len(SPECS) for i in selected)
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("case_ids must be unique integers between 1 and 24")
        selected.sort()
    cases = len(selected)
    output.mkdir(parents=True)
    before = source_manifest(project)
    write_json(output / "source_manifest.json", before)
    write_json(output / "common-head.json", generation_manifest())
    write_json(
        output / "protocol.json",
        {
            "schema_version": "v8-phase2-pilot-1",
            "cases": cases,
            "selected_case_ids": selected,
            "case_selection": "development subset" if case_ids is not None else "prefix",
            "methods": METHODS,
            "synthetic": True,
            "reviewed": False,
            "split": "dev",
            "new_heldout_consumed": False,
            "tool_adapter": CommonEvidenceTools.version,
            "common_head": HEAD_VERSION,
            "max_model_requests": 8,
            "max_tool_dispatches": 8,
            "max_rounds": 3,
            "timeout_seconds": 180,
            "head_max_tokens": HEAD_MAX_TOKENS,
            "temperature": 0,
            "seed": None,
            "vpr_claim": False,
            "independent_fault_count_claim": False,
            "p5_scope_adapter": (
                "P5 device_ids=[]; shared backend enforces multi-object PermissionScope"
            ),
            "direct_context_preparation": "public tools charged and recorded",
            "method_order": "fixed pilot order; not suitable for latency superiority claims",
        },
    )
    # Freeze all reference inputs before the first model output.
    prepared = [prepare_case(output / "cases" / f"pilot-{i:02d}", i - 1) for i in selected]
    validate_splits([gold for _, gold in prepared])
    write_json(
        output / "calibration-status.json",
        {
            "candidate_cases": [
                gold.case_id for _, gold in prepared if int(gold.case_id.split("-")[1]) <= 10
            ],
            "reviews": calibration_report([]),
            "reviewed": False,
            "status": "awaiting independent review records",
        },
    )
    rows = []
    for task, gold in prepared:
        for method in METHODS:
            folder = output / "runs" / gold.case_id / f"method-{METHODS.index(method) + 1}"
            folder.mkdir(parents=True)
            row = run_method(
                project, output / "cases" / gold.case_id, folder, task, gold, method, client_factory
            )
            rows.append(row)
            append_jsonl(output / "metrics-by-case.jsonl", row)
            print(
                json.dumps(
                    {
                        "case": gold.case_id,
                        "method": method,
                        "parsed": row["parsed"],
                        "validator_pass": row["validator_pass"],
                        "failure": row["method_failure"] or row["generation_error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    _blind_export(output, rows)
    flat = [{k: v for k, v in row.items() if k not in {"branch", "issues"}} for row in rows]
    with (output / "metrics-by-case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    failures = [
        row
        for row in rows
        if not row["reference_pass"] or row["method_failure"] or row["generation_error"]
    ]
    write_json(output / "failures.json", failures)
    unchanged = digest(before) == digest(source_manifest(project))
    result = {
        "schema_version": "v8-phase2-pilot-1",
        "cases": cases,
        "method_runs": len(rows),
        "synthetic": True,
        "reviewed": False,
        "new_heldout_consumed": False,
        "real_network_writes": 0,
        "source_unchanged": unchanged,
        "summaries": {m: summarize_cases([r for r in rows if r["method"] == m]) for m in METHODS},
        "method_failures": sum(bool(r["method_failure"]) for r in rows),
        "generation_errors": sum(bool(r["generation_error"]) for r in rows),
        "truncated_outputs": sum(r["finish_reason"] == "length" for r in rows),
        "expert_review_pending": True,
        "scale_decision": (
            "defer 60/150 until protocol failures repaired and independent review calibrated"
        ),
        "execution_pass": unchanged
        and len(rows) == cases * 4
        and not any(r["method_failure"] or r["generation_error"] for r in rows),
        "quality_pass": False,
        "development_quality_pass": unchanged
        and len(rows) == cases * 4
        and all(
            r["parsed"]
            and r["validator_pass"]
            and r["reference_pass"]
            and not r["method_failure"]
            and not r["generation_error"]
            and r["finish_reason"] != "length"
            for r in rows
        ),
        "development_quality_scope": (
            "shared constrained head on unreviewed synthetic drafts; "
            "all methods and all selected cases, not expert VPR"
        ),
        "limitations": [
            "synthetic development fixture families, not independent real faults",
            "no expert Gold, no VPR superiority claim",
            "pilot JSON translation only; ordinary text preserved as parse failure",
            "fixed method order; no performance superiority claim",
        ],
    }
    write_json(output / "summary.json", result)
    paths = [p for p in output.rglob("*") if p.is_file() and p.name != "sha256_manifest.json"]
    write_json(
        output / "sha256_manifest.json",
        {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    )
    return result
