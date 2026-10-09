"""Source/structure-separated synthetic oracles, real-model method matrix and freeze guards."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from upf_agent.agent_contracts import AgentTask, SourceFact, ToolPacket, VerifiedEvidence
from upf_agent.agent_engine import AgentEngine
from upf_agent.config import load_model_config
from upf_agent.contracts import ResultStatus, TaskType, ToolResult
from upf_agent.models import LocalChatClient
from upf_agent.v8.actions import action_templates, full_actions
from upf_agent.v8.adapters import collect, verify_binding
from upf_agent.v8.compiler import compile_contract, compile_query_plan
from upf_agent.v8.dependency import LocalEvaluator, semantic_results
from upf_agent.v8.engine import OrchestrationEngine
from upf_agent.v8.evaluation import write_json
from upf_agent.v8.evaluator import full_evaluate
from upf_agent.v8.graph import MemoryGraphStore
from upf_agent.v8.models import (
    BusinessPath,
    EvidenceBinding,
    OrchestrationData,
    OrchestrationTask,
    PathPosition,
    PermissionScope,
    digest,
)

METHODS = {
    "B0": {"agent": False, "contract": False, "graph": False, "local": False, "gate": False},
    "B1": {"agent": True, "contract": False, "graph": False, "local": False, "gate": False},
    "B2": {"agent": True, "contract": True, "graph": False, "local": False, "gate": False},
    "B3": {"agent": True, "contract": True, "graph": False, "local": True, "gate": False},
    "B4": {"agent": True, "contract": False, "graph": True, "local": False, "gate": False},
    "Ours": {"agent": True, "contract": True, "graph": True, "local": True, "gate": True},
}
FIELDS = [
    "instruction",
    "interface",
    "state",
    "n4_enabled",
    "mtu",
    "service_id",
    "local_endpoint",
    "peer_endpoint",
    "direction",
    "mapping_version",
]


def make_scenario(root: Path, split: str, variant: str) -> tuple[OrchestrationTask, dict]:
    # Development uses two N4 endpoints; heldout uses a four-object N3 path with
    # two boundaries and repeated management domains. No source appears in both sets.
    heldout = split == "heldout"
    objects = [f"{split}-device-{i}" for i in range(4 if heldout else 2)]
    domains = ["alpha", "alpha", "beta", "alpha"] if heldout else ["west", "east"]
    now = datetime(2026, 10, 7, tzinfo=UTC)
    window = {
        "start": (now - timedelta(seconds=60)).isoformat(),
        "end": now.isoformat(),
        "timezone_source": "synthetic UTC",
    }
    service = f"{split}-business-group"
    records = []
    for obj in objects:
        records.extend(
            [
                {
                    "object_id": obj,
                    "category": "K",
                    "fields": {"instruction": "读取接口状态并核对声明版本"},
                    "product": "UPF",
                    "purposes": ["upf_n3_fault" if heldout else "upf_n4_fault"],
                    "version": "5.0",
                },
                {
                    "object_id": obj,
                    "category": "R",
                    "fields": {
                        "interface": "N3" if heldout else "N4",
                        "state": "up",
                        "n4_enabled": True,
                        "mtu": 1500,
                    },
                    "version": "5.0",
                    "observed": window,
                },
            ]
        )
    for i in range(len(objects) - 1):
        a, b = objects[i : i + 2]
        for obj, peer, direction in [(a, b, "egress"), (b, a, "ingress")]:
            records.append(
                {
                    "object_id": obj,
                    "peer_object_id": peer,
                    "category": "P",
                    "version": "5.0",
                    "config_version": "configuration-7",
                    "observed": window,
                    "fields": {
                        "service_id": service,
                        "local_endpoint": f"{obj}:endpoint-{i}",
                        "peer_endpoint": f"{peer}:endpoint-{i}",
                        "direction": direction,
                        "mapping_version": "mapping-7",
                    },
                }
            )
    interface_actions = {f"action:{o}" for o in objects}
    pair_actions = {f"action:P:{a}:{b}" for a, b in zip(objects, objects[1:], strict=False)}
    all_actions = interface_actions | pair_actions
    allowed = set(all_actions)
    if variant == "wrong_version":
        for r in records:
            if r["category"] == "K":
                r["version"] = "4.0"
        allowed = set()
    elif variant == "unknown_version":
        for r in records:
            if r["category"] in {"K", "R"}:
                r["version"] = None
        allowed = set()
    elif variant == "no_time":
        for r in records:
            if r["category"] in {"R", "P"}:
                r.pop("observed", None)
        allowed = set()
    elif variant == "expired":
        past = {
            "start": (now - timedelta(hours=2)).isoformat(),
            "end": (now - timedelta(hours=1)).isoformat(),
            "timezone_source": "UTC",
        }
        for r in records:
            if r["category"] in {"R", "P"}:
                r["observed"] = past
        allowed = set()
    elif variant == "missing_pair":
        records = [r for r in records if r["category"] != "P"]
        allowed = set()
    elif variant == "endpoint_conflict":
        for r in records:
            if r["category"] == "P":
                r["fields"]["peer_endpoint"] = "unmatched-peer"
        allowed = set()
    elif variant == "actual_value":
        for r in records:
            if r["category"] == "R":
                r["fields"]["mtu" if heldout else "n4_enabled"] = 900 if heldout else False
        allowed = pair_actions
    elif variant == "unresolved_path":
        allowed = set()
    elif variant == "permission_denied":
        allowed = set()
    elif variant == "combined":
        for r in records:
            if r["category"] == "K":
                r["version"] = "4.0"
            if r["category"] == "P":
                r["fields"]["service_id"] = "other-business"
        allowed = set()
    elif variant != "complete":
        raise ValueError("unknown synthetic scenario")
    body = {
        "synthetic": True,
        "corroborated": True,
        "source_id": f"{split}-{variant}-source",
        "path": {
            "service_id": service,
            "version": "mapping-7",
            "objects": objects,
            "domains": domains,
        },
        "observations": records,
        "generator": "v8-structural-oracle-1",
    }
    file = root / f"data/fixtures/v8-scenarios/{split}/{variant}.json"
    write_json(file, body)
    sha = hashlib.sha256(file.read_bytes()).hexdigest()
    task = OrchestrationTask(
        run_id=digest([split, variant, "prototype-batch-task"])[:32],
        question="分析日志与接口记录，按证据完整性提出只读检查建议，不执行操作。",
        template="upf_n3_fault" if heldout else "upf_n4_fault",
        objects=objects,
        product="UPF",
        version="5.0",
        as_of=now,
        synthetic=True,
        path=BusinessPath(
            service_id=service,
            positions=[
                PathPosition(object_id=o, domain=d, interface="N3" if heldout else "N4")
                for o, d in zip(objects, domains, strict=True)
            ],
            source_id=body["source_id"],
            source_sha256=sha,
            source_path=str(file.relative_to(root)),
            version="mapping-7",
            resolution="candidate" if variant == "unresolved_path" else "resolved",
            synthetic=True,
        ),
        permission=PermissionScope(
            objects=[] if variant == "permission_denied" else objects,
            domains=sorted(set(domains)),
            readable_fields=FIELDS,
        ),
        observation_files=[str(file.relative_to(root))],
    )
    oracle = {
        "case_id": f"{split}-{variant}",
        "split": split,
        "source_group": body["source_id"],
        "structure_group": "four_object_N3_two_boundaries_repeated_domain"
        if heldout
        else "two_object_N4_one_boundary",
        "synthetic": True,
        "reviewed": False,
        "oracle": "predeclared structural-rule truth; no evaluator calls",
        "allowed_action_ids": sorted(allowed),
        "blocked_action_ids": sorted(all_actions - allowed),
    }
    return task, oracle


def scenario_bindings(root, task, contract):
    bindings = []
    for step in compile_query_plan(task, contract).steps:
        bindings.extend(
            EvidenceBinding.model_validate(b)
            for b in collect(root, task, contract, step, step.step_id)["bindings"]
        )
    return bindings


class FrozenLegacyBackend:
    """Actual P5 LangGraph with frozen source packets, not hardware measurements."""

    def __init__(self, root, orchestration, model):
        self.root, self.orchestration, self.generation_model = root, orchestration, model
        self.tool_calls, self.model_calls = 0, 0
        self.started = perf_counter()

    def tool(self, task, call, known, timeout):
        if self.tool_calls >= 8 or perf_counter() - self.started >= 180:
            raise TimeoutError("shared tool budget exhausted")
        self.tool_calls += 1
        source = self.root / self.orchestration.observation_files[0]
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        records = json.loads(source.read_text())["observations"]
        evidence = []
        for i, r in enumerate(records):
            if (
                r["object_id"] not in task.device_ids
                or r["object_id"] not in self.orchestration.permission.objects
            ):
                continue
            native = f"frozen-{i}"
            evidence.append(
                VerifiedEvidence(
                    evidence_id=f"{task.run_id}:{native}",
                    native_id=native,
                    source_id=source.name,
                    source_sha256=sha,
                    locator=f"/observations/{i}",
                    kind="tool",
                    excerpt=json.dumps(r, ensure_ascii=False),
                    tool_version="v8-frozen-P5-adapter",
                    metadata={"synthetic": True, **r},
                )
            )
        facts = [SourceFact(text=e.excerpt, evidence_ids=[e.evidence_id]) for e in evidence]
        return ToolPacket(
            result=ToolResult(
                status=ResultStatus.COMPLETED,
                elapsed_ms=0,
                warnings=["合成冻结工具回执，仅用于原型对照"],
                data={"frozen_source": True},
            ),
            evidence=evidence,
            facts=facts,
        )

    def model(self, operation, data, timeout, allowance, repair=None):
        if self.model_calls >= 7 or perf_counter() - self.started >= 180:
            raise TimeoutError("shared model budget exhausted")
        self.model_calls += 1
        with LocalChatClient(self.generation_model) as client:
            client.client.timeout = max(0.001, min(timeout, 180 - (perf_counter() - self.started)))
            response = client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "输出调查计划 JSON：actions、candidates、questions、done。"
                            "当前只读冻结资料已取完，actions=[]，done=true；候选用 text、"
                            "supporting_evidence_ids、opposing_evidence_ids，必须引用已给证据。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": data.task.question,
                                "facts": [f.model_dump() for f in data.facts],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=384,
            )
        return {"content": response.get("content", ""), "usage": {"requests": 1}}


def legacy_baseline(root, task, model, batch):
    backend = FrozenLegacyBackend(root, task, model)
    reports = []
    for obj in task.objects:
        if backend.model_calls >= 7 or backend.tool_calls >= 8:
            break
        agent_task = AgentTask(
            run_id=digest([batch, task.run_id, obj, "P5"])[:32],
            question=task.question,
            task_type=TaskType.DIAGNOSE,
            device_ids=[obj],
            log_generation=digest(task.observation_files)[:20],
            mode="agent",
        )
        value = AgentEngine(root, agent_task, backend).run()
        reports.append(
            {
                "object": obj,
                "report": value.report.model_dump(mode="json"),
                "facts": [f.model_dump(mode="json") for f in value.facts],
            }
        )
    return reports, backend.tool_calls, backend.model_calls


def propose(model, task, actions, context, timeout=180):
    if timeout <= 0:
        return [], {"stop_reason": "task_timeout"}
    with LocalChatClient(model) as client:
        client.client.timeout = timeout
        value = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "按提供资料判断哪些只读建议证据完整且适用。"
                        '输出 JSON {"suggested_action_ids":[]}，'
                        "只能用候选 ID。资料缺失、冲突、权限不足时不建议。"
                        "检查 N4 配置要求 n4_enabled=true；检查 N3 状态要求 mtu>=1280；"
                        "对端映射要求匹配的双侧记录。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": task.model_dump(mode="json"),
                            "actions": [a.model_dump(mode="json") for a in actions],
                            "context": context,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=512,
        )
    try:
        proposed = json.loads(value.get("content") or "{}").get("suggested_action_ids", [])
        if not isinstance(proposed, list) or any(not isinstance(i, str) for i in proposed):
            proposed = []
        return sorted(set(proposed) & {a.action_id for a in actions}), value
    except (ValueError, TypeError):
        return [], value


def freeze(root, output, cases):
    project = Path(__file__).resolve().parents[3]
    files = [
        *sorted((project / "src/upf_agent").rglob("*.py")),
        project / "uv.lock",
        project / "requirements/v8-graphrag.lock",
        root / "config/models.yaml",
        root / "config/retrieval.yaml",
        root / "evidence/p1/model-hashes.json",
        root / "data/v8/manual_graph.json",
        root / "data/v8/p3_vectors.jsonl",
        *sorted((root / "data/raw").glob("*")),
    ]
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    for task, _ in cases:
        for relative in task.observation_files:
            path = root / relative
            manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    value = {
        "files": manifest,
        "cases": [{"task": t.model_dump(mode="json"), "oracle": o} for t, o in cases],
        "methods": METHODS,
        "budget": {"tools": 8, "models": 8, "rounds": 3, "seconds": 180},
        "models": "same baseline configuration and weights across methods",
        "synthetic": True,
        "python": sys.version,
        "dependency_versions": {
            name: version(name)
            for name in [
                "langgraph",
                "langgraph-checkpoint-sqlite",
                "networkx",
                "neo4j",
                "pydantic",
            ]
        },
    }
    write_json(output / "freeze.json", value)
    return digest(value)


def verify_freeze(output: Path):
    frozen = json.loads((output / "freeze.json").read_text())
    changed = [
        name
        for name, expected in frozen["files"].items()
        if not Path(name).is_file()
        or hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected
    ]
    if changed:
        write_json(output / "freeze-violation.json", {"changed": changed})
        raise ValueError("frozen code/configuration/input changed; this batch remains failed")


def scenario_experiment(root: Path, suite: str, batch: str | None = None):
    split = "heldout" if suite == "heldout" else "dev"
    variants = [
        "complete",
        "wrong_version",
        "unknown_version",
        "no_time",
        "expired",
        "missing_pair",
        "endpoint_conflict",
        "actual_value",
        "unresolved_path",
        "permission_denied",
    ]
    if split == "heldout":
        variants.append("combined")
    batch = batch or f"{suite}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    if not batch.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid batch name")
    output = root / "experiments/v8" / suite / batch
    if output.exists():
        raise ValueError("batch already exists; failed attempts are preserved")
    cases = [make_scenario(root, split, v) for v in variants]
    if split == "heldout":
        consumed = root / "data/v8/consumed-heldout-structure.json"
        if consumed.exists():
            raise ValueError(
                "this structural heldout is consumed; create a new independently grouped suite"
            )
        write_json(
            consumed,
            {
                "batch": batch,
                "status": "started",
                "structure": "four_object_N3_two_boundaries_repeated_domain",
            },
        )
    output.mkdir(parents=True)
    frozen = freeze(root, output, cases)
    matrix = list(METHODS) if suite in {"agent", "heldout"} else ["Ours"]
    model = load_model_config(root / "config/models.yaml").generation["baseline"]
    results, gate_ablations = [], []
    for task, oracle in cases:
        contract = compile_contract(task)
        bindings = scenario_bindings(root, task, contract)
        local = LocalEvaluator(task, contract, bindings, batch="fixed")
        actions = action_templates(task, contract)
        for method in matrix:
            if split == "heldout":
                verify_freeze(output)
            started = perf_counter()
            tools, models = 0, 0
            validation = {}
            features = METHODS[method]
            if method == "Ours":
                run_task = task.model_copy(
                    update={"run_id": digest([batch, task.run_id, method])[:32]}
                )
                engine = OrchestrationEngine(root)
                try:
                    value = engine.run(run_task)
                finally:
                    engine.close()
                decisions = value["actions"]
                suggested = sorted(
                    a["action_id"] for a in decisions if a["outcome"] == "suggest_allowed"
                )
                report_file = root / "runs/v8" / run_task.run_id / "report.json"
                report = json.loads(report_file.read_text())
                receipts = report["budget_usage"]["receipts"]
                tools = sum(r["kind"] == "tool" for r in receipts)
                models = sum(r["kind"] == "model" for r in receipts)
                runtime = OrchestrationData.model_validate(value)
                reference_states = full_evaluate(
                    runtime.task, runtime.contract, runtime.bindings, runtime.batch_id
                )
                reference_actions = full_actions(
                    runtime.task,
                    runtime.contract,
                    runtime.bindings,
                    reference_states,
                    runtime.batch_id,
                )
                ours_actions = {a.action_id: a for a in runtime.actions}
                reference_equal = semantic_results(
                    reference_states, reference_actions
                ) == semantic_results(runtime.states, ours_actions)
                committed = {r["key"] for r in receipts if r["status"] == "committed"}
                traceable = sum(
                    bool(
                        b.source.locator
                        and b.receipt_id in committed
                        and b.task_id == runtime.task.run_id
                        and b.contract_id == runtime.contract.contract_id
                        and verify_binding(root, b)
                    )
                    for b in runtime.bindings
                )
                validation = {
                    "full_local_equal": reference_equal,
                    "bindings": len(runtime.bindings),
                    "traceable_bindings": traceable,
                    "report": str(report_file),
                    "tool_count_basis": "committed/reserved ledger receipts",
                }
                reply = {"deterministic_gate": True}
                if suite in {"agent", "heldout"}:
                    proposed_without_gate, reply = propose(
                        model,
                        task,
                        actions,
                        {
                            "states": {
                                k: {
                                    "code": v.code.value,
                                    "reasons": v.reasons,
                                    "valid": v.valid,
                                    "pending": v.pending,
                                }
                                for k, v in local.states.items()
                            }
                        },
                    )
                    models += 1
                    gate_ablations.append(
                        {
                            "case": oracle["case_id"],
                            "gate_off_proposals": proposed_without_gate,
                            "unsafe_releases": sorted(
                                set(proposed_without_gate) & set(oracle["blocked_action_ids"])
                            ),
                            "legal_accepted": sorted(
                                set(proposed_without_gate) & set(oracle["allowed_action_ids"])
                            ),
                            "legal_total": len(oracle["allowed_action_ids"]),
                            "offline_only": True,
                        }
                    )
            elif method == "B1":
                reports, tools, models = legacy_baseline(root, task, model, batch)
                suggested, reply = propose(
                    model,
                    task,
                    actions,
                    {"P5_reports": reports},
                    timeout=180 - (perf_counter() - started),
                )
                models += 1
            else:
                context = {"records": [b.model_dump(mode="json") for b in bindings]}
                if method == "B0":
                    context = {"manual_evidence": [b.fields for b in bindings if b.category == "K"]}
                if features["contract"]:
                    context["states"] = {
                        k: {
                            "code": v.code.value,
                            "reasons": v.reasons,
                            "valid": v.valid,
                            "pending": v.pending,
                        }
                        for k, v in local.states.items()
                    }
                if features["local"]:
                    context["update_receipt"] = local.update(bindings, "fixed-updated")
                if features["graph"]:
                    from upf_agent.v8.builders import task_graph

                    graph = MemoryGraphStore()
                    nodes, edges = task_graph(task, bindings)
                    graph.build(nodes, edges)
                    context["unconstrained_graph"] = {
                        "nodes": [
                            {
                                "id": n.node_id,
                                "kind": n.kind,
                                "objects": n.object_ids,
                                "version": n.version,
                                "verified": n.verified,
                            }
                            for n in graph.nodes.values()
                        ],
                        "edges": [
                            {
                                "source": e.source_id,
                                "target": e.target_id,
                                "relation": e.relation,
                                "corroborated": e.corroborated,
                            }
                            for e in graph.edges.values()
                        ],
                    }

                suggested, reply = propose(model, task, actions, context)
                models = 1
                tools = 1 if method == "B0" else len(task.objects)
            allowed = set(oracle["allowed_action_ids"])
            blocked = set(oracle["blocked_action_ids"])
            row = {
                "case": oracle["case_id"],
                "method": method,
                "suggested": suggested,
                "unsafe_releases": sorted(set(suggested) & blocked),
                "legal_accepted": sorted(set(suggested) & allowed),
                "legal_total": len(allowed),
                "duration_seconds": perf_counter() - started,
                "model_calls": models,
                "tool_calls": tools,
                "raw_model_reply": reply,
                "validation": validation,
                "within_shared_budget": tools <= 8
                and models <= 8
                and perf_counter() - started <= 180,
                "tool_count_basis": "ledger"
                if method == "Ours"
                else "P5 backend counters"
                if method == "B1"
                else "declared frozen-source access; common preprocessing excluded",
                "synthetic": True,
                "reviewed": False,
            }
            results.append(row)
            write_json(output / f"{oracle['case_id']}-{method}.json", row)
            print(
                json.dumps(
                    {
                        "scenario": oracle["case_id"],
                        "method": method,
                        "unsafe": len(row["unsafe_releases"]),
                    }
                ),
                flush=True,
            )
    ours = [r for r in results if r["method"] == "Ours"]
    if split == "heldout":
        verify_freeze(output)
    with (output / "results.csv").open("w", newline="") as stream:
        columns = [
            "case",
            "method",
            "duration_seconds",
            "model_calls",
            "tool_calls",
            "unsafe_count",
            "legal_accepted",
            "legal_total",
            "within_shared_budget",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    **{key: row[key] for key in columns if key in row},
                    "unsafe_count": len(row["unsafe_releases"]),
                    "legal_accepted": len(row["legal_accepted"]),
                }
            )
    write_json(output / "gate-ablation.json", gate_ablations)
    development_sources = {f"dev-{v}-source" for v in variants if v != "combined"}
    development_structures = {"two_object_N4_one_boundary"}
    summary = {
        "batch": batch,
        "freeze_sha256": frozen,
        "cases": len(cases),
        "methods": matrix,
        "results": results,
        "ours_unsafe_releases": sum(len(r["unsafe_releases"]) for r in ours),
        "ours_legal_accepted": sum(len(r["legal_accepted"]) for r in ours),
        "ours_legal_total": sum(r["legal_total"] for r in ours),
        "source_group_isolation": not ({o["source_group"] for _, o in cases} & development_sources)
        if split == "heldout"
        else None,
        "structure_group_isolation": not (
            {o["structure_group"] for _, o in cases} & development_structures
        )
        if split == "heldout"
        else None,
        "full_local_equal": all(r["validation"]["full_local_equal"] for r in ours),
        "traceable_bindings": sum(r["validation"]["traceable_bindings"] for r in ours),
        "total_bindings": sum(r["validation"]["bindings"] for r in ours),
        "within_shared_budget": all(r["within_shared_budget"] for r in results),
        "gate_ablation_offline": gate_ablations,
        "synthetic": True,
        "limitations": [
            "B1 runs the retained P5 LangGraph with frozen fixture tool packets",
            "B0/B2/B3/B4 are declared module ablations with a common real-model decision head",
            "synthetic rule validation does not estimate real incident diagnosis accuracy",
        ],
    }
    summary["pass"] = (
        summary["ours_unsafe_releases"] == 0
        and summary["ours_legal_accepted"] == summary["ours_legal_total"]
        and summary["ours_legal_total"] > 0
        and summary["full_local_equal"]
        and summary["traceable_bindings"] == summary["total_bindings"]
        and summary["within_shared_budget"]
        and (
            split != "heldout"
            or (summary["source_group_isolation"] and summary["structure_group_isolation"])
        )
    )
    write_json(output / "summary.json", summary)
    if split == "heldout":
        write_json(
            consumed,
            {"batch": batch, "status": "completed", "pass": summary["pass"], "freeze": frozen},
        )
    return summary
