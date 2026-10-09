"""LangGraph orchestration with transactional receipts and scope-preserving recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from upf_agent.config import load_model_config
from upf_agent.models import LocalChatClient
from upf_agent.v8.adapters import collect, verify_binding, verify_path
from upf_agent.v8.builders import load_manual_graph, task_graph
from upf_agent.v8.compiler import (
    compile_contract,
    compile_gap_plan,
    compile_query_plan,
    object_token,
    permitted,
)
from upf_agent.v8.dependency import LocalEvaluator
from upf_agent.v8.evaluator import route_gaps
from upf_agent.v8.graph import Neo4jGraphStore, compile_graph_query
from upf_agent.v8.models import EvidenceBinding, OrchestrationData, OrchestrationTask, digest
from upf_agent.v8.store import BudgetStop, Ledger, execution_fingerprint


class Flow(TypedDict):
    payload: dict


class OrchestrationEngine:
    def __init__(self, root: Path):
        self.root = root.resolve()
        original = Path("/home/huangxiaolin/upf-agent").resolve()
        if self.root == original or self.root.is_relative_to(original):
            raise ValueError("v8 artifacts cannot be written under the original project")
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        self.ledger = Ledger(self.root)
        self.evaluators = {}
        self.active_started = perf_counter()
        self.base_elapsed = 0

    def remaining(self, task):
        return (
            task.budget.timeout_seconds - self.base_elapsed - (perf_counter() - self.active_started)
        )

    def close(self):
        self.ledger.close()

    def _compile(self, state: Flow):
        data = OrchestrationData.model_validate(state["payload"])
        data.contract = compile_contract(data.task)
        if data.task.template in {"upf_n3_fault", "upf_n4_fault"} and not verify_path(
            self.root, data.task
        ):
            data.contract.gaps.append("business_path_provenance_unverified")
        data.query_plan = compile_query_plan(data.task, data.contract)
        if data.task.model_enabled:
            request = {
                "task": data.task.question,
                "scope": data.task.objects,
                "purpose": "optional_intent",
            }
            key = digest(request)
            started = perf_counter()
            try:
                prior = self.ledger.charge(data.task, key, "model", request)
                if prior is None:
                    config = load_model_config(self.root / "config/models.yaml")
                    with LocalChatClient(config.generation[data.task.model_profile]) as client:
                        client.client.timeout = max(0.001, self.remaining(data.task))
                        result = client.chat(
                            [
                                {
                                    "role": "system",
                                    "content": "只读诊断意图说明。不要建议执行，不得扩大对象范围。",
                                },
                                {
                                    "role": "user",
                                    "content": json.dumps(request, ensure_ascii=False),
                                },
                            ],
                            max_tokens=128,
                        )
                    self.ledger.commit_receipt(data.task, key, result, [], perf_counter() - started)
            except Exception as error:
                if not isinstance(error, BudgetStop):
                    self.ledger.note_error(data.task.run_id, key, error)
                data.stop_reason = f"model_unavailable:{type(error).__name__}"
        data.status = "compiled"
        return {"payload": data.model_dump(mode="json")}

    def _pause(self, state: Flow):
        data = OrchestrationData.model_validate(state["payload"])
        if data.task.interrupt_before_collect:
            answer = interrupt(
                {
                    "run_id": data.task.run_id,
                    "scope_fingerprint": digest(data.task),
                    "message": "已编译只读取证计划；恢复不会增加对象或权限。",
                }
            )
            if answer != {"scope_fingerprint": digest(data.task)}:
                raise ValueError("resume scope fingerprint mismatch")
        return state

    def _collect(self, state: Flow):
        data = OrchestrationData.model_validate(state["payload"])
        if data.stop_reason:
            return state
        for step in data.query_plan.steps:
            request = step.model_dump(mode="json")
            key = digest({"contract": data.contract.contract_id, "step": request})
            started = perf_counter()
            try:
                if self.remaining(data.task) <= 0:
                    raise BudgetStop("task_timeout")
                result = self.ledger.charge(data.task, key, "tool", request)
                if result is None:
                    try:
                        result = collect(
                            self.root,
                            data.task,
                            data.contract,
                            step,
                            key,
                            self.remaining(data.task),
                        )
                    except Exception as error:
                        result = {
                            "bindings": [],
                            "gaps": [f"tool_failed:{type(error).__name__}:{error}"],
                        }
                    bindings = [EvidenceBinding.model_validate(b) for b in result["bindings"]]
                    self.ledger.commit_receipt(
                        data.task, key, result, bindings, perf_counter() - started
                    )
                data.query_plan.gaps.extend(
                    g for g in result.get("gaps", []) if g not in data.query_plan.gaps
                )
            except BudgetStop as error:
                data.stop_reason = str(error)
                break
        data.bindings = self.ledger.evidence(data.task.run_id)
        for binding in data.bindings:
            binding.source_verified = verify_binding(self.root, binding)
        data.status = "collected"
        return {"payload": data.model_dump(mode="json")}

    def _graph(self, state: Flow):
        data = OrchestrationData.model_validate(state["payload"])
        if not data.task.graph_enabled:
            return state
        store = None
        started = perf_counter()
        request = {
            "graph_context": digest({"task": data.task, "contract": data.contract}),
            "bindings": digest(data.bindings),
            "round": data.rounds,
        }
        key = digest(request)
        try:
            previous = self.ledger.charge(data.task, key, "tool", request)
            if previous is not None:
                data.bindings = self.ledger.evidence(data.task.run_id)
                return {"payload": data.model_dump(mode="json")}
            base = load_manual_graph(self.root)
            if data.task.graph_backend == "neo4j":
                config = json.loads((self.root / "config/v8-neo4j.json").read_text())
                store = Neo4jGraphStore(**config, namespace=f"run-{data.task.run_id}")
                snapshot = base.snapshot()
                from upf_agent.v8.graph import GraphEdge, GraphNode

                store.build(
                    [GraphNode.model_validate(n) for n in snapshot["nodes"]],
                    [GraphEdge.model_validate(e) for e in snapshot["edges"]],
                )
            else:
                store = base
            nodes, edges = task_graph(data.task, data.bindings)
            store.build(nodes, edges)
            seed_ids = [
                b.native_id
                if b.category == "K" and b.native_id in base.nodes
                else f"binding:{b.evidence_id}:{b.revision}"
                for b in data.bindings
            ]
            query = compile_graph_query(data.task, data.contract.contract_id, seed_ids)
            result = store.query(query)
            folder = self.root / "runs/v8" / data.task.run_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"subgraph-{data.rounds}.json").write_text(
                result.model_dump_json(indent=2) + "\n"
            )
            (folder / "graph-query.json").write_text(query.model_dump_json(indent=2) + "\n")
            # Recheck applicability and provenance after graph retrieval.
            # Only applicable, sourced blocks can enter the ledger.
            new = []
            for node in result.nodes:
                if node.kind not in {"DocumentBlock", "Command"} or not node.attributes.get("text"):
                    continue
                if node.attributes.get("product") != data.task.product:
                    continue
                for req in data.contract.requirements:
                    if req.kind != "K":
                        continue
                    for obj in req.objects:
                        if not permitted(data.task, obj, req.fields):
                            continue
                        if node.object_ids and obj not in node.object_ids:
                            continue
                        identifier = f"{data.task.run_id}:graph:{node.node_id}:{obj}"
                        if any(b.evidence_id == identifier for b in data.bindings):
                            continue
                        b = EvidenceBinding(
                            evidence_id=identifier,
                            native_id=node.node_id,
                            task_id=data.task.run_id,
                            contract_id=data.contract.contract_id,
                            requirement_id=req.requirement_id,
                            object_id=obj,
                            category="K",
                            fields={"instruction": node.attributes["text"]},
                            source=node.source,
                            product=data.task.product,
                            version=node.version,
                            receipt_id=key,
                            synthetic=node.synthetic,
                        )
                        from upf_agent.v8.knowledge_applicability import annotate_knowledge

                        annotate_knowledge(self.root, b)
                        new.append(b)
            # Graph queries share the same total tool budget and use a committed receipt.
            if previous is None:
                self.ledger.commit_receipt(
                    data.task,
                    key,
                    {
                        "subgraph_sha256": digest(result),
                        "bindings": [b.model_dump(mode="json") for b in new],
                    },
                    new,
                    perf_counter() - started,
                )
            data.bindings = self.ledger.evidence(data.task.run_id)
        except BudgetStop as error:
            data.stop_reason = str(error)
        except Exception as error:
            self.ledger.note_error(data.task.run_id, key, error)
            data.stop_reason = f"graph_unavailable:{type(error).__name__}:{error}"
        finally:
            if isinstance(store, Neo4jGraphStore):
                store.close()
        return {"payload": data.model_dump(mode="json")}

    def _evaluate(self, state: Flow):
        data = OrchestrationData.model_validate(state["payload"])
        if self.remaining(data.task) <= 0:
            data.stop_reason = "task_timeout"
        for binding in data.bindings:
            binding.source_verified = verify_binding(self.root, binding)
        data.batch_id = digest(
            {
                "task": data.task,
                "contract": data.contract,
                "bindings": data.bindings,
                "round": data.rounds,
            }
        )[:24]
        evaluator = self.evaluators.get(data.task.run_id)
        if evaluator is None:
            evaluator = LocalEvaluator(data.task, data.contract, data.bindings, data.batch_id)
            self.evaluators[data.task.run_id] = evaluator
            update = {
                "mode": "initial_full",
                "recomputed_states": len(evaluator.states),
                "recomputed_actions": len(evaluator.actions),
                "read_audit": evaluator.initial_audit,
                "evaluation_context": evaluator.context.model_dump(mode="json"),
            }
        else:
            update = evaluator.update(data.bindings, data.batch_id)
        data.states = evaluator.states
        data.path_vector = [
            {
                "position": i,
                "object_id": p.object_id,
                "domain": p.domain,
                "conditions": {
                    kind: data.states[f"{kind}:{object_token(p.object_id)}"].code.value
                    for kind in ("K", "R", "C", "A")
                    if f"{kind}:{object_token(p.object_id)}" in data.states
                },
                "pair_states": {
                    r.requirement_id: data.states[r.requirement_id].code.value
                    for r in data.contract.requirements
                    if r.kind == "P" and p.object_id in r.objects
                },
            }
            for i, p in enumerate(data.task.path.positions)
        ]
        data.actions = list(evaluator.actions.values())
        data.gaps = route_gaps(data.contract, data.states)
        if data.stop_reason:
            for decision in data.actions:
                if decision.outcome == "suggest_allowed":
                    decision.outcome = "human_review"
                decision.reasons.append(data.stop_reason)
        data.status = "evaluated"
        self.ledger.save(
            data,
            dependencies={
                "edges": list(evaluator.index.graph.edges),
                "observed_read_edges": sorted(evaluator.index.actual_edges),
                "trusted": evaluator.index.audit(),
            },
            update=update,
        )
        return {"payload": data.model_dump(mode="json")}

    def _route(self, state: Flow):
        data = OrchestrationData.model_validate(state["payload"])
        if not data.gaps or data.stop_reason:
            return "report"
        if data.rounds >= data.task.budget.max_rounds:
            return "report"
        # Only unattempted capabilities can close a gap. No blind repetition of unavailable tools.
        plan = compile_gap_plan(
            data.task, data.contract, data.gaps, self.ledger.attempted_tools(data.task.run_id)
        )
        return "supplement" if plan.steps else "report"

    def _supplement(self, state: Flow):
        data = OrchestrationData.model_validate(state["payload"])
        data.rounds += 1
        plan = compile_gap_plan(
            data.task, data.contract, data.gaps, self.ledger.attempted_tools(data.task.run_id)
        )
        data.query_plan = plan
        return {"payload": data.model_dump(mode="json")}

    def _report(self, state: Flow):
        data = OrchestrationData.model_validate(state["payload"])
        data.status = "completed" if not data.gaps and not data.stop_reason else "needs_evidence"
        if data.gaps and not data.stop_reason:
            data.stop_reason = (
                "no_available_evidence_capability"
                if data.rounds < data.task.budget.max_rounds
                else "round_budget_exhausted"
            )
        # Final report is separate from the immutable evaluation batch.
        with self.ledger.transaction():
            self.ledger.db.execute(
                "UPDATE runs SET elapsed=max(elapsed,?) WHERE run_id=?",
                (self.base_elapsed + perf_counter() - self.active_started, data.task.run_id),
            )
            self.ledger.db.execute(
                "UPDATE runs SET data=? WHERE run_id=?", (data.model_dump_json(), data.task.run_id)
            )
        folder = self.root / "runs/v8" / data.task.run_id
        folder.mkdir(parents=True, exist_ok=True)
        report = {
            **data.model_dump(mode="json"),
            "validation": "v8 read-only prototype",
            "budget_usage": self.ledger.inspect(data.task.run_id),
            "limitations": [
                "合成资料仅验证规则原型" if data.task.synthetic else "真实证据缺口保持原状",
                "只读建议不产生执行授权；历史 M6/M7 状态独立",
            ],
        }
        report["budget_usage"].pop("data", None)
        self.ledger.export_metrics(data.task.run_id, folder)
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        lines = [
            f"# UPF-Agent v8：{data.task.run_id}",
            "",
            data.task.question,
            "",
            f"状态：{data.status}；停止原因：{data.stop_reason or '无'}。",
            "",
            "| 需求 | 状态 | 原因 |",
            "|---|---|---|",
        ]
        lines += [
            f"| {s.requirement_id} | {s.code.value} | {'; '.join(s.reasons)} |"
            for s in data.states.values()
        ]
        lines += ["", "只读动作：", ""] + [
            f"- {a.action_id}：{a.outcome}；{'; '.join(a.reasons)}" for a in data.actions
        ]
        lines += ["", "证据来源：", ""] + [
            f"- {b.evidence_id} → {b.source.relative_path} {b.source.locator} "
            f"SHA256={b.source.sha256}"
            for b in data.bindings
        ]
        (folder / "report.md").write_text("\n".join(lines) + "\n")
        return {"payload": data.model_dump(mode="json")}

    def _build(self, saver):
        graph = StateGraph(Flow)
        for name in ("compile", "pause", "collect", "graph", "evaluate", "supplement", "report"):
            graph.add_node(name, getattr(self, f"_{name}"))
        graph.add_edge(START, "compile")
        graph.add_edge("compile", "pause")
        graph.add_edge("pause", "collect")
        graph.add_conditional_edges(
            "collect",
            lambda state: (
                "graph" if state["payload"]["task"].get("graph_enabled", True) else "evaluate"
            ),
            {"graph": "graph", "evaluate": "evaluate"},
        )
        graph.add_edge("graph", "evaluate")
        graph.add_conditional_edges(
            "evaluate", self._route, {"supplement": "supplement", "report": "report"}
        )
        graph.add_edge("supplement", "collect")
        graph.add_edge("report", END)
        return graph.compile(checkpointer=saver)

    def run(self, task: OrchestrationTask) -> dict:
        with self.ledger.execution_lock(task.run_id):
            return self._run(task)

    def _run(self, task: OrchestrationTask) -> dict:
        fingerprint = execution_fingerprint(self.root)
        if task.execution_fingerprint and task.execution_fingerprint != fingerprint:
            raise ValueError("code or locked configuration changed; restore the run snapshot")
        task.execution_fingerprint = fingerprint
        self.ledger.create(task)
        self.active_started = perf_counter()
        self.base_elapsed = self.ledger.inspect(task.run_id)["elapsed_seconds"]
        checkpoint = self.root / "data/v8/checkpoints.sqlite3"
        with SqliteSaver.from_conn_string(str(checkpoint)) as saver:
            graph = self._build(saver)
            cfg = {"configurable": {"thread_id": task.run_id}, "recursion_limit": 40}
            previous = graph.get_state(cfg)
            if previous.values:
                return self.ledger.inspect(task.run_id)
            result = graph.invoke(
                {"payload": OrchestrationData(task=task).model_dump(mode="json")}, cfg
            )
        if "__interrupt__" in result:
            return {
                "status": "interrupted",
                "run_id": task.run_id,
                "interrupt": [i.value for i in result["__interrupt__"]],
            }
        return result["payload"]

    def resume(self, run_id: str) -> dict:
        with self.ledger.execution_lock(run_id):
            return self._resume(run_id)

    def _resume(self, run_id: str) -> dict:
        task = self.ledger.task(run_id)
        self.active_started = perf_counter()
        self.base_elapsed = self.ledger.inspect(run_id)["elapsed_seconds"]
        if task.execution_fingerprint != execution_fingerprint(self.root):
            raise ValueError("code or locked configuration changed; restore the run snapshot")
        with SqliteSaver.from_conn_string(str(self.root / "data/v8/checkpoints.sqlite3")) as saver:
            graph = self._build(saver)
            cfg = {"configurable": {"thread_id": run_id}, "recursion_limit": 40}
            state = graph.get_state(cfg)
            if not state.values:
                raise ValueError("no checkpoint for this run")
            if not state.next:
                return state.values["payload"]
            interrupted = any(t.interrupts for t in state.tasks)
            value = Command(resume={"scope_fingerprint": digest(task)}) if interrupted else None
            return graph.invoke(value, cfg)["payload"]
