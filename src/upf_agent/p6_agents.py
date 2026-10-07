"""Evidence/cost comparison; does not infer root-cause or business accuracy."""

import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from upf_agent.agent_contracts import AgentData, AgentTask, ToolCall
from upf_agent.agent_engine import run_agent
from upf_agent.agent_tools import execute_tool
from upf_agent.agent_worker import CountedClient
from upf_agent.p6_contracts import read_suite
from upf_agent.quality_repair import write_json


def compare_agents(root: Path, suite: Path, profile: str, output: Path) -> dict:
    cases = read_suite(suite, allow_drafts=True)
    selected, seen = [], set()
    for case in cases:
        if case.category == "log" and case.required_event_types[0] not in seen:
            selected.append(case)
            seen.add(case.required_event_types[0])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("Preserve completed comparison")
    rows = []
    for case in selected:
        for mode in ("single_evidence_pass", "fixed_tools", "finite_agent"):
            task = AgentTask(
                run_id=uuid4().hex,
                question=case.question,
                task_type="diagnose",
                log_generation=case.log_generation,
                model_profile=profile,
            )
            begin = perf_counter()
            if mode == "finite_agent":
                state = run_agent(root, task)
                evidence = state.evidence
                metrics = json.loads((root / "runs/p5" / task.run_id / "metrics.json").read_text())
                result = {"state": state.model_dump(mode="json"), "metrics": metrics}
                attempts, calls = (
                    metrics["tool_attempts"],
                    metrics["observed_usage"].get("requests", 0),
                )
                status = state.report.status.value
            else:
                packet = execute_tool(
                    root,
                    task,
                    ToolCall(name="analyze_log_window", arguments={"log_id": task.log_generation}),
                    [],
                )
                evidence = list(packet.evidence)
                result, calls, attempts = {"seed": packet.model_dump(mode="json")}, 0, 1
                if mode == "fixed_tools":
                    sample = execute_tool(
                        root,
                        task,
                        ToolCall(
                            name="find_log_events",
                            arguments={
                                "log_id": task.log_generation,
                                "event_type": case.required_event_types[0],
                                "limit": 3,
                            },
                        ),
                        evidence,
                    )
                    evidence += sample.evidence
                    attempts += 1
                    result["sample"] = sample.model_dump(mode="json")
                    if sample.evidence:
                        context = execute_tool(
                            root,
                            task,
                            ToolCall(
                                name="read_log_context",
                                arguments={
                                    "log_id": task.log_generation,
                                    "event_id": sample.evidence[0].native_id,
                                },
                            ),
                            evidence,
                        )
                        evidence += context.evidence
                        result["context"] = context.model_dump(mode="json")
                        attempts += 1
                else:
                    with CountedClient(root, AgentData(task=task), 180, 1) as client:
                        reply = client.chat(
                            [
                                {
                                    "role": "system",
                                    "content": "单次统计问答；不编造原文或根因。",
                                },
                                {
                                    "role": "user",
                                    "content": json.dumps(
                                        {
                                            "question": case.question,
                                            "evidence": [
                                                item.model_dump(mode="json") for item in evidence
                                            ],
                                        },
                                        ensure_ascii=False,
                                    ),
                                },
                            ],
                            max_tokens=320,
                        )
                        result["model_answer_audit_unverified"] = reply.get("content")
                        result["usage"] = client.stats
                        calls = client.stats["requests"]
                status = "partial"
            row = {
                "case": case.id,
                "mode": mode,
                "task_id": task.run_id,
                "elapsed_ms": (perf_counter() - begin) * 1000,
                "tool_attempts": attempts,
                "model_calls": calls,
                "status": status,
                "representative_log_ids": sorted(
                    {item.native_id for item in evidence if item.kind == "log"}
                ),
                "requested_event_evidence_hit": any(
                    item.metadata.get("event_type") in case.required_event_types
                    for item in evidence
                    if item.kind == "log"
                ),
                "root_cause_accuracy": None,
                "human_business_accuracy": None,
            }
            rows.append(row)
            write_json(output / f"{case.id}-{mode}.json", {"row": row, "result": result})
            print(f"agent {case.id} {mode}: {row['requested_event_evidence_hit']}", flush=True)
    summary = {
        "profile": profile,
        "cases": len(selected),
        "rows": rows,
        "scope": "log evidence coverage/cost; excludes root-cause accuracy",
        "fixed_route_uses_explicit_event_type": True,
        "single_pass_is_statistics_QA_not_full_manual_RAG": True,
        "formal_business_accuracy_claim": False,
    }
    write_json(output / "summary.json", summary)
    return summary
