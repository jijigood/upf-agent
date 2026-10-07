"""Reproducible development evaluation and a guarded, one-use final test boundary."""

import json
import resource
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx

from upf_agent.agent_contracts import AgentData, AgentTask, ToolCall
from upf_agent.agent_tools import REGISTRY, validate_call
from upf_agent.agent_worker import CountedClient
from upf_agent.config import load_model_config
from upf_agent.p3_answer import answer_from_retrieval
from upf_agent.p3_quality import evaluate_answer
from upf_agent.p3_search import search
from upf_agent.p6_contracts import percentiles, ranking_metrics, read_suite
from upf_agent.quality_repair import sha, write_json
from upf_agent.retrieval import RetrievalWorker


def pipeline_identity(root: Path) -> str:
    import hashlib

    files = sorted((root / "src/upf_agent").glob("*.py")) + [root / "uv.lock"]
    files += sorted((root / "config").glob("*"))
    files += [root / "evidence/p1/model-hashes.json", root / "data/p3/indexes/active.json"]
    files += sorted((root / "eval/p6").glob("topology_synthetic_*.csv"))
    pointer = root / "data/p3/indexes/active.json"
    if pointer.exists():
        generation = json.loads(pointer.read_text())["index_generation"]
        files.append(pointer.parent / generation / "manifest.json")
    files += sorted((root / "data/p4/logs").glob("*/manifest.json"))
    files = [path for path in files if path.is_file()]
    return hashlib.sha256("".join(sha(path) for path in files).encode()).hexdigest()


def prepare_retrieval(root: Path, suite: Path, output: Path, device: str = "cuda:1") -> dict:
    cases = read_suite(suite, allow_drafts=True)
    models = load_model_config(root / "config/models.yaml")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("Use a fresh evaluation cache directory")
    rows = []
    start = perf_counter()
    with RetrievalWorker(root, "embed", models.embedding.path, device) as embedding:
        with RetrievalWorker(root, "rerank", models.reranker.path, device) as reranking:
            loading_ms = (perf_counter() - start) * 1000
            for case in cases:
                if (
                    case.category not in {"command", "knowledge", "troubleshooting"}
                    and case.version != "9.9"
                ):
                    continue
                begin = perf_counter()
                result = search(
                    root, case.question, case.product, case.version, device, (embedding, reranking)
                )
                elapsed = (perf_counter() - begin) * 1000
                write_json(output / f"{case.id}.json", result)
                grades = {eid: 1 for eid in case.evidence_ids}
                rows.append(
                    {
                        "id": case.id,
                        "elapsed_ms": elapsed,
                        "label_status": case.label_status,
                        "metrics_scope": "draft_source_group_proxy_not_human_relevance",
                        "routes": {
                            name: ranking_metrics(ids, grades)
                            for name, ids in result["rankings"].items()
                        },
                        "context_tokens": result["context_tokens"],
                        "evidence_hash": sha(output / f"{case.id}.json"),
                    }
                )
                print(f"retrieval {case.id}: {elapsed:.0f} ms", flush=True)
    manifest = {
        "suite_sha256": sha(suite),
        "split": "dev",
        "rows": rows,
        "model_loading_ms": loading_ms,
        "warm_retrieval_ms": percentiles([row["elapsed_ms"] for row in rows]),
        "process_lifetime_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "formal_recall_or_business_accuracy": False,
        "persistent_workers_scope": "warm workers for this batch; CLI remains task-scoped",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def compare_model(root: Path, suite: Path, cache: Path, profile: str, output: Path) -> dict:
    cases = read_suite(suite, allow_drafts=True)
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["suite_sha256"] != sha(suite):
        raise ValueError("Retrieval inputs differ from evaluation suite")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("Preserve completed model run; use a new output")
    rows = []
    for case in cases:
        source = cache / f"{case.id}.json"
        if not source.exists():
            continue
        expected_hash = next(
            row["evidence_hash"] for row in manifest["rows"] if row["id"] == case.id
        )
        if sha(source) != expected_hash:
            raise ValueError("Cached evidence changed before model comparison")
        data = AgentData(
            task=AgentTask(
                run_id=uuid4().hex,
                question=case.question,
                task_type="manual",
                model_profile=profile,
            )
        )
        start = perf_counter()
        with CountedClient(root, data, 180, 4) as client:
            try:
                answer = answer_from_retrieval(
                    root, case.question, json.loads(source.read_text()), client
                )
                failure = None
            except (RuntimeError, ValueError, httpx.HTTPError) as error:
                answer = {"status": "failed", "claims": [], "citations": []}
                failure = str(error)
            usage = dict(client.stats)
        elapsed = (perf_counter() - start) * 1000
        literals = []
        if case.category == "command":
            literals = [case.expected_extracts[0].splitlines()[1]]
        score = evaluate_answer(
            {"expected_statuses": case.expected_statuses, "critical_literals": literals}, answer
        )
        cited = {cite["chunk_id"] for cite in answer["citations"]}
        row = {
            "id": case.id,
            "model": profile,
            "evidence_sha256": expected_hash,
            "elapsed_ms": elapsed,
            "usage": usage,
            "machine_checks": score,
            "draft_source_group_hit": bool(cited & set(case.evidence_ids))
            if case.evidence_ids
            else None,
            "failure": failure,
        }
        write_json(output / f"{case.id}.json", {"row": row, "answer": answer})
        rows.append(row)
        print(f"model {profile} {case.id}: {answer['status']}, {elapsed:.0f} ms", flush=True)
    summary = {
        "profile": profile,
        "suite_sha256": sha(suite),
        "split": "dev",
        "rows": rows,
        "cases": len(rows),
        "machine_pass": sum(row["machine_checks"]["machine_contract_pass"] for row in rows),
        "latency_ms": percentiles([row["elapsed_ms"] for row in rows]),
        "generation_requests": sum(row["usage"]["requests"] for row in rows),
        "human_business_accuracy": None,
        "formal_business_accuracy_claim": False,
        "pipeline_identity": pipeline_identity(root),
        "cold_retrieval_included": False,
    }
    write_json(output / "summary.json", summary)
    return summary


def score_parameter_response(
    task: AgentTask, expected: ToolCall, raw: str
) -> tuple[bool, str | None]:
    expected_args, expected_error = validate_call(task, expected, [])
    if expected_error:
        raise ValueError("Invalid parameter benchmark specification")
    try:
        call = ToolCall.model_validate_json(raw)
        actual_args, error = validate_call(task, call, [])
        exact = call.name == expected.name and actual_args == expected_args
        return error is None and exact, error or (None if exact else "requested_parameters_changed")
    except ValueError as error:
        return False, str(error)


def parameter_requests(root: Path, profile: str, output: Path) -> dict:
    """100 actual model requests in 10 families; tools are validated but never dispatched."""
    families = [
        ("analyze_log_window", {}),
        ("find_log_events", {"event_type": "pfcp_node_unreachable", "limit": 3}),
        ("find_log_events", {"event_type": "pfcp_heartbeat_request", "limit": 5}),
        ("find_log_events", {"event_type": "中文';DROP TABLE events;--", "limit": 1}),
        ("find_log_events", {"event_type": "far_handover", "limit": 50}),
        ("find_log_events", {"event_type": "license_not_installed", "limit": 1}),
        ("analyze_log_window", {"bucket_minutes": 60}),
        ("correlate_events", {"limit": 20}),
        ("find_log_events", {"event_type": "n4_name_failure", "limit": 2}),
        ("find_log_events", {"event_type": "pfcp_session_deleted", "limit": 4}),
    ]
    rows = []
    for number in range(100):
        name, additions = families[number // 10]
        task = AgentTask(
            run_id=uuid4().hex,
            question="固定参数调用评测",
            task_type="diagnose",
            model_profile=profile,
            log_generation="a6f18ac81a18aa2125fa",
        )
        expected = {"log_id": task.log_generation, **additions}
        # The expected request must itself satisfy the real production contract.
        _, expected_error = validate_call(task, ToolCall(name=name, arguments=expected), [])
        if expected_error:
            raise ValueError(f"Invalid benchmark specification: {expected_error}")
        messages = [
            {
                "role": "system",
                "content": "输出JSON {name,arguments}。保留指定参数，SQL仅是字符串。",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "case": number,
                        "request": {"name": name, "arguments": expected},
                        "confirmed_task": task.model_dump(mode="json"),
                        "scope_rule": "未提供时窗或设备：省略/null，不得猜测",
                        "schema": REGISTRY[name].model_json_schema(),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        attempts = []
        begin = perf_counter()
        with CountedClient(root, AgentData(task=task), 180, 2) as client:
            for _attempt in range(2):
                try:
                    response = client.chat(
                        messages, response_format={"type": "json_object"}, max_tokens=320
                    )
                    raw = response.get("content") or ""
                    valid, reason = score_parameter_response(
                        task, ToolCall(name=name, arguments=expected), raw
                    )
                except (RuntimeError, ValueError) as error:
                    raw, valid, reason = "", False, str(error)
                attempts.append({"raw": raw, "pass": valid, "reason": reason})
                if valid:
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": f"一次修复：{reason}。保留原请求参数，重新输出JSON。",
                    }
                )
            usage = dict(client.stats)
        rows.append(
            {
                "id": number,
                "family": number // 10,
                "attempts": attempts,
                "usage": usage,
                "elapsed_ms": (perf_counter() - begin) * 1000,
            }
        )
        print(f"parameters {profile} {number + 1}/100: {attempts[-1]['pass']}", flush=True)
        write_json(
            output, {"profile": profile, "rows": rows, "complete": False, "tools_executed": 0}
        )
    result = {
        "profile": profile,
        "cases": 100,
        "families": 10,
        "rows": rows,
        "first_pass": sum(row["attempts"][0]["pass"] for row in rows),
        "after_one_repair": sum(row["attempts"][-1]["pass"] for row in rows),
        "by_family": {
            str(i): {
                "n": 10,
                "first_pass": sum(row["attempts"][0]["pass"] for row in rows if row["family"] == i),
                "after_repair": sum(
                    row["attempts"][-1]["pass"] for row in rows if row["family"] == i
                ),
            }
            for i in range(10)
        },
        "complete": True,
        "tools_executed": 0,
        "prompt_version": "p6-parameters-v2-confirmed-task",
        "sampling_note": "10 families x 10 deterministic repeats; not 100 independent cases",
        "scope": "explicit argument generation; excludes autonomous selection/business accuracy",
    }
    write_json(output, result)
    return result


def reserve_heldout(root: Path, suite: Path, frozen: Path, output: Path) -> list:
    cases = read_suite(suite, "heldout")
    config = json.loads(frozen.read_text())
    if config["suite_sha256"] != sha(suite) or config["pipeline_identity"] != pipeline_identity(
        root
    ):
        raise ValueError("Frozen configuration or labels differ from current runtime")
    if config.get("status") != "frozen_after_reviewed_development":
        raise ValueError("Final test requires reviewed development selection")
    marker = root / "eval/p6/heldout_consumed.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "suite_sha256": sha(suite),
                "frozen_sha256": sha(frozen),
                "output": str(output),
                "consumed_before_execution": True,
            },
            handle,
        )
    return cases


def stream_probe(
    root: Path,
    profile: str,
    prompt: str,
    max_tokens: int = 256,
    thinking_policy: str = "auto",
) -> dict:
    if thinking_policy not in {"auto", "off"}:
        raise ValueError("Unknown generation thinking policy")
    model = load_model_config(root / "config/models.yaml").generation[profile]
    body = {
        "model": model.served_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if profile == "target":
        body["chat_template_kwargs"] = (
            {"enable_thinking": False} if thinking_policy == "off" else {"reasoning_effort": "low"}
        )
    begin, first_visible, first_event, content, usage, finish = (
        perf_counter(),
        None,
        None,
        "",
        {},
        None,
    )
    try:
        with httpx.Client(timeout=180, trust_env=False) as client:
            with client.stream("POST", model.endpoint + "/chat/completions", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        if first_event is None and (
                            delta.get("content")
                            or delta.get("reasoning_content")
                            or delta.get("reasoning")
                        ):
                            first_event = (perf_counter() - begin) * 1000
                        if delta.get("content"):
                            first_visible = first_visible or (perf_counter() - begin) * 1000
                            content += delta["content"]
                        finish = choice.get("finish_reason") or finish
        failure = None
    except (httpx.HTTPError, ValueError) as error:
        failure = str(error)
    return {
        "total_ms": (perf_counter() - begin) * 1000,
        "first_visible_ms": first_visible,
        "first_model_event_ms": first_event,
        "usage": usage,
        "finish_reason": finish,
        "content": content,
        "failure": failure,
    }


def benchmark(
    root: Path,
    profile: str,
    context: int,
    concurrency: int,
    output: Path,
    thinking_policy: str = "auto",
) -> dict:
    if context not in {16384, 32768} or concurrency not in {1, 2}:
        raise ValueError("Unsupported benchmark profile")
    if output.exists():
        raise ValueError("Preserve existing benchmark")
    # Identical evidence lengths for capacity profiles, two output/context budgets per profile.
    rows = []
    for budget in (1200, 5000):
        prefix = ("本地资料：消息数不是故障数；未知时区不计算绝对时长。\n" * 500)[:budget]
        prompt = prefix + "\n请用三句中文说明统计与根因结论的边界，不超过80字。"
        warmup = stream_probe(root, profile, prompt, thinking_policy=thinking_policy)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            observed = list(
                pool.map(
                    lambda _, prompt=prompt: stream_probe(
                        root, profile, prompt, thinking_policy=thinking_policy
                    ),
                    range(20),
                )
            )
        rows.append(
            {
                "context_characters": len(prompt),
                "requested_context_budget_characters": budget,
                "warmup": warmup,
                "runs": observed,
                "total_ms": percentiles([row["total_ms"] for row in observed]),
                "first_visible_ms": percentiles(
                    [
                        row["first_visible_ms"]
                        for row in observed
                        if row["first_visible_ms"] is not None
                    ]
                ),
                "nonempty_completed": sum(
                    bool(row["content"]) and row["finish_reason"] == "stop" for row in observed
                ),
            }
        )
        print(
            f"benchmark {profile} {context} c={concurrency} budget={budget}: "
            f"{rows[-1]['total_ms']}",
            flush=True,
        )
    result = {
        "profile": profile,
        "thinking_policy": thinking_policy,
        "chat_template_kwargs": (
            {"enable_thinking": False}
            if profile == "target" and thinking_policy == "off"
            else {"reasoning_effort": "low"}
            if profile == "target"
            else None
        ),
        "server_context_capacity": context,
        "client_concurrency": concurrency,
        "cases_per_budget": 20,
        "rows": rows,
        "scope": "warm generation only; retrieval excluded; character budgets differ from tokens",
        "prefix_cache_policy": "server defaults; repeated prompts may use prefix cache",
        "pipeline_identity": pipeline_identity(root),
    }
    write_json(output, result)
    return result


def warm_rag_benchmark(root: Path, suite: Path, profile: str, output: Path) -> dict:
    """20 sequential published answers including warm retrieval; not streaming TTFT."""
    if output.exists():
        raise ValueError("Use a new performance output")
    case = next(item for item in read_suite(suite, allow_drafts=True) if item.id == "command-001")
    models = load_model_config(root / "config/models.yaml")
    rows = []
    started = perf_counter()
    with RetrievalWorker(root, "embed", models.embedding.path, "cuda:1") as embed:
        with RetrievalWorker(root, "rerank", models.reranker.path, "cuda:1") as rerank:
            loading_ms = (perf_counter() - started) * 1000
            for number in range(21):
                begin = perf_counter()
                retrieval = search(
                    root, case.question, case.product, case.version, "cuda:1", (embed, rerank)
                )
                retrieved_ms = (perf_counter() - begin) * 1000
                task = AgentTask(
                    run_id=uuid4().hex,
                    question=case.question,
                    task_type="manual",
                    model_profile=profile,
                )
                with CountedClient(root, AgentData(task=task), 180, 4) as client:
                    answer = answer_from_retrieval(root, case.question, retrieval, client)
                    usage = dict(client.stats)
                row = {
                    "status": answer["status"],
                    "elapsed_ms": (perf_counter() - begin) * 1000,
                    "retrieval_ms": retrieved_ms,
                    "usage": usage,
                    "context_tokens": retrieval["context_tokens"],
                }
                rows.append(row)
                print(f"warm RAG {profile} {number}/20: {row['elapsed_ms']:.0f} ms", flush=True)
    result = {
        "profile": profile,
        "case": case.id,
        "warmup": rows[0],
        "rows": rows[1:],
        "model_loading_ms": loading_ms,
        "complete_answer_ms": percentiles([row["elapsed_ms"] for row in rows[1:]]),
        "first_visible_ms": None,
        "concurrency": 1,
        "suite_sha256": sha(suite),
        "pipeline_identity": pipeline_identity(root),
        "scope": "20 repeats of one command question, warm retrieval and published answer",
        "production_latency_population_claim": False,
    }
    write_json(output, result)
    return result
