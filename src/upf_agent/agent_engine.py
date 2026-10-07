"""A finite LangGraph investigation with task-scoped SQLite recovery and evidence-only facts."""

import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict, cast

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import JsonValue, ValidationError

from upf_agent.agent_contracts import (
    AgentData,
    AgentTask,
    InvestigationPlan,
    Receipt,
    ResumeInput,
    SourceFact,
    ToolCall,
)
from upf_agent.agent_runtime import Backend, LocalBackend
from upf_agent.agent_store import Journal, load_task
from upf_agent.agent_tools import failure, validate_call
from upf_agent.config import load_model_config
from upf_agent.contracts import AnswerReport, ResultStatus, TaskType
from upf_agent.log_tools import LogStore, write_json
from upf_agent.p3_index import load_active_index


class GraphState(TypedDict):
    payload: dict[str, JsonValue]


def state_data(state: GraphState) -> AgentData:
    return AgentData.model_validate_json(json.dumps(state["payload"]))


def state_value(data: AgentData) -> GraphState:
    return {"payload": data.model_dump(mode="json")}


def route_task(task: AgentTask) -> str:
    if task.task_type == TaskType.NETWORK_STATUS:
        return "missing_source"
    if requests_actual_topology(task.question) or requests_masked_real_identity(task.question):
        return "missing_source"
    if task.task_type == TaskType.DIAGNOSE:
        return "diagnose"
    if re.search(r"当前.*(?:温度|状态|在线|CPU|cpu)|实时.*(?:状态|性能)", task.question):
        return "missing_source"
    if re.search(r"日志|这次故障|这些告警|发生.*多少|统计.*告警", task.question):
        return "diagnose"
    return "manual"


def requests_actual_topology(question: str) -> bool:
    return bool(
        re.search(r"现场|实际|真实|当前", question)
        and re.search(r"拓扑|交换机|对端|链路", question)
        and re.search(r"端口|接口|连接|邻接|路径", question)
    )


def requests_masked_real_identity(question: str) -> bool:
    return bool(
        re.search(r"脱敏|掩码|匿名", question)
        and re.search(r"真实设备|实际设备|哪台设备|哪台真实|设备身份", question)
    )


def explicitly_requests_log_and_manual(task: AgentTask) -> bool:
    """Pin both sources before planning when the user explicitly asks to compare them."""
    return bool(re.search(r"手册|文档", task.question) and re.search(r"日志|告警", task.question))


def execution_fingerprint(root: Path) -> str:
    files = sorted(Path(__file__).parent.glob("agent_*.py"))
    files += [
        path
        for path in (root / "config/models.yaml", root / "config/retrieval.yaml", root / "uv.lock")
        if path.exists()
    ]
    return hashlib.sha256(
        "\n".join(hashlib.sha256(path.read_bytes()).hexdigest() for path in files).encode()
    ).hexdigest()


def same_log_source(root: Path, old: str, new: str) -> bool:
    left = LogStore(root / "data/p4/logs", old).manifest
    right = LogStore(root / "data/p4/logs", new).manifest
    return (
        left["source_sha256"] == right["source_sha256"] and left["synthetic"] == right["synthetic"]
    )


class AgentEngine:
    def __init__(self, root: Path, task: AgentTask, backend: Backend | None = None) -> None:
        self.root, self.task = root, task
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        self.backend = backend or LocalBackend(root)
        self.journal = Journal(root, task)

    def _hydrate(self, data: AgentData) -> None:
        for receipt in self.journal.receipts():
            if receipt.key not in {item.key for item in data.receipts}:
                data.receipts.append(receipt)
        available = {item.evidence_id: item for item in data.evidence}
        for receipt in data.receipts:
            for entry in receipt.packet.evidence:
                if not entry.evidence_id.startswith(data.task.run_id + ":"):
                    data.gaps.append("工具证据不属于当前任务，已拒绝。")
                    data.degraded = True
                    continue
                available[entry.evidence_id] = entry
        data.evidence = list(available.values())
        for receipt in data.receipts:
            for fact in receipt.packet.facts:
                if set(fact.evidence_ids) <= available.keys() and fact not in data.facts:
                    data.facts.append(fact)
            data.warnings.extend(
                item for item in receipt.packet.result.warnings if item not in data.warnings
            )
            data.gaps.extend(item for item in receipt.packet.gaps if item not in data.gaps)

    def _call(self, data: AgentData, call: ToolCall) -> Receipt:
        args, error = validate_call(data.task, call, data.evidence)
        canonical = (
            ToolCall(name=call.name, arguments=args.model_dump(mode="json")) if args else call
        )
        key = self.journal.key(canonical)
        if cached := self.journal.cached(key):
            receipt = Receipt(key=key, call=canonical, packet=cached, cached=True)
            self.journal.event("cache_hit", {"key": key, "call": canonical.model_dump(mode="json")})
            return receipt
        packet = failure(error, f"请求被拒绝：{error}。") if error else None
        if error:
            self.journal.event("rejected", {"call": call.model_dump(mode="json"), "error": error})
            return Receipt(key=key, call=call, packet=packet)
        while self.journal.attempts(key) < 2:
            if not self.journal.reserve("tools"):
                packet = failure("budget_exhausted", "工具次数或任务时间预算已耗尽。")
                break
            self.journal.attempted(key, canonical)
            self.journal.event(
                "tool_started", {"key": key, "call": canonical.model_dump(mode="json")}
            )
            try:
                packet = self.backend.tool(
                    data.task, canonical, data.evidence, self.journal.remaining()
                )
            except TimeoutError as error:
                packet = failure("timeout", str(error))
            except Exception as error:
                packet = failure("tool_failed", f"{type(error).__name__}: {error}")
            self.journal.account()
            self.journal.event(
                "tool_finished",
                {
                    "key": key,
                    "status": packet.result.status.value,
                    "error": packet.error_code,
                    "evidence_ids": [item.evidence_id for item in packet.evidence],
                },
            )
            if not packet.retryable:
                break
        packet = packet or failure("tool_failed", "恢复请求已达到一次重试上限。")
        packet.result.data["_task_scope"] = self.journal.scope_signature()
        if self.journal.attempts(key):
            self.journal.save_packet(key, packet)
        return Receipt(key=key, call=canonical, packet=packet)

    def _model(
        self, data: AgentData, operation: str, key: str, allowance: int, repair: str | None = None
    ) -> dict[str, JsonValue]:
        cache_key = f"{self.journal.scope_signature()}:{key}"
        if cached := self.journal.model_cached(cache_key):
            return cached
        if not self.journal.reserve("models", allowance):
            raise TimeoutError("模型调用或时间预算已耗尽。")
        self.journal.event(
            "model_started", {"operation": operation, "key": key, "reserved_requests": allowance}
        )
        try:
            reply = self.backend.model(operation, data, self.journal.remaining(), allowance, repair)
            self.journal.save_model(cache_key, reply)
            self.journal.event(
                "model_finished", {"operation": operation, "key": key, "usage": reply.get("usage")}
            )
            return reply
        finally:
            self.journal.account()

    def classify(self, state: GraphState) -> GraphState:
        data = state_data(state)
        data.route = route_task(data.task)
        if data.route == "missing_source":
            if requests_actual_topology(data.task.question):
                data.gaps.append(
                    "缺少经过核验的现场真实拓扑和交换机端口映射；手册中的服务器侧接口"
                    "或示例端口不能确定实际N3交换机端口。"
                )
            elif requests_masked_real_identity(data.task.question):
                data.gaps.append(
                    "当前资料缺少受控的脱敏标识与真实设备身份映射，无法判断实际是哪台设备；"
                    "重新导入同一份脱敏资料也不能还原身份。"
                )
            else:
                data.gaps.append("未接入实时网管/设备遥测数据源，无法报告当前现场状态。")
            data.terminal = True
        elif data.route == "diagnose" and not data.task.log_generation:
            data.questions = ["请先导入日志并提供已登记的 log_generation。"]
        return state_value(data)

    def seed(self, state: GraphState) -> GraphState:
        data = state_data(state)
        self._hydrate(data)
        if data.questions or data.terminal:
            return state_value(data)
        if data.route == "diagnose" and data.task.log_generation is None:
            data.questions = ["仍缺少日志，请提供已登记的 log_generation；无法提供时可保留缺口。"]
            return state_value(data)
        if data.route == "manual":
            call = ToolCall(name="search_manual", arguments={"query": data.task.question})
        else:
            call = ToolCall(
                name="analyze_log_window", arguments={"log_id": data.task.log_generation}
            )
        receipt = self._call(data, call)
        if receipt.key not in {item.key for item in data.receipts}:
            data.receipts.append(receipt)
        self._hydrate(data)
        if receipt.packet.result.status == ResultStatus.NEED_INPUT:
            data.questions = list(receipt.packet.gaps) or ["请补充查询所需的明确时区/时间范围。"]
            data.terminal = True
        elif (
            receipt.packet.error_code or receipt.packet.result.status == ResultStatus.MISSING_SOURCE
        ):
            data.terminal = True
            data.degraded = receipt.packet.error_code not in (None, "missing_source")
        if (
            data.route == "diagnose"
            and not data.terminal
            and data.task.manual_generation
            and explicitly_requests_log_and_manual(data.task)
        ):
            manual_call = ToolCall(name="search_manual", arguments={"query": data.task.question})
            manual_receipt = self._call(data, manual_call)
            if manual_receipt.key not in {item.key for item in data.receipts}:
                data.receipts.append(manual_receipt)
            self._hydrate(data)
            retrieval = manual_receipt.packet.result.data.get("retrieval", {})
            native = retrieval.get("native_section") if isinstance(retrieval, dict) else None
            selected_ids = set(native.get("complete_chunk_ids", [])) if native else set()
            for entry in manual_receipt.packet.evidence:
                if entry.kind == "manual" and entry.native_id in selected_ids:
                    fact = SourceFact(text=entry.excerpt[:10000], evidence_ids=[entry.evidence_id])
                    if fact not in data.facts:
                        data.facts.append(fact)
            if manual_receipt.packet.error_code:
                data.degraded = True
        if data.route == "diagnose" and (
            data.task.mode == "deterministic"
            or re.search(r"多少|数量|统计|计数", data.task.question)
            and not re.search(r"原因|为什么|定位|排查", data.task.question)
        ):
            data.terminal = True
        return state_value(data)

    def manual(self, state: GraphState) -> GraphState:
        data = state_data(state)
        if data.terminal:
            return state_value(data)
        if data.task.mode == "deterministic":
            data.gaps.append("确定性模式只返回候选手册原文，不把检索命中当作问答完成。")
            data.degraded = True
            return state_value(data)
        remaining = data.task.budget.max_model_calls - int(
            self.journal.counters()["model_requests_reserved"]
        )
        try:
            reply = self._model(
                data, "manual_answer", f"manual:{data.resume_count}", min(4, remaining)
            )
            answer = cast(dict[str, JsonValue], reply["answer"])
            known = {item.native_id: item for item in data.evidence}
            facts = []
            for claim in cast(list[dict[str, JsonValue]], answer.get("claims", [])):
                quotes, ids = [], []
                for citation in cast(list[dict[str, str]], claim["citations"]):
                    entry = known.get(citation["chunk_id"])
                    if entry is None or citation["quote"] not in entry.excerpt:
                        raise ValueError("手册引用不是当前任务已验证原文。")
                    quotes.append(citation["quote"])
                    ids.append(entry.evidence_id)
                text = "\n".join(dict.fromkeys(quotes))
                if claim["text"] != text:
                    raise ValueError("手册发布文字不是引用原文。")
                facts.append(SourceFact(text=text, evidence_ids=list(dict.fromkeys(ids))))
            data.facts.extend(facts)
            data.gaps.extend(cast(list[str], answer.get("missing_information", [])))
            data.warnings.extend(cast(list[str], answer.get("warnings", [])))
            if answer["status"] == "failed":
                data.degraded = True
            if answer["status"] == "missing_source":
                data.evidence = []  # Candidate hits are not an answer.
        except (TimeoutError, ValueError, KeyError, RuntimeError, ValidationError) as error:
            data.gaps.append(f"手册问答降级：{error}")
            data.degraded = True
        return state_value(data)

    def plan(self, state: GraphState) -> GraphState:
        data = state_data(state)
        self._hydrate(data)
        if data.terminal or self.journal.remaining() <= 0:
            data.terminal = True
            if self.journal.remaining() <= 0:
                data.gaps.append("任务时间预算已耗尽。")
            return state_value(data)
        cached_key = f"plan:{data.resume_count}:{data.rounds + 1}"
        model_key = f"{self.journal.scope_signature()}:{cached_key}"
        if self.journal.model_cached(model_key) is None and not self.journal.reserve("rounds"):
            data.gaps.append("调查已达到最多 3 轮的业务限制。")
            data.gaps.extend(data.questions)
            data.questions = []
            data.terminal = True
            return state_value(data)
        data.rounds = int(self.journal.counters()["rounds"])
        repair = None
        repaired_cached = self.journal.model_cached(model_key + ":repair") is not None
        data.repairs = int(self.journal.counters()["parameter_repairs"])
        for attempt in [1] if repaired_cached else range(2):
            try:
                reply = self._model(
                    data, "plan", cached_key + (":repair" if attempt else ""), 1, repair
                )
                proposal = InvestigationPlan.model_validate_json(str(reply.get("content") or ""))
                errors = [
                    validate_call(data.task, call, data.evidence)[1] for call in proposal.actions
                ]
                if any(errors):
                    raise ValueError("工具参数错误：" + str(errors))
                ids = {item.evidence_id for item in data.evidence}
                if any(
                    not set([*item.supporting_evidence_ids, *item.opposing_evidence_ids]) <= ids
                    for item in proposal.candidates
                ):
                    raise ValueError("候选解释引用未知/其他任务证据。")
                data.actions = proposal.actions
                data.questions = proposal.questions
                data.candidates = proposal.candidates
                data.terminal = proposal.done or not proposal.actions
                return state_value(data)
            except (TimeoutError, ValueError, KeyError, RuntimeError, ValidationError) as error:
                self.journal.event("plan_rejected", {"error": str(error), "attempt": attempt + 1})
                if (
                    attempt == 0
                    and isinstance(error, (ValueError, KeyError, ValidationError))
                    and self.journal.reserve("repairs")
                    and self.journal.remaining() > 0
                ):
                    data.repairs += 1
                    repair = str(error)
                    continue
                data.gaps.append(f"规划失败，保留确定性证据：{error}")
                data.degraded = data.terminal = True
                return state_value(data)
        return state_value(data)

    def execute(self, state: GraphState) -> GraphState:
        data = state_data(state)
        progress = False
        for call in data.actions:
            receipt = self._call(data, call)
            if receipt.key not in {item.key for item in data.receipts}:
                progress = progress or not receipt.cached
                data.receipts.append(receipt)
            self._hydrate(data)
            if receipt.packet.error_code:
                data.gaps.extend(receipt.packet.gaps)
                if receipt.packet.error_code in {"timeout", "budget_exhausted"}:
                    data.terminal = True
                    break
        data.actions = []
        if not progress:
            data.gaps.append("补查没有新增信息，已终止重复工具循环。")
            data.terminal = True
        return state_value(data)

    def verify(self, state: GraphState) -> GraphState:
        data = state_data(state)
        self._hydrate(data)
        ids = {item.evidence_id for item in data.evidence}
        if any(not set(item.evidence_ids) <= ids for item in data.facts):
            data.facts = [item for item in data.facts if set(item.evidence_ids) <= ids]
            data.gaps.append("未通过当前任务证据校验的事实已移除。")
        if data.route == "manual" and data.task.mode == "agent":
            # Hydration restores tool candidates from receipts; only the sources
            # retained by the manual publication checks may enter the report.
            published_ids = {eid for fact in data.facts for eid in fact.evidence_ids}
            data.evidence = [entry for entry in data.evidence if entry.evidence_id in published_ids]
        return state_value(data)

    def await_input(self, state: GraphState) -> GraphState:
        data = state_data(state)
        raw = interrupt({"run_id": data.task.run_id, "questions": data.questions})
        answer = ResumeInput.model_validate_json(json.dumps(raw))
        old = self.journal.scope_signature()
        if answer.log_generation:
            if (
                data.task.log_generation
                and data.task.log_generation != answer.log_generation
                and not same_log_source(self.root, data.task.log_generation, answer.log_generation)
            ):
                raise ValueError("不能把现有任务换绑其他日志，请创建新任务。")
            data.task.log_generation = answer.log_generation
        if answer.topology_generation:
            if (
                data.task.topology_generation
                and data.task.topology_generation != answer.topology_generation
            ):
                raise ValueError("不能替换当前拓扑 generation，请创建新任务。")
            data.task.topology_generation = answer.topology_generation
        if answer.device_ids is not None:
            data.task.device_ids = answer.device_ids
        if answer.time_window is not None:
            data.task.time_window = answer.time_window
        self.task = data.task
        self.journal.task = data.task
        data.user_answers.append(answer.answer)
        data.resume_count += 1
        data.questions = []
        data.terminal = False
        if old != self.journal.scope_signature():
            data.receipts, data.evidence, data.facts, data.gaps, data.candidates = (
                [],
                [],
                [],
                [],
                [],
            )
        return state_value(data)

    def finalize(self, state: GraphState) -> GraphState:
        data = state_data(state)
        cross_source = data.route == "diagnose" and explicitly_requests_log_and_manual(data.task)
        if cross_source:
            data.gaps.append(
                "手册历史案例与日志仅为并列来源；没有经过核验的同设备身份、版本和时间关联，"
                "不能认定两者是同一次故障，也不能据现有证据排除。"
            )
        if data.route == "diagnose":
            if re.search(r"独立故障|根因", data.task.question) and data.task.log_generation:
                data.gaps.append(
                    "仅凭当前日志事件与计数，不能确认这是独立故障或确定根因；"
                    "仍需设备身份、时间关联及现场验证。"
                )
                if (
                    "pfcp_heartbeat_response" in data.task.question
                    and any(
                        "pfcp_heartbeat_response / observed" in fact.text
                        for fact in data.facts
                    )
                ):
                    data.gaps.append(
                        "所引 pfcp_heartbeat_response 是已观测到的心跳响应消息，"
                        "不是 raised 故障告警；消息条数也不是独立故障次数。"
                        "这条响应原文没有给出同一设备和时间窗内的心跳超时、"
                        "关联告警或业务影响证据，因此不能由响应本身划定一次故障，"
                        "更不能据此推断根因。需补充可配对的请求/响应、告警与现场状态。"
                    )
            elif re.search(r"独立故障|根因", data.task.question):
                data.gaps.append("尚未提供可核对的日志事件，不能判断独立故障或根因。")
            if not data.task.topology_generation:
                data.gaps.append("没有真实拓扑，无法确认实际链路/交换机端口与传播路径。")
            if data.task.log_generation:
                manifest_path = (
                    self.root / "data/p4/logs" / data.task.log_generation / "manifest.json"
                )
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    if manifest.get("unresolved_timezone_events"):
                        data.gaps.append("源时区未知，未计算绝对时间关联或故障持续时间。")
                    if not manifest.get("device_id"):
                        data.gaps.append("缺少原始真实设备身份，脱敏值不能用于事件配对。")
                    if manifest.get("synthetic"):
                        data.warnings.append("输入日志为合成测试数据，不代表真实事故。")
        status = ResultStatus.COMPLETED
        if self.journal.remaining() <= 0:
            data.gaps.extend(data.questions)
            data.questions = []
        if data.questions:
            status = ResultStatus.NEED_INPUT
        elif data.gaps or data.degraded:
            status = (
                ResultStatus.PARTIAL if data.facts or data.evidence else ResultStatus.MISSING_SOURCE
            )
            if data.degraded and not data.evidence and not data.facts:
                status = ResultStatus.FAILED
        data.gaps = list(dict.fromkeys(data.gaps))
        data.warnings = list(dict.fromkeys(data.warnings))
        model_id = data.task.model_profile
        try:
            model_id = (
                load_model_config(self.root / "config/models.yaml")
                .generation[data.task.model_profile]
                .id
            )
        except (OSError, ValueError):
            pass
        data.report = AnswerReport(
            run_id=data.task.run_id,
            status=status,
            facts=[item.text for item in data.facts],
            hypotheses=[]
            if cross_source
            else ["模型候选（未经因果验证）：" + item.text for item in data.candidates],
            supporting_evidence_ids=list(
                dict.fromkeys(eid for fact in data.facts for eid in fact.evidence_ids)
            ),
            opposing_evidence_ids=[]
            if cross_source
            else list(
                dict.fromkeys(eid for item in data.candidates for eid in item.opposing_evidence_ids)
            ),
            missing_information=list(dict.fromkeys([*data.gaps, *data.questions])),
            recommendations=["按报告来源位置复核原文，并补充列出的现场资料；建议不自动执行。"]
            if data.gaps
            else [],
            model_id=model_id if data.task.mode == "agent" else None,
            corpus_generation=data.task.manual_generation,
        )
        return state_value(data)

    def graph(self, saver: SqliteSaver, pause_after: str | None = None):
        graph = StateGraph(GraphState)
        for name in (
            "classify",
            "seed",
            "manual",
            "plan",
            "execute",
            "verify",
            "await_input",
            "finalize",
        ):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "classify")
        graph.add_edge("classify", "seed")

        def after_seed(state: GraphState) -> str:
            data = state_data(state)
            if data.questions:
                return "await_input"
            if data.terminal:
                return "finalize"
            return "manual" if data.route == "manual" else "plan"

        def after_plan(state: GraphState) -> str:
            data = state_data(state)
            if data.actions and not data.terminal:
                return "execute"
            if data.questions and not data.degraded and self.journal.remaining() > 0:
                return "await_input"
            return "finalize" if data.terminal else "execute"

        graph.add_conditional_edges("seed", after_seed)
        graph.add_edge("manual", "verify")
        graph.add_conditional_edges("plan", after_plan)
        graph.add_edge("execute", "verify")
        graph.add_conditional_edges(
            "verify",
            lambda state: (
                "finalize"
                if state_data(state).route == "manual" or state_data(state).terminal
                else "plan"
            ),
        )
        graph.add_edge("await_input", "seed")
        graph.add_edge("finalize", END)
        return graph.compile(
            checkpointer=saver, interrupt_after=[pause_after] if pause_after else None
        )

    def run(
        self,
        resume: ResumeInput | None = None,
        recover: bool = False,
        pause_after: str | None = None,
    ) -> AgentData:
        locks = self.root / "data/p5/locks"
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / f"{self.task.run_id}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("当前任务已有运行进程。") from error
            connection = sqlite3.connect(
                self.root / "data/p5/checkpoints.sqlite3", check_same_thread=False
            )
            try:
                saver = SqliteSaver(connection)
                graph = self.graph(saver, pause_after)
                config = {"configurable": {"thread_id": self.task.run_id}, "recursion_limit": 40}
                snapshot = graph.get_state(config)
                if recover:
                    if (
                        self.task.execution_fingerprint
                        and self.task.execution_fingerprint != execution_fingerprint(self.root)
                    ):
                        raise ValueError(
                            "代码、环境锁或配置已改变，请恢复该任务的锁定版本或创建新任务。"
                        )
                    if not snapshot.values:
                        raise ValueError("任务没有可恢复 checkpoint。")
                    if not snapshot.next:
                        return state_data(snapshot.values)
                    if snapshot.interrupts and resume is None:
                        raise ValueError("等待输入的任务需要 --answer 或明确参数。")
                    initial = (
                        Command(resume=resume.model_dump(mode="json"))
                        if snapshot.interrupts
                        else None
                    )
                else:
                    if snapshot.values:
                        raise ValueError("run_id 已存在，不能覆盖任务。")
                    self.task.execution_fingerprint = execution_fingerprint(self.root)
                    if self.task.manual_generation is None:
                        try:
                            _, manifest = load_active_index(self.root)
                            self.task.manual_generation = manifest["index_generation"]
                        except (OSError, ValueError, KeyError):
                            pass
                    self.journal.task = self.task
                    self.journal.register()
                    initial = state_value(AgentData(task=self.task))
                self.journal.begin()
                try:
                    graph.invoke(initial, config=config)
                finally:
                    self.journal.account(finish=True)
                snapshot = graph.get_state(config)
                data = state_data(snapshot.values)
                if snapshot.next:
                    data = state_data(self.finalize(state_value(data)))
                    data.report.status = ResultStatus.NEED_INPUT
                    if not data.questions:
                        data.report.missing_information.append(
                            "运行已保存 checkpoint，可用 resume 继续。"
                        )
                export_run(self.root, data, self.journal, list(snapshot.next))
                return data
            finally:
                connection.close()
                self.journal.close()


def run_agent(
    root: Path, task: AgentTask, backend: Backend | None = None, pause_after: str | None = None
) -> AgentData:
    return AgentEngine(root, task, backend).run(pause_after=pause_after)


def resume_agent(
    root: Path, run_id: str, answer: ResumeInput | None = None, backend: Backend | None = None
) -> AgentData:
    task = load_task(root, run_id)
    if (
        answer
        and answer.log_generation
        and task.log_generation
        and answer.log_generation != task.log_generation
        and not same_log_source(root, task.log_generation, answer.log_generation)
    ):
        raise ValueError("不能把现有任务换绑其他日志，请创建新任务。")
    if (
        answer
        and answer.topology_generation
        and task.topology_generation
        and answer.topology_generation != task.topology_generation
    ):
        raise ValueError("不能替换当前拓扑 generation，请创建新任务。")
    return AgentEngine(root, task, backend).run(resume=answer, recover=True)


def export_run(root: Path, data: AgentData, journal: Journal, pending: list[str]) -> None:
    directory = root / "runs/p5" / data.task.run_id
    directory.mkdir(parents=True, exist_ok=True)
    prior = directory / "report.json"
    if prior.exists():
        archive = directory / "history" / hashlib.sha256(prior.read_bytes()).hexdigest()[:16]
        archive.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if path.is_file() and not (archive / path.name).exists():
                shutil.copy2(path, archive / path.name)
    report = {
        **data.report.model_dump(mode="json"),
        "question": data.task.question,
        "fact_records": [item.model_dump(mode="json") for item in data.facts],
        "candidates": [item.model_dump(mode="json") for item in data.candidates],
        "evidence": [item.model_dump(mode="json") for item in data.evidence],
        "warnings": data.warnings,
        "degraded": data.degraded,
        "publication_policy": "deterministic_tool_facts_and_verified_manual_source_extracts",
    }
    write_json(directory / "report.json", report)
    write_json(
        directory / "manifest.json",
        {
            "task": data.task.model_dump(mode="json"),
            "pipeline_version": "p5-agent-v1",
            "pending_nodes": pending,
            "no_equipment_changes": True,
            "framework": {
                "langgraph": version("langgraph"),
                "sqlite_checkpointer": version("langgraph-checkpoint-sqlite"),
            },
        },
    )
    write_json(directory / "metrics.json", journal.counters())
    write_json(directory / "state.json", data.model_dump(mode="json"))
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in journal.events()),
        encoding="utf-8",
    )
    lines = [
        f"问题：{data.task.question}",
        f"任务：{data.task.run_id}",
        f"状态：{data.report.status.value}",
        "",
        "观测事实与手册原文：",
    ]
    lines += [f"- {fact.text} [{', '.join(fact.evidence_ids)}]" for fact in data.facts]
    lines += [
        "",
        "候选解释（未验证因果）：",
        *[f"- {item}" for item in data.report.hypotheses],
        "",
        "缺失信息：",
        *[f"- {item}" for item in data.report.missing_information],
        "",
        "建议检查：",
        *[f"- {item}" for item in data.report.recommendations],
        "",
        "提示：",
        *[f"- {item}" for item in data.warnings],
        "",
        "来源：",
    ]
    lines += [
        f"- [{entry.evidence_id}] {entry.source_id}；{entry.locator}；"
        f"SHA256={entry.source_sha256}\n  {entry.excerpt[:1600]}"
        for entry in data.evidence
    ]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
