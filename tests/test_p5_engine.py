import json
from pathlib import Path
from uuid import uuid4

import pytest

from upf_agent.agent_contracts import AgentBudget, AgentData, AgentTask, ResumeInput
from upf_agent.agent_engine import resume_agent, run_agent
from upf_agent.agent_tools import execute_tool, failure
from upf_agent.log_tools import ingest_log


class FakeBackend:
    def __init__(self, root: Path, behavior: str = "normal") -> None:
        self.root, self.behavior = root, behavior
        self.executed = []
        self.models = 0

    def tool(self, task, call, known, timeout):
        self.executed.append(call.name)
        if self.behavior == "retry" and self.executed.count(call.name) == 1:
            return failure("tool_failed", "temporary unavailable", retryable=True)
        return execute_tool(self.root, task, call, known)

    def model(self, operation, data, timeout, allowance, repair=None):
        self.models += 1
        if self.behavior == "timeout":
            raise TimeoutError("synthetic model timeout")
        if self.behavior == "empty":
            return {"content": ""}
        if self.behavior == "ask":
            return {
                "content": json.dumps(
                    {
                        "done": True,
                        "questions": ["请补充时间范围及原始时区"] if self.models == 1 else [],
                    }
                )
            }
        if self.behavior == "repair" and self.models == 1:
            return {"content": json.dumps({"actions": [{"name": "shell_exec", "arguments": {}}]})}
        limit = data.rounds if self.behavior == "rounds" else 1
        actions = [
            {
                "name": "find_log_events",
                "arguments": {"log_id": data.task.log_generation, "limit": limit},
            }
        ]
        done = self.models > (2 if self.behavior == "repair" else 1) and self.behavior not in {
            "rounds",
            "repeat",
        }
        return {
            "content": json.dumps({"actions": [] if done else actions, "done": done}),
            "usage": {
                "requests": 1,
                "prompt_tokens_observed": 10,
                "completion_tokens_observed": 10,
            },
        }


def setup_log(root: Path) -> str:
    source = root / "sample.log"
    source.write_text(
        "2025-05-14 00:00:00 ERROR #Alarm# report PFCP node unreachable!\n"
        "2025-05-14 00:00:01 DEBUG Recv #MsgHeartbeat# Req from cp.\n"
    )
    return ingest_log(source, root / "data/p4/logs")["generation"]


def new_task(generation: str | None, **kwargs) -> AgentTask:
    return AgentTask(
        run_id=uuid4().hex,
        question="分析这次故障的日志原因",
        task_type="diagnose",
        log_generation=generation,
        **kwargs,
    )


def metrics(root: Path, data: AgentData) -> dict:
    return json.loads((root / "runs/p5" / data.task.run_id / "metrics.json").read_text())


def test_rejected_manual_candidates_do_not_reappear_in_published_report(tmp_path):
    from upf_agent.agent_contracts import ToolPacket
    from upf_agent.agent_tools import evidence
    from upf_agent.contracts import ResultStatus, ToolResult

    class MissingVersionBackend:
        def tool(self, task, call, known, timeout):
            return ToolPacket(
                result=ToolResult(status=ResultStatus.COMPLETED, elapsed_ms=0),
                evidence=[
                    evidence(
                        task,
                        "native-1",
                        "manual",
                        "a" * 64,
                        "p[1]",
                        "manual",
                        "不适用版本的配置步骤",
                    )
                ],
            )

        def model(self, operation, data, timeout, allowance, repair=None):
            return {
                "answer": {
                    "status": "missing_source",
                    "claims": [],
                    "missing_information": ["缺少所请求版本的手册"],
                    "warnings": [],
                },
                "usage": {"requests": 0},
            }

    current = AgentTask(
        run_id=uuid4().hex, question="9.9版本如何配置接口", task_type="manual", version="9.9"
    )
    data = run_agent(tmp_path, current, backend=MissingVersionBackend())
    assert data.report.status == ResultStatus.MISSING_SOURCE
    assert data.evidence == []
    published = (tmp_path / "runs/p5" / current.run_id / "report.md").read_text()
    assert "不适用版本的配置步骤" not in published
    assert "缺少所请求版本的手册" in published
    # The original tool receipt is still retained for debugging, not published.
    assert data.receipts[0].packet.evidence[0].excerpt == "不适用版本的配置步骤"


def test_manual_report_retains_selected_source_and_hides_unselected_candidate(tmp_path):
    from upf_agent.agent_contracts import ToolPacket
    from upf_agent.agent_tools import evidence
    from upf_agent.contracts import ResultStatus, ToolResult

    class SelectedSourceBackend:
        def tool(self, task, call, known, timeout):
            return ToolPacket(
                result=ToolResult(status=ResultStatus.COMPLETED, elapsed_ms=0),
                evidence=[
                    evidence(
                        task, "selected", "manual", "a" * 64, "p[1]", "manual", "show cp-node"
                    ),
                    evidence(task, "other", "manual", "a" * 64, "p[2]", "manual", "无关配置步骤"),
                ],
            )

        def model(self, operation, data, timeout, allowance, repair=None):
            return {
                "answer": {
                    "status": "completed",
                    "claims": [
                        {
                            "text": "show cp-node",
                            "citations": [{"chunk_id": "selected", "quote": "show cp-node"}],
                        }
                    ],
                    "missing_information": [],
                    "warnings": [],
                },
                "usage": {"requests": 0},
            }

    current = AgentTask(run_id=uuid4().hex, question="引用命令原文", task_type="manual")
    data = run_agent(tmp_path, current, backend=SelectedSourceBackend())
    assert data.report.status == ResultStatus.COMPLETED
    assert [entry.native_id for entry in data.evidence] == ["selected"]
    published = (tmp_path / "runs/p5" / current.run_id / "report.md").read_text()
    assert "show cp-node" in published
    assert "无关配置步骤" not in published
    assert len(data.receipts[0].packet.evidence) == 2


def test_log_agent_flow_and_reports_have_only_code_facts(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path)
    data = run_agent(tmp_path, new_task(setup_log(tmp_path)), backend)
    assert data.report.status.value == "partial"
    assert backend.executed == ["analyze_log_window", "find_log_events"]
    assert data.facts[0].text.startswith("所选范围包含 2 条")
    assert any("时区未知" in gap for gap in data.report.missing_information)
    assert all(
        eid.startswith(data.task.run_id + ":") for eid in data.report.supporting_evidence_ids
    )
    for name in ["report.json", "report.md", "events.jsonl", "manifest.json", "metrics.json"]:
        assert (tmp_path / "runs/p5" / data.task.run_id / name).is_file()


def test_explicit_cross_source_question_seeds_manual_before_planning(tmp_path: Path) -> None:
    from upf_agent.agent_contracts import ToolPacket
    from upf_agent.agent_tools import evidence
    from upf_agent.contracts import ResultStatus, ToolResult

    class CrossSourceBackend(FakeBackend):
        def tool(self, task, call, known, timeout):
            if call.name == "search_manual":
                self.executed.append(call.name)
                return ToolPacket(
                    result=ToolResult(
                        status=ResultStatus.COMPLETED,
                        elapsed_ms=0,
                        data={
                            "retrieval": {
                                "native_section": {"complete_chunk_ids": ["manual-license"]}
                            }
                        },
                    ),
                    evidence=[
                        evidence(
                            task,
                            "manual-license",
                            "manual",
                            "a" * 64,
                            "p[1]",
                            "manual",
                            "历史案例：未安装license时pod crash",
                        )
                    ],
                )
            return super().tool(task, call, known, timeout)

        def model(self, operation, data, timeout, allowance, repair=None):
            assert any(item.kind == "manual" for item in data.evidence)
            return {
                "content": json.dumps(
                    {
                        "actions": [],
                        "done": True,
                        "candidates": [
                            {
                                "text": "已证明同一故障",
                                "supporting_evidence_ids": [data.evidence[-1].evidence_id],
                            }
                        ],
                    }
                )
            }

    generation = setup_log(tmp_path)
    task = AgentTask(
        run_id=uuid4().hex,
        question="对照日志告警和手册历史案例，能否认定同一次故障？",
        task_type="diagnose",
        log_generation=generation,
        manual_generation="a" * 20,
    )
    backend = CrossSourceBackend(tmp_path)
    data = run_agent(tmp_path, task, backend)
    assert backend.executed[:2] == ["analyze_log_window", "search_manual"]
    assert any(item.native_id == "manual-license" for item in data.evidence)
    assert all(item.evidence_id.startswith(task.run_id + ":") for item in data.evidence)
    assert any("历史案例：未安装license" in fact for fact in data.report.facts)
    assert data.report.hypotheses == []
    assert any("不能认定两者是同一次故障" in gap for gap in data.report.missing_information)


def test_simple_counts_skip_planning_and_missing_live_source_skips_all_tools(
    tmp_path: Path,
) -> None:
    generation = setup_log(tmp_path)
    backend = FakeBackend(tmp_path)
    task = new_task(generation)
    task.question = "统计日志消息数量"
    assert run_agent(tmp_path, task, backend).facts
    assert backend.models == 0
    task = AgentTask(run_id=uuid4().hex, question="当前设备CPU温度", task_type="network_status")
    backend.executed.clear()
    data = run_agent(tmp_path, task, backend)
    assert data.report.status.value == "missing_source"
    assert backend.executed == []


@pytest.mark.parametrize(
    ("question", "expected_gap"),
    [
        (
            "没有实际拓扑，请确定现场 N3 交换机端口。",
            "手册中的服务器侧接口或示例端口不能确定实际N3交换机端口",
        ),
        (
            "仅有脱敏日志，实际发生告警的是哪台真实设备？",
            "缺少受控的脱敏标识与真实设备身份映射",
        ),
        (
            "请把掩码地址还原成实际 UPF 的设备身份。",
            "缺少受控的脱敏标识与真实设备身份映射",
        ),
    ],
)
def test_live_topology_and_masked_identity_questions_do_not_use_manual_examples(
    tmp_path: Path, question: str, expected_gap: str
) -> None:
    backend = FakeBackend(tmp_path)
    task = AgentTask(run_id=uuid4().hex, question=question, task_type="other")
    data = run_agent(tmp_path, task, backend)
    assert data.report.status.value == "missing_source"
    assert data.report.facts == []
    assert any(expected_gap in gap for gap in data.report.missing_information)
    assert backend.executed == [] and backend.models == 0


def test_log_report_answers_unverified_root_cause_question_explicitly(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path)
    task = new_task(setup_log(tmp_path))
    task.question = "从日志原文核对告警，说明能否据此确定独立故障和根因。"
    data = run_agent(tmp_path, task, backend)
    assert data.facts
    assert data.report.hypotheses == []
    assert any(
        "不能确认这是独立故障或确定根因" in gap
        for gap in data.report.missing_information
    )


def test_heartbeat_response_explains_why_it_cannot_prove_a_fault(tmp_path: Path) -> None:
    source = tmp_path / "heartbeat.log"
    source.write_text(
        "2025-05-15 09:10:45.375964 DEBUG FID.0 "
        "upfd_pfcp.c:2112(upfdPfcpMsgEntryHeartBeatReq): "
        "Send #MsgHeartbeat# Rsp to cp node:2408:8142:60ff:fc02:1706:406::, msg len:16\n"
    )
    generation = ingest_log(source, tmp_path / "data/p4/logs")["generation"]
    task = new_task(generation)
    task.question = (
        "从日志原文核对 pfcp_heartbeat_response，说明能否据此确定独立故障和根因。"
    )
    data = run_agent(tmp_path, task, FakeBackend(tmp_path))
    assert any("pfcp_heartbeat_response / observed" in fact.text for fact in data.facts)
    assert any(
        "不是 raised 故障告警" in gap and "不能由响应本身划定一次故障" in gap
        for gap in data.report.missing_information
    )
    assert data.report.hypotheses == []


def test_root_cause_question_without_log_does_not_imply_log_was_seen(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path)
    task = AgentTask(
        run_id=uuid4().hex,
        question="未提供日志，能确认 UPF 昨日故障根因吗？",
        task_type="diagnose",
    )
    data = run_agent(tmp_path, task, backend)
    assert data.report.status.value == "need_input"
    assert any("尚未提供可核对的日志事件" in gap for gap in data.report.missing_information)
    assert not any("当前日志事件与计数" in gap for gap in data.report.missing_information)


@pytest.mark.parametrize("behavior", ["empty", "timeout"])
def test_failed_model_keeps_proven_statistics(tmp_path: Path, behavior: str) -> None:
    backend = FakeBackend(tmp_path, behavior)
    data = run_agent(tmp_path, new_task(setup_log(tmp_path)), backend)
    assert data.degraded and data.report.status.value == "partial"
    assert data.facts and "模型" not in data.facts[0].text
    assert backend.models <= 2


def test_parameter_repair_once_and_unsafe_tool_never_executes(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path, "repair")
    data = run_agent(tmp_path, new_task(setup_log(tmp_path)), backend)
    assert "shell_exec" not in backend.executed
    assert metrics(tmp_path, data)["parameter_repairs"] == 1
    assert backend.executed == ["analyze_log_window", "find_log_events"]


def test_round_and_tool_budgets_are_actual_business_limits(tmp_path: Path) -> None:
    generation = setup_log(tmp_path)
    backend = FakeBackend(tmp_path, "rounds")
    data = run_agent(tmp_path, new_task(generation), backend)
    assert metrics(tmp_path, data)["rounds"] == 3
    assert backend.models == 3
    backend = FakeBackend(tmp_path)
    data = run_agent(tmp_path, new_task(generation, budget=AgentBudget(max_tool_calls=1)), backend)
    assert backend.executed == ["analyze_log_window"]
    assert metrics(tmp_path, data)["tool_attempts"] == 1


def test_duplicate_call_hits_cache_and_terminates_without_repeat_execution(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path, "repeat")
    data = run_agent(tmp_path, new_task(setup_log(tmp_path)), backend)
    assert backend.executed.count("find_log_events") == 1
    assert metrics(tmp_path, data)["tool_attempts"] == 2
    assert any("重复" in gap for gap in data.gaps)


def test_checkpoint_resume_keeps_receipts_and_other_task_evidence_separate(tmp_path: Path) -> None:
    generation = setup_log(tmp_path)
    backend = FakeBackend(tmp_path)
    task = new_task(generation)
    first = run_agent(tmp_path, task, backend, pause_after="execute")
    assert first.report.status.value == "need_input"
    resumed = resume_agent(tmp_path, task.run_id, backend=backend)
    assert backend.executed == ["analyze_log_window", "find_log_events"]
    assert resumed.report.status.value == "partial"
    other = run_agent(tmp_path, new_task(generation), FakeBackend(tmp_path))
    assert set(item.evidence_id for item in other.evidence).isdisjoint(
        item.evidence_id for item in resumed.evidence
    )


def test_missing_log_interrupt_can_attach_registered_source_and_resume(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path)
    task = new_task(None, interactive=True)
    first = run_agent(tmp_path, task, backend)
    assert first.report.status.value == "need_input"
    generation = setup_log(tmp_path)
    resumed = resume_agent(
        tmp_path, task.run_id, ResumeInput(log_generation=generation, answer="补充日志"), backend
    )
    assert resumed.facts[0].text.startswith("所选范围包含 2 条")
    assert resumed.task.log_generation == generation


def test_corrupt_log_is_failed_and_tool_retry_at_most_once(tmp_path: Path) -> None:
    generation = setup_log(tmp_path)
    backend = FakeBackend(tmp_path, "retry")
    data = run_agent(tmp_path, new_task(generation), backend)
    assert backend.executed.count("analyze_log_window") == 2
    assert metrics(tmp_path, data)["tool_attempts"] <= 8
    (tmp_path / "sample.log").write_text("corrupted")
    data = run_agent(tmp_path, new_task(generation), FakeBackend(tmp_path))
    assert data.report.status.value == "failed"
    assert data.facts == []


def test_noninteractive_missing_input_is_still_resumable(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path)
    task = new_task(None)
    first = run_agent(tmp_path, task, backend)
    assert first.report.status.value == "need_input"
    result = resume_agent(
        tmp_path, task.run_id, ResumeInput(log_generation=setup_log(tmp_path)), backend
    )
    assert result.facts and result.task.log_generation


def test_resume_same_source_timezone_and_narrowing_removes_old_scope_facts(tmp_path: Path) -> None:
    from datetime import datetime

    from upf_agent.contracts import TimeWindow

    generation = setup_log(tmp_path)
    backend = FakeBackend(tmp_path, "ask")
    task = new_task(generation)
    first = run_agent(tmp_path, task, backend)
    assert first.facts[0].text.startswith("所选范围包含 2 条")
    assert first.report.status.value == "need_input"
    corrected = ingest_log(tmp_path / "sample.log", tmp_path / "data/p4/logs", timezone_name="UTC")[
        "generation"
    ]
    window = TimeWindow(
        start=datetime.fromisoformat("2025-05-14T00:00:00+00:00"),
        end=datetime.fromisoformat("2025-05-14T00:00:01+00:00"),
        timezone_source="human",
    )
    result = resume_agent(
        tmp_path, task.run_id, ResumeInput(log_generation=corrected, time_window=window), backend
    )
    assert result.facts[0].text.startswith("所选范围包含 1 条")
    assert not any("包含 2 条" in fact.text for fact in result.facts)
    assert not any("时区未知" in gap for gap in result.gaps)


def test_resume_other_source_is_rejected_before_touching_checkpoint(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path, "ask")
    task = new_task(setup_log(tmp_path))
    run_agent(tmp_path, task, backend)
    source = tmp_path / "different.log"
    source.write_text("2025-05-14 00:00:00 INFO different source\n")
    different = ingest_log(source, tmp_path / "data/p4/logs")["generation"]
    with pytest.raises(ValueError, match="换绑"):
        resume_agent(tmp_path, task.run_id, ResumeInput(log_generation=different), backend)


def test_crash_after_repair_receipt_reuses_valid_cached_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from upf_agent.agent_store import Journal

    backend = FakeBackend(tmp_path, "repair")
    task = new_task(setup_log(tmp_path))
    original = Journal.save_model

    def save_and_interrupt(self, key, payload):
        original(self, key, payload)
        if key.endswith(":repair"):
            raise KeyboardInterrupt()

    monkeypatch.setattr(Journal, "save_model", save_and_interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_agent(tmp_path, task, backend)
    monkeypatch.setattr(Journal, "save_model", original)
    result = resume_agent(tmp_path, task.run_id, backend=backend)
    assert backend.executed == ["analyze_log_window", "find_log_events"]
    assert metrics(tmp_path, result)["parameter_repairs"] == 1
    assert backend.models == 3
