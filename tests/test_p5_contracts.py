import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from upf_agent.agent_contracts import AgentBudget, AgentTask, ToolCall
from upf_agent.agent_tools import validate_call


def task() -> AgentTask:
    return AgentTask(
        run_id=uuid4().hex, question="分析日志", task_type="diagnose", log_generation="a" * 20
    )


@pytest.mark.parametrize("index", range(100))
def test_fixed_tool_contract_variants(index: int) -> None:
    current = task()
    family = index // 10
    arguments = {"log_id": "a" * 20, "limit": index % 10 + 1}
    call = ToolCall(name="find_log_events", arguments=arguments)
    if family == 0:
        expected = None
    elif family == 1:
        call.name, expected = "shell_exec", "unknown_tool"
    elif family == 2:
        call.arguments["log_id"], expected = "b" * 20, "scope_denied"
    elif family == 3:
        call.arguments["limit"], expected = str(index), "invalid_arguments"
    elif family == 4:
        call.arguments["limit"], expected = 10001 + index, "invalid_arguments"
    elif family == 5:
        call.arguments["path"], expected = "/etc/passwd", "invalid_arguments"
    elif family == 6:
        del call.arguments["log_id"]
        expected = "invalid_arguments"
    elif family == 7:
        call.arguments["limit"], expected = None, "invalid_arguments"
    elif family == 8:
        call.arguments["limit"], expected = True, "invalid_arguments"
    else:
        call.arguments["event_type"], expected = "';DROP TABLE events;--中文" + str(index), None
    _, error = validate_call(current, call, [])
    assert error == expected
    # This is a fixed code/schema suite, not a model parameter accuracy measurement.
    assert json.loads(call.model_dump_json())["name"] == call.name


def test_hard_budget_and_generation_limits() -> None:
    with pytest.raises(ValidationError):
        AgentBudget(max_tool_calls=9)
    with pytest.raises(ValidationError):
        AgentBudget(max_rounds=4)
    with pytest.raises(ValidationError):
        AgentTask(run_id="../other", question="x", task_type="manual")


def test_log_count_report_includes_requested_raw_samples_with_states(tmp_path):
    from upf_agent.agent_tools import execute_tool
    from upf_agent.log_tools import ingest_log

    source = tmp_path / "source.log"
    raw = (
        "2025-05-14 00:00:00 ERROR #Alarm# report PFCP node unreachable!\n"
        "2025-05-14 00:00:01 ERROR #Alarm# clear PFCP node unreachable.\n"
        "2025-05-14 00:00:02 DEBUG Recv #MsgHeartbeat# Req from cp.\n"
    )
    source.write_text(raw)
    manifest = ingest_log(source, tmp_path / "data/p4/logs")
    current = AgentTask(
        run_id=uuid4().hex,
        question="统计pfcp_node_unreachable并提供代表日志原文",
        task_type="diagnose",
        log_generation=manifest["generation"],
    )
    call = ToolCall(name="analyze_log_window", arguments={"log_id": manifest["generation"]})
    packet = execute_tool(tmp_path, current, call, [])
    assert packet.result.data["total_events"] == 3
    samples = [entry for entry in packet.evidence if entry.kind == "log"]
    assert {entry.metadata["event_state"] for entry in samples} == {"raised", "cleared"}
    assert len(samples) == 2
    assert all(entry.excerpt in raw and "lines=" in entry.locator for entry in samples)
    assert all(entry.metadata["event_type"] == "pfcp_node_unreachable" for entry in samples)
    text = "\n".join(fact.text for fact in packet.facts)
    assert "report PFCP node unreachable!" in text
    assert "clear PFCP node unreachable." in text
    assert "Recv #MsgHeartbeat#" not in text
    assert "样例" in text


def test_log_samples_do_not_escape_confirmed_time_window(tmp_path):
    from datetime import datetime

    from upf_agent.agent_tools import execute_tool
    from upf_agent.contracts import TimeWindow
    from upf_agent.log_tools import ingest_log

    source = tmp_path / "source.log"
    source.write_text(
        "2025-05-14 00:00:00 ERROR #Alarm# report PFCP node unreachable!\n"
        "2025-05-14 00:00:01 ERROR #Alarm# clear PFCP node unreachable.\n"
    )
    manifest = ingest_log(source, tmp_path / "data/p4/logs", timezone_name="UTC")
    current = AgentTask(
        run_id=uuid4().hex,
        question="pfcp_node_unreachable 原文",
        task_type="diagnose",
        log_generation=manifest["generation"],
        time_window=TimeWindow(
            start=datetime.fromisoformat("2025-05-14T00:00:01+00:00"),
            end=datetime.fromisoformat("2025-05-14T00:00:02+00:00"),
            timezone_source="user",
        ),
    )
    packet = execute_tool(
        tmp_path,
        current,
        ToolCall(name="analyze_log_window", arguments={"log_id": manifest["generation"]}),
        [],
    )
    samples = [entry for entry in packet.evidence if entry.kind == "log"]
    assert len(samples) == 1
    assert samples[0].metadata["event_state"] == "cleared"
    assert packet.result.data["total_events"] == 1
