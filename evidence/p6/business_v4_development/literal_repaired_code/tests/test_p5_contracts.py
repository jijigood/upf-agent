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
