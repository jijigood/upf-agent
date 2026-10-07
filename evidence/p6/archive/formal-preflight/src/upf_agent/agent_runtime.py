"""Cancelable local worker boundary, protecting the task deadline and child processes."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from upf_agent.agent_contracts import AgentData, AgentTask, ToolCall, ToolPacket, VerifiedEvidence
from upf_agent.agent_tools import failure


class Backend(Protocol):
    def tool(
        self, task: AgentTask, call: ToolCall, known: list[VerifiedEvidence], timeout: float
    ) -> ToolPacket: ...
    def model(
        self,
        operation: str,
        data: AgentData,
        timeout: float,
        allowance: int,
        repair: str | None = None,
    ) -> dict[str, JsonValue]: ...


class WorkerError(RuntimeError):
    pass


class LocalBackend:
    def __init__(self, root: Path) -> None:
        self.root = root

    def request(self, request: dict[str, JsonValue], timeout: float) -> dict[str, JsonValue]:
        if timeout <= 0:
            raise TimeoutError("task deadline exhausted")
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
                "LANGSMITH_TRACING": "false",
                "LANGCHAIN_TRACING_V2": "false",
            }
        )
        logs = self.root / "runs/p5"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / "workers.log").open("a", encoding="utf-8") as errors:
            process = subprocess.Popen(
                [sys.executable, "-m", "upf_agent.agent_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                cwd=self.root,
                env=env,
                start_new_session=True,
            )
            try:
                output, _ = process.communicate(
                    json.dumps({"root": str(self.root), **request}), timeout=timeout
                )
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # Only the newly created worker process group is stopped.
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise TimeoutError("local worker exceeded task deadline") from None
        if process.returncode:
            raise WorkerError(f"worker exit={process.returncode}; see runs/p5/workers.log")
        try:
            value = json.loads(output)
            if not isinstance(value, dict):
                raise ValueError("worker output is not an object")
            if "worker_error" in value:
                raise WorkerError(str(value["worker_error"]))
            return value
        except (ValueError, json.JSONDecodeError) as error:
            raise WorkerError("invalid/empty local worker response") from error

    def tool(
        self, task: AgentTask, call: ToolCall, known: list[VerifiedEvidence], timeout: float
    ) -> ToolPacket:
        try:
            response = self.request(
                {
                    "operation": "tool",
                    "task": task.model_dump(mode="json"),
                    "call": call.model_dump(mode="json"),
                    "known": [item.model_dump(mode="json") for item in known],
                },
                timeout,
            )
            return ToolPacket.model_validate_json(json.dumps(response))
        except TimeoutError as error:
            return failure("timeout", str(error))
        except WorkerError as error:
            return failure("tool_failed", str(error), retryable=True)

    def model(
        self,
        operation: str,
        data: AgentData,
        timeout: float,
        allowance: int,
        repair: str | None = None,
    ) -> dict[str, JsonValue]:
        return self.request(
            {
                "operation": operation,
                "state": data.model_dump(mode="json"),
                "allowance": allowance,
                "repair": repair,
                "timeout": timeout,
            },
            timeout,
        )
