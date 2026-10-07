"""Terminal/TXT/stdin, JSONL batches and local sessions for the finite agent."""

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from rich.console import Console

from upf_agent.agent_contracts import AgentBudget, AgentData, AgentTask, ResumeInput
from upf_agent.agent_engine import resume_agent, run_agent
from upf_agent.contracts import ResultStatus, TimeWindow
from upf_agent.log_tools import ingest_log, write_json

ROOT = Path(__file__).resolve().parents[2]
console = Console()
app = typer.Typer()


def parse_window(start: str | None, end: str | None) -> TimeWindow | None:
    if bool(start) != bool(end):
        raise ValueError("--start/--end 必须同时提供，且包含明确 UTC offset。")
    return (
        TimeWindow.model_validate_json(
            json.dumps({"start": start, "end": end, "timezone_source": "user_explicit_offset"})
        )
        if start
        else None
    )


def question_input(question: str | None, file: Path | None, default: str | None = None) -> str:
    if question is not None and file is not None:
        raise ValueError("问题文本与 --question-file/--file 只选一种。")
    if file:
        with file.open(encoding="utf-8") as handle:
            question = handle.read(2001)
    elif question is None and not sys.stdin.isatty():
        question = sys.stdin.read(2001)
    value = (question or default or "").strip()
    if not value or len(value) > 2000:
        raise ValueError("问题应为 1–2000 字符的 UTF-8 文本。")
    return value


def emit(data: AgentData, output: Path | None = None) -> None:
    original = ROOT / "runs/p5" / data.task.run_id
    if output and output.resolve() != original.resolve():
        if output.exists() and any(output.iterdir()):
            manifest = output / "manifest.json"
            if (
                not manifest.exists()
                or json.loads(manifest.read_text())["task"]["run_id"] != data.task.run_id
            ):
                raise ValueError("输出目录已有其他内容，请使用新目录。")
        shutil.copytree(original, output, dirs_exist_ok=True)
    console.print_json(
        data={
            "run_id": data.task.run_id,
            "status": data.report.status.value,
            "report": str((output or original) / "report.md"),
            "facts": data.report.facts,
            "missing_information": data.report.missing_information,
        }
    )


def exit_for(data: AgentData) -> None:
    if data.report.status == ResultStatus.FAILED:
        raise typer.Exit(1)
    if data.report.status != ResultStatus.COMPLETED:
        raise typer.Exit(2)


@app.command("diagnose")
def diagnose_command(
    question: Annotated[str | None, typer.Argument()] = None,
    log: Annotated[Path | None, typer.Option()] = None,
    log_generation: Annotated[str | None, typer.Option()] = None,
    topology_generation: Annotated[str | None, typer.Option()] = None,
    question_file: Annotated[Path | None, typer.Option("--question-file", "--file")] = None,
    log_timezone: Annotated[str | None, typer.Option()] = None,
    device_id: Annotated[str | None, typer.Option()] = None,
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option()] = None,
    model: Annotated[str, typer.Option()] = "baseline",
    deterministic: Annotated[bool, typer.Option()] = False,
    interactive: Annotated[bool, typer.Option()] = False,
    timeout: Annotated[float, typer.Option(min=0.01, max=180)] = 180.0,
    pause_after: Annotated[
        str | None, typer.Option(help="seed/execute: save and stop before continuing")
    ] = None,
) -> None:
    try:
        if log and log_generation:
            raise ValueError("--log 和 --log-generation 只选一种。")
        if log_timezone and not log:
            raise ValueError("--log-timezone 只能用于重新导入 --log，不修改既有 generation。")
        if pause_after not in (None, "seed", "execute"):
            raise ValueError("--pause-after 只接受 seed/execute。")
        if log:
            log_generation = ingest_log(log, ROOT / "data/p4/logs", log_timezone, device_id)[
                "generation"
            ]
        task = AgentTask(
            run_id=uuid4().hex,
            question=question_input(question, question_file, "分析日志事件及证据缺口"),
            task_type="diagnose",
            log_generation=log_generation,
            topology_generation=topology_generation,
            device_ids=[device_id] if device_id else [],
            time_window=parse_window(start, end),
            model_profile=model,
            mode="deterministic" if deterministic else "agent",
            interactive=interactive,
            budget=AgentBudget(timeout_seconds=timeout),
        )
        data = run_agent(ROOT, task, pause_after=pause_after)
        emit(data, output)
    except (OSError, ValueError, RuntimeError) as error:
        console.print(f"诊断失败：{error}")
        raise typer.Exit(1) from error
    exit_for(data)


@app.command("investigate")
def investigate_command(
    question: Annotated[str | None, typer.Argument()] = None,
    file: Annotated[Path | None, typer.Option()] = None,
    product: Annotated[str | None, typer.Option()] = None,
    version: Annotated[str | None, typer.Option()] = None,
    log_generation: Annotated[str | None, typer.Option()] = None,
    topology_generation: Annotated[str | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option()] = None,
    model: Annotated[str, typer.Option()] = "baseline",
    device: Annotated[str, typer.Option()] = "cuda:1",
    deterministic: Annotated[bool, typer.Option()] = False,
) -> None:
    try:
        task = AgentTask(
            run_id=uuid4().hex,
            question=question_input(question, file),
            task_type="other",
            product=product,
            version=version,
            log_generation=log_generation,
            topology_generation=topology_generation,
            model_profile=model,
            retrieval_device=device,
            mode="deterministic" if deterministic else "agent",
        )
        data = run_agent(ROOT, task)
        emit(data, output)
    except (OSError, ValueError, RuntimeError) as error:
        console.print(f"任务失败：{error}")
        raise typer.Exit(1) from error
    exit_for(data)


@app.command("resume")
def resume_command(
    run_id: str,
    answer: Annotated[str | None, typer.Option()] = None,
    log_generation: Annotated[str | None, typer.Option()] = None,
    topology_generation: Annotated[str | None, typer.Option()] = None,
    device_id: Annotated[str | None, typer.Option()] = None,
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    try:
        supplied = any(
            value is not None
            for value in (answer, log_generation, topology_generation, device_id, start, end)
        )
        update = (
            ResumeInput(
                answer=answer or "",
                log_generation=log_generation,
                topology_generation=topology_generation,
                device_ids=[device_id] if device_id else None,
                time_window=parse_window(start, end),
            )
            if supplied
            else None
        )
        data = resume_agent(ROOT, run_id, update)
        emit(data, output)
    except (OSError, ValueError, RuntimeError) as error:
        console.print(f"恢复失败：{error}")
        raise typer.Exit(1) from error
    exit_for(data)


@app.command("batch")
def batch_command(
    input: Annotated[Path, typer.Option()], output: Annotated[Path, typer.Option()]
) -> None:
    if output.exists() and any(output.iterdir()):
        raise typer.BadParameter("批处理输出必须使用空目录。")
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    rows = output / "results.jsonl"
    with input.open("rb") as source, rows.open("w", encoding="utf-8") as target:
        line_number = 0
        while raw := source.readline(16385):
            line_number += 1
            run_id = uuid4().hex
            try:
                if len(raw) > 16384:
                    while not raw.endswith(b"\n"):
                        raw = source.readline(16385)
                        if not raw:
                            break
                    raise ValueError("JSONL 单条超过 16KiB。")
                item = json.loads(raw.decode("utf-8"))
                if not isinstance(item, dict) or set(item) & {
                    "run_id",
                    "interactive",
                    "session_id",
                }:
                    raise ValueError("JSONL 不接受任务身份覆盖或交互字段。")
                task = AgentTask.model_validate_json(
                    json.dumps({"run_id": run_id, "task_type": "other", **item})
                )
                data = run_agent(ROOT, task)
                destination = output / f"{line_number:04d}-{run_id}"
                shutil.copytree(ROOT / "runs/p5" / run_id, destination)
                result = {
                    "line": line_number,
                    "run_id": run_id,
                    "status": data.report.status.value,
                    "report": str(destination / "report.json"),
                }
            except (OSError, ValueError, RuntimeError) as error:
                result = {
                    "line": line_number,
                    "run_id": run_id,
                    "status": "failed",
                    "error": str(error),
                }
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
    write_json(output / "summary.json", {"items": line_number, "status_counts": counts})
    console.print_json(data={"items": line_number, "status_counts": counts, "results": str(rows)})
    if counts.get("failed"):
        raise typer.Exit(1)
    if any(key != "completed" for key in counts):
        raise typer.Exit(2)


@app.command("chat")
def chat_command(
    session: Annotated[str, typer.Option()] = "local",
    log_generation: Annotated[str | None, typer.Option()] = None,
    product: Annotated[str | None, typer.Option()] = None,
    version: Annotated[str | None, typer.Option()] = None,
    deterministic: Annotated[bool, typer.Option()] = False,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session):
        raise typer.BadParameter("session 只接受字母、数字、下划线与连字符。")
    path = ROOT / "data/p5/sessions" / f"{session}.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    defaults = {
        "product": product or saved.get("product"),
        "version": version or saved.get("version"),
        "log_generation": log_generation or saved.get("log_generation"),
    }
    pending = saved.get("pending_run_id")
    console.print("输入问题；/exit 退出；/new 开始新任务。等待补充时可输入 ResumeInput JSON。")
    while True:
        try:
            text = input("UPF> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/exit":
            break
        if text == "/new":
            pending = None
            continue
        if not text:
            continue
        try:
            if pending:
                update = (
                    ResumeInput.model_validate_json(text)
                    if text.startswith("{")
                    else ResumeInput(answer=text)
                )
                data = resume_agent(ROOT, pending, update)
            else:
                task = AgentTask(
                    run_id=uuid4().hex,
                    question=text,
                    task_type="other",
                    interactive=True,
                    session_id=session,
                    mode="deterministic" if deterministic else "agent",
                    **defaults,
                )
                data = run_agent(ROOT, task)
            emit(data)
            pending = data.task.run_id if data.report.status == ResultStatus.NEED_INPUT else None
            defaults = {name: getattr(data.task, name) for name in defaults}
            write_json(
                path,
                {
                    **defaults,
                    "pending_run_id": pending,
                    "last_run_id": data.task.run_id,
                    "memory_policy": "confirmed_parameters_only_no_previous_task_evidence",
                },
            )
        except (OSError, ValueError, RuntimeError) as error:
            console.print(f"会话操作失败：{error}")
