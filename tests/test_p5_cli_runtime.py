import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from upf_agent.agent_cli import app
from upf_agent.agent_runtime import LocalBackend
from upf_agent.log_tools import ingest_log


def log_id(root: Path) -> str:
    source = root / "a.log"
    source.write_text("2025-05-14 00:00:00 INFO Recv #MsgHeartbeat# Req from cp.\n")
    return ingest_log(source, root / "data/p4/logs")["generation"]


def test_jsonl_failure_does_not_skip_later_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("upf_agent.agent_cli.ROOT", tmp_path)
    generation = log_id(tmp_path)
    source = tmp_path / "input.jsonl"
    source.write_text(
        "invalid JSON\n"
        + json.dumps(
            {"question": "统计日志数量", "log_generation": generation, "mode": "deterministic"}
        )
        + "\n"
    )
    output = tmp_path / "batch"
    result = CliRunner().invoke(app, ["batch", "--input", str(source), "--output", str(output)])
    assert result.exit_code == 1
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["status"] == "failed" and rows[1]["status"] == "partial"
    assert Path(rows[1]["report"]).is_file()


def test_txt_and_stdin_inputs_and_partial_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("upf_agent.agent_cli.ROOT", tmp_path)
    generation = log_id(tmp_path)
    question = tmp_path / "question.txt"
    question.write_text("统计日志消息数量")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "diagnose",
            "--log-generation",
            generation,
            "--question-file",
            str(question),
            "--deterministic",
        ],
    )
    assert result.exit_code == 2, result.output
    run = json.loads(result.output)
    assert run["facts"][0].startswith("所选范围包含 1 条")
    result = runner.invoke(app, ["investigate", "--deterministic"], input="当前CPU温度\n")
    assert result.exit_code == 2, result.output
    assert json.loads(result.output)["status"] == "missing_source"


def test_chat_retains_confirmed_parameters_but_new_task_evidence_is_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("upf_agent.agent_cli.ROOT", tmp_path)
    generation = log_id(tmp_path)
    result = CliRunner().invoke(
        app,
        ["chat", "--session", "test", "--log-generation", generation, "--deterministic"],
        input="统计日志数量\n统计日志数量\n/exit\n",
    )
    assert result.exit_code == 0, result.output
    reports = [
        json.loads(path.read_text()) for path in (tmp_path / "runs/p5").glob("*/report.json")
    ]
    assert len(reports) == 2
    assert set(reports[0]["supporting_evidence_ids"]).isdisjoint(
        reports[1]["supporting_evidence_ids"]
    )


def test_worker_timeout_stops_its_own_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = subprocess.Popen
    child_id = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
        f"Path({str(child_id)!r}).write_text(str(child.pid)); time.sleep(60)"
    )

    def launch(args, **kwargs):
        return original([sys.executable, "-c", code], **kwargs)

    monkeypatch.setattr("upf_agent.agent_runtime.subprocess.Popen", launch)
    with pytest.raises(TimeoutError):
        LocalBackend(tmp_path).request({"operation": "tool"}, 0.5)
    pid = int(child_id.read_text())
    state = Path(f"/proc/{pid}/stat")
    try:
        assert state.read_text().split()[2] == "Z"
    except (FileNotFoundError, ProcessLookupError):
        pass
