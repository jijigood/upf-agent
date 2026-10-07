import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from upf_agent.p4_cli import app


def test_terminal_import_statistics_and_context_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("upf_agent.p4_cli.ROOT", tmp_path)
    source = tmp_path / "input.log"
    source.write_text("2025-05-14 00:00:00 INFO Recv #MsgHeartbeat# Req from cp.\n")
    runner = CliRunner()
    imported = runner.invoke(app, ["import-log", str(source), "--timezone", "UTC"])
    assert imported.exit_code == 0, imported.output
    generation = json.loads(imported.output)["generation"]
    found = runner.invoke(
        app, ["find-events", generation, "--event-type", "pfcp_heartbeat_request"]
    )
    assert found.exit_code == 0, found.output
    event = json.loads(found.output)["data"]["events"][0]
    context = runner.invoke(
        app, ["read-context", generation, event["event_id"], "--before", "0", "--after", "0"]
    )
    assert context.exit_code == 0, context.output
    assert json.loads(context.output)["data"]["raw_text"] == source.read_text()
    summary = runner.invoke(app, ["analyze-log", generation])
    assert summary.exit_code == 0, summary.output
    assert json.loads(summary.output)["data"]["total_events"] == 1
