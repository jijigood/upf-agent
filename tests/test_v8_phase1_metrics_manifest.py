"""Counters must reflect dispatches and immutable freezes must retain dirty files."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from upf_agent.agent_contracts import ToolCall
from upf_agent.v8.experiment_manifest import freeze_worktree
from upf_agent.v8.experiment_metrics import MeasuredBackend, MetricCollector
from upf_agent.v8.fixtures import example
from upf_agent.v8.phase1 import prepare_synthetic_root
from upf_agent.v8.store import BudgetStop, Ledger


def test_ledger_real_dispatch_reuse_error_and_blocked_are_separate(tmp_path):
    task = example(tmp_path)
    task.budget.max_tool_calls = 1
    ledger = Ledger(tmp_path)
    try:
        ledger.create(task)
        request = {"tool": "read_observation", "object_id": "UPF-A"}
        assert ledger.charge(task, "one", "tool", request) is None
        result = {"bindings": [], "gaps": ["tool_failed:TimeoutError"]}
        ledger.commit_receipt(task, "one", result, [], 0.1)
        ledger.commit_receipt(task, "one", result, [], 0.1)
        assert ledger.charge(task, "one", "tool", request) == result
        with pytest.raises(BudgetStop):
            ledger.charge(task, "two", "tool", request)
        summary = ledger.metrics(task.run_id)
        assert summary["invocations"]["tool"] == {
            "attempted": 1,
            "committed": 1,
            "reused": 1,
            "error": 1,
            "blocked": 1,
        }
        assert summary["coverage_complete"]
        assert ledger.inspect(task.run_id)["elapsed_seconds"] == 0.1
        assert summary["source_reads"] is None
    finally:
        ledger.close()


def test_uncertain_reserved_call_is_never_counted_as_a_second_dispatch(tmp_path):
    task = example(tmp_path)
    ledger = Ledger(tmp_path)
    try:
        ledger.create(task)
        ledger.charge(task, "uncertain", "model", {})
        ledger.note_error(task.run_id, "uncertain", TimeoutError())
        with pytest.raises(BudgetStop, match="uncommitted"):
            ledger.charge(task, "uncertain", "model", {})
        summary = ledger.metrics(task.run_id)
        assert summary["invocations"]["model"]["attempted"] == 1
        assert summary["invocations"]["model"]["committed"] == 0
        assert summary["token_usage"]["input"] is None
    finally:
        ledger.close()


def test_event_insert_rolls_back_with_receipt(tmp_path, monkeypatch):
    task = example(tmp_path)
    ledger = Ledger(tmp_path)
    try:
        ledger.create(task)
        original = ledger._metric_event

        def crash(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("simulated transaction failure")

        monkeypatch.setattr(ledger, "_metric_event", crash)
        with pytest.raises(RuntimeError):
            ledger.charge(task, "one", "tool", {})
        assert ledger.inspect(task.run_id)["receipts"] == []
        assert ledger.metric_events(task.run_id) == []
    finally:
        ledger.close()


def test_legacy_receipts_are_reported_as_incomplete_coverage(tmp_path):
    task = example(tmp_path)
    ledger = Ledger(tmp_path)
    try:
        ledger.create(task)
        ledger.charge(task, "old", "model", {})
        ledger.commit_receipt(task, "old", {}, [], 0)
        ledger.db.execute("DELETE FROM metric_events")
        ledger.db.commit()
        summary = ledger.metrics(task.run_id)
        assert not summary["coverage_complete"]
        assert summary["historical_receipts_without_attempt_events"] == 1
        assert summary["token_usage"]["input"] is None
    finally:
        ledger.close()


def test_backend_wrapper_preserves_output_and_missing_token_telemetry():
    reply = {"content": "frozen response", "usage": {"requests": 1}}
    backend = SimpleNamespace(model=lambda *args: reply)
    collector = MetricCollector("case")
    measured = MeasuredBackend(backend, collector)
    assert (
        measured.model("plan", SimpleNamespace(task=SimpleNamespace(run_id="case")), 1, 1) is reply
    )
    assert collector.summary()["invocations"]["model"]["attempted"] == 1
    assert collector.summary()["token_usage"]["input"] is None


def test_backend_exception_and_failure_packet_are_counted():
    collector = MetricCollector("case")
    failure = SimpleNamespace(error_code="timeout")
    backend = SimpleNamespace(tool=lambda *args: failure)
    wrapper = MeasuredBackend(backend, collector)
    call = ToolCall(name="search_manual", arguments={"query": "UPF"})
    assert wrapper.tool(None, call, [], 1) is failure
    assert collector.summary()["invocations"]["tool"]["error"] == 1

    def crash(*args):
        raise TimeoutError("test timeout")

    backend.tool = crash
    with pytest.raises(TimeoutError):
        wrapper.tool(None, call, [], 1)
    counts = collector.summary()["invocations"]["tool"]
    assert counts["attempted"] == 2
    assert counts["committed"] == 1
    assert counts["error"] == 2


def test_known_tokens_are_aggregated_only_when_complete():
    collector = MetricCollector("case")
    with collector.measure("model", "plan", {}) as receipt:
        receipt.update(input_tokens=10, output_tokens=3)
    assert collector.summary()["token_usage"]["input"] == 10
    with collector.measure("model", "plan", {}):
        pass
    assert collector.summary()["token_usage"]["input"] is None
    assert collector.summary()["token_usage"]["known_model_receipts"] == 1


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)


def test_freeze_preserves_staged_unstaged_untracked_and_deleted(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    git(project, "init")
    (project / "tracked.txt").write_text("base\n")
    (project / "deleted.txt").write_text("base\n")
    git(project, "add", ".")
    git(
        project,
        "-c",
        "user.name=phase1",
        "-c",
        "user.email=phase1@example.invalid",
        "commit",
        "-m",
        "base",
    )
    (project / "tracked.txt").write_text("staged\n")
    git(project, "add", "tracked.txt")
    (project / "tracked.txt").write_text("unstaged\n")
    (project / "deleted.txt").unlink()
    (project / "新文件.json").write_text('{"new": true}\n')
    (project / "pointer").symlink_to("tracked.txt")
    before_index = git(project, "show", ":tracked.txt")
    frozen = tmp_path / "frozen"
    manifest = freeze_worktree(project, frozen)
    assert git(project, "show", ":tracked.txt") == before_index
    assert (frozen / "worktree/tracked.txt").read_text() == "unstaged\n"
    assert (frozen / "staged.patch").read_text().find("staged") >= 0
    assert (frozen / "unstaged.patch").read_text().find("unstaged") >= 0
    assert manifest["files"]["新文件.json"]["untracked"]
    assert manifest["files"]["deleted.txt"]["kind"] == "deleted"
    assert (frozen / "worktree/pointer").is_symlink()
    assert json.loads((frozen / "manifest.json").read_text())["files"] == manifest["files"]
    with pytest.raises(ValueError, match="already exists"):
        freeze_worktree(project, frozen)
    with pytest.raises(ValueError, match="inside the project"):
        freeze_worktree(project, tmp_path / "another", extra_files=["../outside"])


def test_minimal_synthetic_root_has_no_model_or_private_corpus_dependency(tmp_path):
    from upf_agent.v8.engine import OrchestrationEngine

    root = tmp_path / "synthetic"
    task = prepare_synthetic_root(root)
    engine = OrchestrationEngine(root)
    try:
        result = engine.run(task)
        assert result["status"] == "completed"
        folder = root / "runs/v8" / task.run_id
        metrics = json.loads((folder / "metrics.json").read_text())
        assert metrics["invocations"]["tool"]["attempted"] == 3
        assert metrics["invocations"]["model"]["attempted"] == 0
        assert metrics["coverage_complete"]
        assert not (root / "data/raw").exists()
        assert all(not a["execute_authorized"] for a in result["actions"])
    finally:
        engine.close()
    with pytest.raises(ValueError, match="already exists"):
        prepare_synthetic_root(root)


def test_phase1_cannot_attest_a_different_import_root(tmp_path):
    from upf_agent.v8.phase1 import run_phase1

    output = tmp_path / "output"
    with pytest.raises(ValueError, match="imported code root"):
        run_phase1(tmp_path / "different-project", output)
    assert not output.exists()


def test_idempotency_key_cannot_change_counter_kind(tmp_path):
    task = example(tmp_path)
    ledger = Ledger(tmp_path)
    try:
        ledger.create(task)
        ledger.charge(task, "same-key", "tool", {})
        ledger.commit_receipt(task, "same-key", {}, [], 0)
        with pytest.raises(ValueError, match="different request or kind"):
            ledger.charge(task, "same-key", "model", {})
        assert ledger.metrics(task.run_id)["invocations"]["model"]["reused"] == 0
    finally:
        ledger.close()
