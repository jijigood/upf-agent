"""A terminal session reuses both retrieval workers across questions."""

import json

from typer.testing import CliRunner

from upf_agent import cli


def test_manual_chat_reuses_retrieval_workers(monkeypatch):
    created = []
    received = []

    class Worker:
        def __init__(self, _root, task, _path, _device):
            self.task = task
            created.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_errors):
            return None

    def answer(_root, question, _product, _version, _device, _model, workers):
        received.append((question, workers))
        return {"status": "completed", "run_id": question}

    monkeypatch.setattr(cli, "RetrievalWorker", Worker)
    monkeypatch.setattr(cli, "ask", answer)
    result = CliRunner().invoke(cli.app, ["manual-chat", "--jsonl"], input="first\nsecond\n/exit\n")

    assert result.exit_code == 0, result.output
    assert [worker.task for worker in created] == ["embed", "rerank"]
    assert [question for question, _workers in received] == ["first", "second"]
    assert all(workers == tuple(created) for _question, workers in received)
    lines = [json.loads(line) for line in result.output.splitlines()]
    assert lines[0] == {"event": "ready"}
    assert [line["run_id"] for line in lines[1:]] == ["first", "second"]
