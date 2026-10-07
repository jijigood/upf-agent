"""Synchronous JSONL client for the independently locked retrieval environment."""

import json
import os
import select
import subprocess
from pathlib import Path
from typing import Literal
from uuid import uuid4


class RetrievalWorkerError(RuntimeError):
    pass


class RetrievalWorker:
    def __init__(
        self,
        project_root: Path,
        task: Literal["embed", "rerank"],
        model_path: Path,
        device: str = "cpu",
        startup_timeout: int = 600,
    ) -> None:
        interpreter = project_root / ".venv-retrieval" / "bin" / "python"
        if not interpreter.is_file():
            raise RetrievalWorkerError(f"retrieval environment missing: {interpreter}")
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
            }
        )
        log_dir = project_root / "runs" / "p1"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.error_log = log_dir / f"retrieval-worker-{task}.log"
        error_stream = self.error_log.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(interpreter),
                str(project_root / "tools" / "retrieval_worker.py"),
                "--task",
                task,
                "--model",
                str(model_path),
                "--device",
                device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=error_stream,
            text=True,
            bufsize=1,
            env=environment,
        )
        error_stream.close()
        if self.process.stdout is None:
            raise RetrievalWorkerError("retrieval worker stdout unavailable")
        if not select.select([self.process.stdout], [], [], startup_timeout)[0]:
            self.close()
            raise RetrievalWorkerError("retrieval worker startup timed out")
        ready_line = self.process.stdout.readline()
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as error:
            self.close()
            raise RetrievalWorkerError(
                f"retrieval worker failed before readiness; see {self.error_log}: {ready_line}"
            ) from error
        if not ready.get("ready") or ready.get("task") != task:
            self.close()
            raise RetrievalWorkerError(f"retrieval worker not ready: {ready}")

    def request(self, body: dict[str, object], timeout: int = 300) -> dict[str, object]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RetrievalWorkerError("retrieval worker pipes unavailable")
        request_id = uuid4().hex
        payload = {"request_id": request_id, **body}
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            raise RetrievalWorkerError("retrieval worker request timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RetrievalWorkerError("retrieval worker exited during request")
        result = json.loads(line)
        if result.get("request_id") != request_id:
            raise RetrievalWorkerError("retrieval worker request id mismatch")
        if "error" in result:
            raise RetrievalWorkerError(str(result["error"]))
        return result

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)

    def __enter__(self) -> "RetrievalWorker":
        return self

    def __exit__(self, *_errors: object) -> None:
        self.close()
