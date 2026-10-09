"""Transactional evidence revisions, immutable batches and idempotent budget receipts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from time import time

from upf_agent.v8.experiment_metrics import InvocationEvent, summarize_events
from upf_agent.v8.models import (
    EvidenceBinding,
    OrchestrationData,
    OrchestrationTask,
    canonical,
    digest,
)


class BudgetStop(RuntimeError):
    pass


def execution_fingerprint(root: Path) -> str:
    project = Path(__file__).resolve().parents[3]
    paths = [
        *sorted(Path(__file__).parent.glob("*.py")),
        project / "uv.lock",
        root / "config/models.yaml",
        root / "config/retrieval.yaml",
        root / "data/v8/knowledge_applicability.json",
    ]
    return digest(
        {
            (
                f"code/{p.name}"
                if p.parent == Path(__file__).parent
                else str(p.relative_to(root))
                if p.is_relative_to(root)
                else "uv.lock"
            ): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
            if p.is_file()
        }
    )


class Ledger:
    def __init__(self, root: Path):
        self.root = root
        path = root / "data/v8/ledger.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
          run_id TEXT PRIMARY KEY, task TEXT NOT NULL, fingerprint TEXT NOT NULL,
          created REAL NOT NULL, elapsed REAL NOT NULL DEFAULT 0, data TEXT);
        CREATE TABLE IF NOT EXISTS receipts (
          run_id TEXT, key TEXT, kind TEXT NOT NULL, request TEXT NOT NULL,
          status TEXT NOT NULL, result TEXT, PRIMARY KEY(run_id,key));
        CREATE TABLE IF NOT EXISTS evidence (
          run_id TEXT, id TEXT, revision INTEGER, payload TEXT NOT NULL,
          PRIMARY KEY(run_id,id,revision));
        CREATE TABLE IF NOT EXISTS batches (
          run_id TEXT, batch_id TEXT, snapshot TEXT NOT NULL, result TEXT NOT NULL,
          PRIMARY KEY(run_id,batch_id));
        CREATE TABLE IF NOT EXISTS dependencies (
          run_id TEXT, batch_id TEXT, payload TEXT NOT NULL, PRIMARY KEY(run_id,batch_id));
        CREATE TABLE IF NOT EXISTS updates (
          run_id TEXT, batch_id TEXT, payload TEXT NOT NULL, PRIMARY KEY(run_id,batch_id));
        CREATE TABLE IF NOT EXISTS metric_events (
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
          event_id TEXT NOT NULL UNIQUE, payload TEXT NOT NULL);
        """)

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    @contextmanager
    def execution_lock(self, run_id: str):
        # A shared runtime lock permits independent runs but excludes snapshot copying.
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("invalid run_id")
        folder = self.root / "data/v8/locks"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "snapshot.lock").open("a") as runtime_lock:
            with (folder / f"{run_id}.lock").open("a") as run_lock:
                try:
                    fcntl.flock(runtime_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError("run or runtime snapshot is already active") from error
                try:
                    yield
                finally:
                    fcntl.flock(run_lock, fcntl.LOCK_UN)
                    fcntl.flock(runtime_lock, fcntl.LOCK_UN)

    def create(self, task: OrchestrationTask):
        with self.transaction():
            previous = self.db.execute(
                "SELECT fingerprint FROM runs WHERE run_id=?", (task.run_id,)
            ).fetchone()
            if previous and previous[0] != digest(task):
                raise ValueError(
                    "run identity cannot be rebound to another task or permission scope"
                )
            self.db.execute(
                "INSERT OR IGNORE INTO runs(run_id,task,fingerprint,created) VALUES(?,?,?,?)",
                (task.run_id, task.model_dump_json(), digest(task), time()),
            )

    def task(self, run_id: str) -> OrchestrationTask:
        row = self.db.execute("SELECT task FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("unknown run_id")
        return OrchestrationTask.model_validate_json(row[0])

    def charge(self, task: OrchestrationTask, key: str, kind: str, request: dict) -> dict | None:
        try:
            return self._charge(task, key, kind, request)
        except BudgetStop as error:
            with self.transaction():
                self._metric_event(
                    task.run_id, key, kind, request, "blocked", error_type=str(error)
                )
            raise

    def _charge(self, task, key, kind, request):
        if kind not in {"tool", "model"}:
            raise ValueError("unknown budget kind")
        with self.transaction():
            old = self.db.execute(
                "SELECT status,result,request,kind FROM receipts WHERE run_id=? AND key=?",
                (task.run_id, key),
            ).fetchone()
            if old:
                if old[2] != canonical(request) or old[3] != kind:
                    raise ValueError("idempotency key reused for a different request or kind")
                if old[0] == "committed":
                    self._metric_event(task.run_id, key, kind, request, "reused")
                    return json.loads(old[1])
                # A crash after reserving but before committing has an uncertain receipt.
                # Do not silently repeat collection or a model call.
                raise BudgetStop("uncommitted_receipt_requires_review")
            count = self.db.execute(
                "SELECT count(*) FROM receipts WHERE run_id=? AND kind=?", (task.run_id, kind)
            ).fetchone()[0]
            elapsed = self.db.execute(
                "SELECT elapsed FROM runs WHERE run_id=?", (task.run_id,)
            ).fetchone()[0]
            maximum = task.budget.max_tool_calls if kind == "tool" else task.budget.max_model_calls
            if count >= maximum:
                raise BudgetStop(f"{kind}_budget_exhausted")
            if elapsed >= task.budget.timeout_seconds:
                raise BudgetStop("task_timeout")
            self.db.execute(
                "INSERT INTO receipts VALUES(?,?,?,?,?,NULL)",
                (task.run_id, key, kind, canonical(request), "reserved"),
            )
            self._metric_event(task.run_id, key, kind, request, "attempted")
        return None

    def commit_receipt(
        self,
        task: OrchestrationTask,
        key: str,
        result: dict,
        bindings: list[EvidenceBinding],
        elapsed: float,
    ):
        with self.transaction():
            prior = self.db.execute(
                "SELECT kind,request,status,result FROM receipts WHERE run_id=? AND key=?",
                (task.run_id, key),
            ).fetchone()
            if prior is None:
                raise ValueError("cannot commit an unreserved receipt")
            for binding in bindings:
                if binding.task_id != task.run_id:
                    raise ValueError("cross-task evidence rejected")
                old = self.db.execute(
                    "SELECT payload FROM evidence WHERE run_id=? AND id=? AND revision=?",
                    (task.run_id, binding.evidence_id, binding.revision),
                ).fetchone()
                if old and old[0] != binding.model_dump_json():
                    raise ValueError("evidence revision collision")
                if prior[2] == "committed" and not old:
                    raise ValueError("cannot append evidence to a committed receipt")
            if prior[2] == "committed":
                if prior[3] != canonical(result):
                    raise ValueError("committed receipt cannot be changed")
                return
            for binding in bindings:
                self.db.execute(
                    "INSERT OR IGNORE INTO evidence VALUES(?,?,?,?)",
                    (task.run_id, binding.evidence_id, binding.revision, binding.model_dump_json()),
                )
            self.db.execute(
                "UPDATE receipts SET status='committed',result=? WHERE run_id=? AND key=?",
                (canonical(result), task.run_id, key),
            )
            self.db.execute(
                "UPDATE runs SET elapsed=elapsed+? WHERE run_id=?", (elapsed, task.run_id)
            )
            if prior:
                request = json.loads(prior[1])
                usage = result.get("usage") or {}
                self._metric_event(
                    task.run_id,
                    key,
                    prior[0],
                    request,
                    "committed",
                    duration_seconds=elapsed,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                )
                if any(str(g).startswith("tool_failed:") for g in result.get("gaps", [])):
                    self._metric_event(
                        task.run_id, key, prior[0], request, "error", error_type="tool_failed"
                    )

    def _metric_event(self, run_id, key, kind, request, status, **kwargs):
        event = InvocationEvent(
            run_id=run_id,
            invocation_id=key,
            kind=kind,
            operation=request.get("tool") or request.get("purpose") or "graph_query",
            status=status,
            request_sha256=digest(request),
            **kwargs,
        )
        self.db.execute(
            "INSERT INTO metric_events(run_id,event_id,payload) VALUES(?,?,?)",
            (run_id, event.event_id, event.model_dump_json()),
        )

    def note_error(self, run_id, key, error):
        with self.transaction():
            row = self.db.execute(
                "SELECT kind,request FROM receipts WHERE run_id=? AND key=? AND status='reserved'",
                (run_id, key),
            ).fetchone()
            if row:
                self._metric_event(
                    run_id,
                    key,
                    row[0],
                    json.loads(row[1]),
                    "error",
                    error_type=type(error).__name__,
                )

    def metric_events(self, run_id):
        return [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT payload FROM metric_events WHERE run_id=? ORDER BY sequence", (run_id,)
            )
        ]

    def metrics(self, run_id):
        events = self.metric_events(run_id)
        result = summarize_events(events)
        observed = {e["invocation_id"] for e in events if e["status"] == "attempted"}
        keys = {
            row[0] for row in self.db.execute("SELECT key FROM receipts WHERE run_id=?", (run_id,))
        }
        result["historical_receipts_without_attempt_events"] = len(keys - observed)
        result["coverage_complete"] = keys <= observed
        if not result["coverage_complete"]:
            result["token_usage"].update(input=None, output=None)
        return result

    def export_metrics(self, run_id, folder):
        (folder / "metrics.json").write_text(canonical(self.metrics(run_id)) + "\n")
        (folder / "metrics-traces.jsonl").write_text(
            "".join(canonical(e) + "\n" for e in self.metric_events(run_id))
        )

    def evidence(self, run_id: str) -> list[EvidenceBinding]:
        rows = self.db.execute(
            """
          SELECT e.payload FROM evidence e JOIN (
            SELECT id,max(revision) rev FROM evidence WHERE run_id=? GROUP BY id
          ) last ON e.id=last.id AND e.revision=last.rev
          WHERE e.run_id=? ORDER BY e.id
        """,
            (run_id, run_id),
        ).fetchall()
        return [EvidenceBinding.model_validate_json(r[0]) for r in rows]

    def save(
        self, data: OrchestrationData, dependencies: dict | None = None, update: dict | None = None
    ):
        with self.transaction():
            encoded = data.model_dump_json()
            self.db.execute("UPDATE runs SET data=? WHERE run_id=?", (encoded, data.task.run_id))
            if data.batch_id:
                snapshot = canonical(
                    {
                        "task": data.task,
                        "contract": data.contract,
                        "bindings": [b.model_dump(mode="json") for b in data.bindings],
                        "as_of": data.task.as_of.isoformat(),
                    }
                )
                previous = self.db.execute(
                    "SELECT snapshot,result FROM batches WHERE run_id=? AND batch_id=?",
                    (data.task.run_id, data.batch_id),
                ).fetchone()
                if previous and previous != (snapshot, encoded):
                    raise ValueError("committed batch cannot be changed")
                self.db.execute(
                    "INSERT OR IGNORE INTO batches VALUES(?,?,?,?)",
                    (data.task.run_id, data.batch_id, snapshot, encoded),
                )
                if dependencies is not None:
                    self.db.execute(
                        "INSERT OR REPLACE INTO dependencies VALUES(?,?,?)",
                        (data.task.run_id, data.batch_id, canonical(dependencies)),
                    )
                if update is not None:
                    self.db.execute(
                        "INSERT OR REPLACE INTO updates VALUES(?,?,?)",
                        (data.task.run_id, data.batch_id, canonical(update)),
                    )

    def inspect(self, run_id: str) -> dict:
        row = self.db.execute("SELECT data,elapsed FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("unknown run_id")
        receipts = self.db.execute(
            "SELECT key,kind,status FROM receipts WHERE run_id=? ORDER BY key", (run_id,)
        ).fetchall()
        return {
            "data": json.loads(row[0]) if row[0] else None,
            "elapsed_seconds": row[1],
            "receipts": [dict(zip(("key", "kind", "status"), r, strict=True)) for r in receipts],
            "metrics": self.metrics(run_id),
        }

    def attempted_tools(self, run_id: str) -> set[tuple[str, str]]:
        rows = self.db.execute(
            "SELECT request FROM receipts WHERE run_id=? AND kind='tool'", (run_id,)
        ).fetchall()
        requests = [json.loads(row[0]) for row in rows]
        return {(r["tool"], r["object_id"]) for r in requests if "tool" in r and "object_id" in r}
