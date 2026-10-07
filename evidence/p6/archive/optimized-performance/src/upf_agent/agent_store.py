"""Task-scoped durable receipts, budgets and audit events, separate from graph checkpoints."""

import hashlib
import json
import resource
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from upf_agent.agent_contracts import AgentTask, Receipt, ToolCall, ToolPacket


class Journal:
    def __init__(self, root: Path, task: AgentTask) -> None:
        self.root, self.task = root, task
        path = root / "data/p5/tasks.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks(
              id TEXT PRIMARY KEY, spec TEXT NOT NULL, tools INTEGER DEFAULT 0,
              models INTEGER DEFAULT 0, rounds INTEGER DEFAULT 0, repairs INTEGER DEFAULT 0,
              spent REAL DEFAULT 0, active_wall REAL, usage TEXT DEFAULT '{}');
            CREATE TABLE IF NOT EXISTS receipts(
              task TEXT, key TEXT, call TEXT, packet TEXT, attempts INTEGER DEFAULT 0,
              PRIMARY KEY(task,key));
            CREATE TABLE IF NOT EXISTS model_cache(
              task TEXT,key TEXT,payload TEXT,PRIMARY KEY(task,key));
            CREATE TABLE IF NOT EXISTS events(
              seq INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, kind TEXT, payload TEXT);
        """)
        self.db.commit()
        self.clock = time.monotonic()
        self.starts: dict[str, float] = {}

    def close(self) -> None:
        self.db.close()

    def register(self) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO tasks(id,spec) VALUES(?,?)",
                (self.task.run_id, self.task.model_dump_json()),
            )

    def begin(self) -> None:
        row = self.db.execute(
            "SELECT active_wall FROM tasks WHERE id=?", (self.task.run_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unregistered task")
        with self.db:
            if row[0] is not None:
                # Charge crashed active time; exclude normal waiting at a checkpoint.
                self.db.execute(
                    "UPDATE tasks SET spent=spent+? WHERE id=?",
                    (max(0, time.time() - row[0]), self.task.run_id),
                )
            self.db.execute(
                "UPDATE tasks SET active_wall=? WHERE id=?", (time.time(), self.task.run_id)
            )
        self.clock = time.monotonic()

    def account(self, finish: bool = False) -> None:
        now = time.monotonic()
        with self.db:
            self.db.execute(
                "UPDATE tasks SET spent=spent+?, active_wall=?, spec=? WHERE id=?",
                (
                    now - self.clock,
                    None if finish else time.time(),
                    self.task.model_dump_json(),
                    self.task.run_id,
                ),
            )
        self.clock = now

    def counters(self) -> dict[str, JsonValue]:
        row = self.db.execute(
            "SELECT tools,models,rounds,repairs,spent,usage FROM tasks WHERE id=?",
            (self.task.run_id,),
        ).fetchone()
        timings: dict[str, int] = {}
        for event in self.events():
            if isinstance(event.get("elapsed_ms"), int):
                stage = str(event.get("operation") or event.get("kind"))
                timings[stage] = timings.get(stage, 0) + event["elapsed_ms"]
        rss = (
            max(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            )
            * 1024
        )
        return {
            "tool_attempts": row[0],
            "model_requests_reserved": row[1],
            "rounds": row[2],
            "parameter_repairs": row[3],
            "active_seconds": row[4],
            "observed_usage": json.loads(row[5]),
            "stage_elapsed_ms": timings,
            "peak_observed_process_rss_bytes": rss,
            "resource_scope": (
                "max parent/direct-child Linux ru_maxrss for process lifetime; "
                "not summed tree/task-exclusive memory or GPU model service"
            ),
        }

    def remaining(self) -> float:
        return max(
            0.0,
            self.task.budget.timeout_seconds
            - float(self.counters()["active_seconds"])
            - (time.monotonic() - self.clock),
        )

    def reserve(self, field: str, amount: int = 1) -> bool:
        limits = {
            "tools": self.task.budget.max_tool_calls,
            "models": self.task.budget.max_model_calls,
            "rounds": self.task.budget.max_rounds,
            "repairs": 1,
        }
        if field not in limits or amount < 1 or self.remaining() <= 0:
            return False
        with self.db:
            cursor = self.db.execute(
                f"UPDATE tasks SET {field}={field}+? WHERE id=? AND {field}+?<=?",
                (amount, self.task.run_id, amount, limits[field]),
            )
        return cursor.rowcount == 1

    def event(self, kind: str, payload: dict[str, JsonValue]) -> None:
        marker = str(payload.get("key")) + ":" + ("model" if kind.startswith("model") else "tool")
        if kind.endswith("_started"):
            self.starts[marker] = time.monotonic()
        elif kind.endswith("_finished") and marker in self.starts:
            payload = {
                **payload,
                "elapsed_ms": int((time.monotonic() - self.starts.pop(marker)) * 1000),
            }
        payload = {**payload, "timestamp_utc": datetime.now(UTC).isoformat()}
        with self.db:
            self.db.execute(
                "INSERT INTO events(task,kind,payload) VALUES(?,?,?)",
                (self.task.run_id, kind, json.dumps(payload, ensure_ascii=False)),
            )

    def key(self, call: ToolCall) -> str:
        scope = {
            key: self.task.model_dump(mode="json")[key]
            for key in (
                "log_generation",
                "topology_generation",
                "manual_generation",
                "product",
                "version",
                "device_ids",
                "time_window",
            )
        }
        versions = {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("agent_tools.py", "agent_contracts.py")
        }
        return hashlib.sha256(
            json.dumps(
                {"call": call.model_dump(mode="json"), "scope": scope, "versions": versions},
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def scope_signature(self) -> str:
        values = self.task.model_dump(mode="json")
        scope = {
            name: values[name]
            for name in (
                "log_generation",
                "topology_generation",
                "manual_generation",
                "product",
                "version",
                "device_ids",
                "time_window",
            )
        }
        return hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()

    def cached(self, key: str) -> ToolPacket | None:
        row = self.db.execute(
            "SELECT packet FROM receipts WHERE task=? AND key=?", (self.task.run_id, key)
        ).fetchone()
        return ToolPacket.model_validate_json(row[0]) if row and row[0] else None

    def attempts(self, key: str) -> int:
        row = self.db.execute(
            "SELECT attempts FROM receipts WHERE task=? AND key=?", (self.task.run_id, key)
        ).fetchone()
        return row[0] if row else 0

    def attempted(self, key: str, call: ToolCall) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO receipts(task,key,call,attempts) VALUES(?,?,?,1) "
                "ON CONFLICT(task,key) DO UPDATE SET attempts=attempts+1",
                (self.task.run_id, key, call.model_dump_json()),
            )

    def save_packet(self, key: str, packet: ToolPacket) -> None:
        with self.db:
            self.db.execute(
                "UPDATE receipts SET packet=? WHERE task=? AND key=?",
                (packet.model_dump_json(), self.task.run_id, key),
            )

    def model_cached(self, key: str) -> dict[str, JsonValue] | None:
        row = self.db.execute(
            "SELECT payload FROM model_cache WHERE task=? AND key=?", (self.task.run_id, key)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_model(self, key: str, payload: dict[str, JsonValue]) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO model_cache VALUES(?,?,?)",
                (self.task.run_id, key, json.dumps(payload)),
            )
            usage = json.loads(
                self.db.execute(
                    "SELECT usage FROM tasks WHERE id=?", (self.task.run_id,)
                ).fetchone()[0]
            )
            observed = payload.get("usage", {})
            if isinstance(observed, dict):
                for name, value in observed.items():
                    if isinstance(value, int):
                        usage[name] = usage.get(name, 0) + value
            self.db.execute(
                "UPDATE tasks SET usage=? WHERE id=?", (json.dumps(usage), self.task.run_id)
            )

    def receipts(self) -> list[Receipt]:
        rows = self.db.execute(
            "SELECT key,call,packet FROM receipts "
            "WHERE task=? AND packet IS NOT NULL ORDER BY rowid",
            (self.task.run_id,),
        )
        return [
            Receipt(
                key=key,
                call=ToolCall.model_validate_json(call),
                packet=ToolPacket.model_validate_json(packet),
            )
            for key, call, packet in rows
            if json.loads(packet)["result"]["data"].get("_task_scope") == self.scope_signature()
        ]

    def events(self) -> list[dict[str, JsonValue]]:
        return [
            {"seq": seq, "kind": kind, **json.loads(payload)}
            for seq, kind, payload in self.db.execute(
                "SELECT seq,kind,payload FROM events WHERE task=? ORDER BY seq", (self.task.run_id,)
            )
        ]


def load_task(root: Path, run_id: str) -> AgentTask:
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("invalid run_id")
    with sqlite3.connect(root / "data/p5/tasks.sqlite3") as db:
        row = db.execute("SELECT spec FROM tasks WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise ValueError("unknown run_id")
    return AgentTask.model_validate_json(row[0])
