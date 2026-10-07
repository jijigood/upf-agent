"""Continue the target profile after repairing structured generation, preserving controls."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx

from upf_agent.p6_evaluation import pipeline_identity
from upf_agent.quality_repair import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "runs/p6"
EVENTS = json.loads((DEST / "runner.json").read_text())["events"]
OWNED = 3514386


def record(**item):
    EVENTS.append({"time": time.time(), **item})
    write_json(
        DEST / "runner.json",
        {
            "events": EVENTS,
            "owned_server_pid": OWNED,
            "pipeline_identity": pipeline_identity(ROOT),
            "heldout_executed": False,
            "adapter_policy": "target JSON disables thinking; plain replies keep low",
            "baseline_control_policy_unchanged": True,
        },
    )


def run(name, *arguments):
    record(step=name, state="running")
    with (DEST / f"{name}.log").open("x") as log:
        code = subprocess.run(
            [str(ROOT / ".venv/bin/upf-agent"), "p6", *arguments],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    record(step=name, state="done" if code == 0 else "failed", returncode=code)


def stop():
    global OWNED
    if OWNED is None:
        return
    pid = OWNED
    proc = Path(f"/proc/{pid}")
    if proc.exists():
        if "/home/huangxiaolin/models/Qwen3.8-27B" not in (
            proc / "cmdline"
        ).read_bytes().decode().split("\0"):
            raise RuntimeError("Server ownership changed")
        os.kill(pid, signal.SIGTERM)
        for _ in range(90):
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            if not proc.exists() or (proc / "stat").read_text().split()[2] == "Z":
                break
            time.sleep(1)
    OWNED = None
    record(step="server-stop", state="done", pid=pid)
    for _ in range(60):
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and int(result.stdout.strip()) >= 68 * 1024:
            return
        time.sleep(1)
    raise RuntimeError("GPU headroom unavailable")


def start(context):
    global OWNED
    with (DEST / f"server-target-{context}-s2.log").open("x") as log:
        child = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/upf-agent"),
                "serve",
                "--model",
                "target",
                "--gpu",
                "0",
                "--max-model-len",
                str(context),
                "--max-num-seqs",
                "2",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    OWNED = child.pid
    for _ in range(900):
        if child.poll() is not None:
            raise RuntimeError("Target service exited")
        try:
            if (
                httpx.get("http://127.0.0.1:18001/health", timeout=2, trust_env=False).status_code
                == 200
            ):
                record(step=f"server-target-{context}-s2", state="ready", pid=child.pid)
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("Target startup timed out")


def main():
    args = Path(f"/proc/{OWNED}/cmdline").read_bytes().decode().split("\0")
    if "/home/huangxiaolin/models/Qwen3.8-27B" not in args:
        raise RuntimeError("Retained target service ownership mismatch")
    record(
        step="adapter-repair",
        state="done",
        previous_pipeline="760309bcd4672dbbd0fd1c8dd805c3a6ff656e8e5bb5d59ee1279313639bf7a3",
        source_sha256=sha(ROOT / "src/upf_agent/agent_worker.py"),
        low_thinking_probe="model-target-low-thinking-incomplete",
    )
    try:
        run(
            "model-target",
            "compare-model",
            "--cache",
            "runs/p6/retrieval-dev",
            "--model",
            "target",
            "--output",
            "runs/p6/model-target",
        )
        run(
            "parameters-target-v2",
            "tool-parameters",
            "--model",
            "target",
            "--output",
            "runs/p6/parameters-target-v2.json",
        )
        run(
            "agents-target",
            "compare-agents",
            "--model",
            "target",
            "--output",
            "runs/p6/agents-target",
        )
        run(
            "rag-target",
            "benchmark-rag",
            "--model",
            "target",
            "--output",
            "runs/p6/rag-target.json",
        )
        stop()
        for context in (16384, 32768):
            start(context)
            for concurrency in (1, 2):
                name = f"perf-target-{context}-c{concurrency}"
                run(
                    name,
                    "benchmark",
                    "--model",
                    "target",
                    "--context",
                    str(context),
                    "--concurrency",
                    str(concurrency),
                    "--server-pid",
                    str(OWNED),
                    "--output",
                    f"runs/p6/{name}.json",
                )
            stop()
        record(step="all", state="done")
    except Exception as error:
        record(step="runner", state="failed", error=str(error))
        raise
    finally:
        stop()


if __name__ == "__main__":
    main()
