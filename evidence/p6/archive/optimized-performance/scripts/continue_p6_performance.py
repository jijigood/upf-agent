"""Measure usable no-thinking generation after preserving the failed low-thinking probes."""

import os
import signal
import subprocess
import time
from pathlib import Path

import httpx

from upf_agent.p6_evaluation import pipeline_identity
from upf_agent.quality_repair import write_json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "runs/p6"
OWNED = 3608971
EVENTS = []


def record(**item):
    EVENTS.append({"time": time.time(), **item})
    write_json(
        DEST / "runner-performance.json",
        {
            "events": EVENTS,
            "owned_server_pid": OWNED,
            "pipeline_identity": pipeline_identity(ROOT),
            "generation_policy": {"enable_thinking": False},
            "scope": "generation component optimization; not a new Agent/business evaluation",
            "heldout_executed": False,
        },
    )


def run(name, context, concurrency):
    record(step=name, state="running")
    with (DEST / f"{name}.log").open("x") as log:
        code = subprocess.run(
            [
                str(ROOT / ".venv/bin/upf-agent"),
                "p6",
                "benchmark",
                "--model",
                "target",
                "--thinking-policy",
                "off",
                "--context",
                str(context),
                "--concurrency",
                str(concurrency),
                "--server-pid",
                str(OWNED),
                "--output",
                f"runs/p6/{name}.json",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    record(step=name, state="done" if code == 0 else "failed", returncode=code)
    if code:
        raise RuntimeError("Performance probe failed; preserve logs and inspect")


def stop():
    global OWNED
    if OWNED is None:
        return
    pid = OWNED
    proc = Path(f"/proc/{pid}")
    if proc.exists():
        if str(ROOT.parent / "models/Qwen3.8-27B") not in (
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


def start():
    global OWNED
    with (DEST / "server-target-32768-off-s2.log").open("x") as log:
        child = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/upf-agent"),
                "serve",
                "--model",
                "target",
                "--gpu",
                "0",
                "--max-model-len",
                "32768",
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
                record(step="server-target-32768-s2", state="ready", pid=child.pid)
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("Target startup timed out")


def main():
    args = Path(f"/proc/{OWNED}/cmdline").read_bytes().decode().split("\0")
    if (
        str(ROOT.parent / "models/Qwen3.8-27B") not in args
        or args[args.index("--max-model-len") + 1] != "16384"
        or args[args.index("--max-num-seqs") + 1] != "2"
    ):
        raise RuntimeError("Retained service profile mismatch")
    record(step="adopt-owned-16K-service", state="ready", pid=OWNED)
    try:
        for concurrency in (1, 2):
            run(f"perf-target-off-16384-c{concurrency}", 16384, concurrency)
        stop()
        start()
        for concurrency in (1, 2):
            run(f"perf-target-off-32768-c{concurrency}", 32768, concurrency)
        stop()
        record(step="all", state="done")
    finally:
        stop()


if __name__ == "__main__":
    main()
