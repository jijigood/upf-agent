"""Run serial local development probes; never select or consume the heldout partition."""

import argparse
import json
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
STATUS = DEST / "runner.json"
EVENTS = []
OWNED = None


def record(**event):
    EVENTS.append({"time": time.time(), **event})
    write_json(
        STATUS,
        {
            "events": EVENTS,
            "owned_server_pid": OWNED,
            "pipeline_identity": pipeline_identity(ROOT),
            "heldout_executed": False,
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
    return code == 0


def stop():
    global OWNED
    if OWNED is None:
        return
    pid = OWNED
    if Path(f"/proc/{pid}/cmdline").exists():
        args = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        if "serve" not in args or not any(
            value.startswith("/home/huangxiaolin/models/Qwen3") for value in args
        ):
            raise RuntimeError("Owned PID identity changed; refusing to stop")
        os.kill(pid, signal.SIGTERM)
        for _ in range(90):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(1)
    record(step="server-stop", state="done", pid=pid)
    OWNED = None
    for _ in range(60):
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and int(result.stdout.strip()) >= 68 * 1024:
            return
        time.sleep(1)
    raise RuntimeError("GPU 0 not available after stopping owned server")


def start(profile, context, sequences):
    global OWNED
    name = f"server-{profile}-{context}-s{sequences}"
    with (DEST / f"{name}.log").open("x") as log:
        child = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/upf-agent"),
                "serve",
                "--model",
                profile,
                "--gpu",
                "0",
                "--max-model-len",
                str(context),
                "--max-num-seqs",
                str(sequences),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    OWNED = child.pid
    port = 18000 if profile == "baseline" else 18001
    for _ in range(300):
        if child.poll() is not None:
            raise RuntimeError(f"Server exited: {name}")
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2, trust_env=False)
            if response.status_code == 200:
                record(step=name, state="ready", pid=child.pid)
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server not ready: {name}")


def main():
    global OWNED
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-baseline-pid", type=int, required=True)
    initial = parser.parse_args().initial_baseline_pid
    args = Path(f"/proc/{initial}/cmdline").read_bytes().decode().split("\0")
    if "/home/huangxiaolin/models/Qwen3-30B-A3B-Instruct-2507" not in args:
        raise RuntimeError("Baseline ownership mismatch")
    OWNED = initial
    try:
        # The already-running parameter probe owns this service until completion.
        for _ in range(1200):
            path = DEST / "parameters-baseline-v2.json"
            try:
                if json.loads(path.read_text()).get("complete"):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Initial parameter probe did not complete")
        record(step="parameters-baseline-v2", state="already-completed")
        run(
            "model-baseline",
            "compare-model",
            "--cache",
            "runs/p6/retrieval-dev",
            "--model",
            "baseline",
            "--output",
            "runs/p6/model-baseline",
        )
        run(
            "agents-baseline",
            "compare-agents",
            "--model",
            "baseline",
            "--output",
            "runs/p6/agents-baseline",
        )
        run(
            "rag-baseline",
            "benchmark-rag",
            "--model",
            "baseline",
            "--output",
            "runs/p6/rag-baseline.json",
        )
        stop()
        for profile in ("baseline", "target"):
            if profile == "target":
                start(profile, 16384, 1)
                run(
                    "model-target",
                    "compare-model",
                    "--cache",
                    "runs/p6/retrieval-dev",
                    "--model",
                    profile,
                    "--output",
                    "runs/p6/model-target",
                )
                run(
                    "parameters-target-v2",
                    "tool-parameters",
                    "--model",
                    profile,
                    "--output",
                    "runs/p6/parameters-target-v2.json",
                )
                run(
                    "agents-target",
                    "compare-agents",
                    "--model",
                    profile,
                    "--output",
                    "runs/p6/agents-target",
                )
                run(
                    "rag-target",
                    "benchmark-rag",
                    "--model",
                    profile,
                    "--output",
                    "runs/p6/rag-target.json",
                )
                stop()
            for context in (16384, 32768):
                start(profile, context, 2)
                for concurrency in (1, 2):
                    name = f"perf-{profile}-{context}-c{concurrency}"
                    run(
                        name,
                        "benchmark",
                        "--model",
                        profile,
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
