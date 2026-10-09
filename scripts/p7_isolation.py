"""Rebuild and smoke-test a restored checkout in Linux mount/network namespaces."""

from __future__ import annotations

import argparse
import json
import selectors
import socket
import socketserver
import subprocess
import tempfile
import threading
from pathlib import Path

APP_ROOT = Path("/home/huangxiaolin/upf-agent")
PYTHONS = Path("/home/huangxiaolin/.local/share/uv/python")
MODELS = Path("/home/huangxiaolin/models")
UV = Path("/home/huangxiaolin/.local/bin/uv")
CACHE = Path("/home/huangxiaolin/.cache/uv")
MODEL_SOCKET = "/tmp/upf-model-ipc/model.sock"


def namespace_command(
    workspace: Path,
    command: list[str],
    *,
    cache: bool = False,
    ipc: Path | None = None,
    environments: Path | None = None,
) -> list[str]:
    args = [
        "bwrap",
        "--ro-bind",
        "/",
        "/",
        "--tmpfs",
        "/home/huangxiaolin",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/run",
        "--tmpfs",
        "/var/lib/docker",
        "--bind",
        str(workspace.resolve()),
        str(APP_ROOT),
        "--ro-bind",
        str(PYTHONS),
        str(PYTHONS),
        "--ro-bind",
        str(MODELS),
        str(MODELS),
        "--ro-bind",
        str(UV),
        "/tmp/uv",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--new-session",
        "--chdir",
        str(APP_ROOT),
        "--setenv",
        "HF_HUB_OFFLINE",
        "1",
        "--setenv",
        "TRANSFORMERS_OFFLINE",
        "1",
        "--setenv",
        "HF_HUB_DISABLE_TELEMETRY",
        "1",
        "--setenv",
        "DO_NOT_TRACK",
        "1",
        "--setenv",
        "PYTHONPATH",
        str(APP_ROOT / "src"),
        "--setenv",
        "UV_PYTHON_DOWNLOADS",
        "never",
    ]
    if cache:
        args += ["--bind", str(CACHE), "/tmp/uv-cache", "--setenv", "UV_CACHE_DIR", "/tmp/uv-cache"]
    if ipc:
        args += ["--bind", str(ipc), "/tmp/upf-model-ipc"]
    if environments:
        for name in (".venv", ".venv-retrieval", ".venv-inference"):
            args += ["--ro-bind", str(environments.resolve() / name), str(APP_ROOT / name)]
    return [*args, "--", *command]


def transfer(left: socket.socket, right: socket.socket) -> None:
    with selectors.DefaultSelector() as selector:
        selector.register(left, selectors.EVENT_READ, right)
        selector.register(right, selectors.EVENT_READ, left)
        while True:
            ready = selector.select(timeout=300)
            if not ready:
                return
            for event, _ in ready:
                data = event.fileobj.recv(65536)
                if not data:
                    return
                event.data.sendall(data)


class ModelRelay(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        # The bridge cannot select a destination: only the existing local model is exposed.
        with socket.create_connection(("127.0.0.1", 18000), timeout=10) as upstream:
            transfer(self.request, upstream)


class UnixModelServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class LoopbackRelay(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.connect(MODEL_SOCKET)
            transfer(self.request, upstream)


def blocked_connect(address: tuple[str, int]) -> str:
    try:
        with socket.create_connection(address, timeout=1):
            return "unexpectedly_connected"
    except OSError as error:
        return type(error).__name__


def child_probe(output: Path, resume_id: str | None = None) -> None:
    from upf_agent.agent_contracts import AgentTask
    from upf_agent.agent_engine import resume_agent, run_agent
    from upf_agent.config import load_model_config
    from upf_agent.log_tools import LogStore
    from upf_agent.models import LocalChatClient
    from upf_agent.p3_answer import answer_from_retrieval, source_lookup
    from upf_agent.p3_index import load_active_index
    from upf_agent.p3_search import search
    from upf_agent.retrieval import RetrievalWorker, RetrievalWorkerError

    absent = {name: not Path(f"/home/huangxiaolin/{name}").exists() for name in ("dify", "AITO")}
    blocked = {
        "external_ipv4": blocked_connect(("1.1.1.1", 443)),
        "external_ipv6": blocked_connect(("2606:4700:4700::1111", 443)),
        **{
            f"host_loopback_port_{port}": blocked_connect(("127.0.0.1", port))
            for port in (80, 8080, 5001, 8000)
        },
    }
    assert all(absent.values()) and all(
        value != "unexpectedly_connected" for value in blocked.values()
    )
    model_server = socketserver.ThreadingTCPServer(("127.0.0.1", 18000), LoopbackRelay)
    model_server.daemon_threads = True
    threading.Thread(target=model_server.serve_forever, daemon=True).start()
    config = load_model_config(APP_ROOT / "config/models.yaml")
    _, index = load_active_index(APP_ROOT)
    generation = "a6f18ac81a18aa2125fa"
    store = LogStore(APP_ROOT / "data/p4/logs", generation)
    events = store.analyze()
    assert events.data["total_events"] == 32539
    sample = store.find(event_type="pfcp_heartbeat_request", limit=1).data["events"][0]
    context = store.context(sample["event_id"])
    assert context.data.get("event")
    with LocalChatClient(config.generation["baseline"]) as client:
        model_reply = client.chat(
            [{"role": "user", "content": "请只回复：本地服务正常。"}], max_tokens=32
        )
    assert (model_reply.get("content") or "").strip()
    questions = [
        "手册‘配置root SSH登陆权限’中 systemctl 的命令原文是什么？",
        "手册历史案例‘UPFD 的N4 接口8805端口未侦听’记录了什么症状、信息收集及处理方法？",
    ]
    answers = []
    try:
        RetrievalWorker(APP_ROOT / "unavailable", "embed", config.embedding.path, "cuda:1")
    except RetrievalWorkerError:
        missing_environment_rejected = True
    else:
        raise AssertionError("Missing retrieval environment was not rejected")
    with RetrievalWorker(APP_ROOT, "embed", config.embedding.path, "cuda:1") as embedding:
        with RetrievalWorker(APP_ROOT, "rerank", config.reranker.path, "cuda:1") as reranking:
            for question in questions:
                found = search(APP_ROOT, question, "UPF", None, "cuda:1", (embedding, reranking))
                with LocalChatClient(config.generation["baseline"]) as client:
                    answer = answer_from_retrieval(APP_ROOT, question, found, client)
                assert answer["status"] in {"completed", "partial"} and answer["claims"]
                for citation in answer["citations"]:
                    assert source_lookup(APP_ROOT, index["index_generation"], citation["chunk_id"])[
                        "source_verified"
                    ]
                answers.append({"question": question, "answer": answer})
    cited = answers[0]["answer"]["citations"][0]
    raw = APP_ROOT / "data/raw" / cited["relative_path"]
    hidden = raw.with_name(raw.name + ".negative-control")
    raw.rename(hidden)
    try:
        try:
            source_lookup(APP_ROOT, index["index_generation"], cited["chunk_id"])
        except ValueError:
            missing_original_rejected = True
        else:
            raise AssertionError("Missing original source was not rejected")
    finally:
        hidden.rename(raw)
    if resume_id:
        restored = resume_agent(APP_ROOT, resume_id)
        assert restored.report and restored.report.status.value == "partial"
        task = {
            "resumed_id": resume_id,
            "status": restored.report.status.value,
            "facts": restored.report.facts,
        }
    else:
        from uuid import uuid4

        paused = run_agent(
            APP_ROOT,
            AgentTask(
                run_id=uuid4().hex,
                question="统计日志消息数量及证据缺口",
                task_type="diagnose",
                log_generation=generation,
                mode="deterministic",
            ),
            pause_after="seed",
        )
        assert paused.report.status.value == "need_input"
        task = {"paused_id": paused.task.run_id, "status": paused.report.status.value}
    report = {
        "dify_paths_absent": absent,
        "blocked_connections": blocked,
        "network_routes": Path("/proc/net/route").read_text(),
        "model_bridge_destination": "host loopback 127.0.0.1:18000 only",
        "model_reply": model_reply["content"],
        "log_events": 32539,
        "original_log_context_verified": True,
        "negative_controls": {
            "missing_retrieval_environment_rejected": missing_environment_rejected,
            "missing_original_source_rejected": missing_original_rejected,
        },
        "answers": answers,
        "checkpoint": task,
        "scope": "isolated application/retrieval; host generation service reused; not M6 accuracy",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    model_server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["rebuild", "check", "validate", "child"])
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-id")
    parser.add_argument("--environment-workspace", type=Path)
    args = parser.parse_args()
    if args.operation == "child":
        child_probe(args.output, args.resume_id)
        return
    if not args.workspace or not args.workspace.is_dir():
        parser.error("rebuild/check/validate require an existing --workspace")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.operation in {"rebuild", "check"}:
        commands = [
            ["/tmp/uv", "sync", "--offline", "--frozen", "--extra", "dev"],
            ["/tmp/uv", "venv", ".venv-retrieval", "--offline", "--python", "3.11"],
            [
                "/tmp/uv",
                "pip",
                "install",
                "--offline",
                "--python",
                ".venv-retrieval/bin/python",
                "-r",
                "environment-lock/retrieval.freeze.txt",
            ],
            ["/tmp/uv", "venv", ".venv-inference", "--offline", "--python", "3.10"],
            [
                "/tmp/uv",
                "pip",
                "install",
                "--offline",
                "--python",
                ".venv-inference/bin/python",
                "-r",
                "environment-lock/inference.freeze.txt",
            ],
            *[
                ["/tmp/uv", "pip", "check", "--python", f"{env}/bin/python"]
                for env in (".venv", ".venv-retrieval", ".venv-inference")
            ],
            [
                str(APP_ROOT / ".venv-inference/bin/python"),
                "-c",
                "import vllm, torch; print(vllm.__version__, torch.__version__); "
                "assert torch.cuda.is_available()",
            ],
        ]
        if args.operation == "check":
            commands = commands[5:]
        records = []
        for command in commands:
            completed = subprocess.run(
                namespace_command(args.workspace, command, cache=True),
                capture_output=True,
                text=True,
                check=False,
            )
            records.append(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            args.output.write_text(json.dumps({"commands": records}, indent=2) + "\n")
            if completed.returncode:
                raise RuntimeError(f"Offline rebuild failed; see {args.output}")
        print(f"Offline environment {args.operation} passed in the namespace")
    else:
        with tempfile.TemporaryDirectory(prefix="upf-p7-ipc-") as temporary:
            ipc = Path(temporary)
            with UnixModelServer(str(ipc / "model.sock"), ModelRelay) as relay:
                threading.Thread(target=relay.serve_forever, daemon=True).start()
                inner_output = APP_ROOT / "evidence/p7/isolated_probe.json"
                command = [
                    str(APP_ROOT / ".venv/bin/python"),
                    "scripts/p7_isolation.py",
                    "child",
                    "--output",
                    str(inner_output),
                ]
                if args.resume_id:
                    command += ["--resume-id", args.resume_id]
                completed = subprocess.run(
                    namespace_command(
                        args.workspace, command, ipc=ipc, environments=args.environment_workspace
                    ),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                relay.shutdown()
            if completed.returncode:
                args.output.write_text(
                    json.dumps(
                        {
                            "returncode": completed.returncode,
                            "stdout": completed.stdout,
                            "stderr": completed.stderr,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                raise RuntimeError(f"Isolated smoke failed; see {args.output}")
            shutil_source = args.workspace / "evidence/p7/isolated_probe.json"
            report = json.loads(shutil_source.read_text())
            report["data_workspace"] = str(args.workspace.resolve())
            report["environment_workspace"] = str(
                (args.environment_workspace or args.workspace).resolve()
            )
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(f"Isolated smoke passed: {args.output}")


if __name__ == "__main__":
    main()
