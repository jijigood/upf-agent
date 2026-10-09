"""Start only the dedicated loopback Neo4j instance, pinned by image digest."""

import json
import secrets
import socket
import subprocess
from pathlib import Path

IMAGE = (
    "neo4j:5.26.31-community@sha256:"
    "c7d25c0eeebfe125718d58b72ed5d663c85d0479047733845eb2237c67ce8069"
)
ROOT = Path(__file__).resolve().parents[1] / "runtime/v8"
NAME = "upf-agent-v8-neo4j"


def start():
    config_path = ROOT / "config/v8-neo4j.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = {
            "uri": "bolt://127.0.0.1:17687",
            "username": "neo4j",
            "password": secrets.token_urlsafe(24),
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config) + "\n")
        config_path.chmod(0o600)
    existing = subprocess.run(["docker", "inspect", NAME], capture_output=True)
    if existing.returncode == 0:
        subprocess.run(["docker", "start", NAME], check=True, capture_output=True)
        print(json.dumps({"container": NAME, "status": "existing", "image": IMAGE}))
        return
    for port in (17474, 17687):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    subprocess.run(
        ["docker", "volume", "create", "upf-agent-v8-neo4j-data"], check=True, capture_output=True
    )
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            NAME,
            "-p",
            "127.0.0.1:17474:7474",
            "-p",
            "127.0.0.1:17687:7687",
            "-v",
            "upf-agent-v8-neo4j-data:/data",
            "--env-file",
            "/dev/stdin",
            "--memory",
            "2g",
            "--cpus",
            "2",
            IMAGE,
        ],
        input=f"NEO4J_AUTH=neo4j/{config['password']}\nNEO4J_server_memory_heap_initial__size=256m\n"
        "NEO4J_server_memory_heap_max__size=512m\nNEO4J_server_memory_pagecache_size=256m\n",
        text=True,
        capture_output=True,
        check=True,
    )
    print(
        json.dumps(
            {
                "container": NAME,
                "id": result.stdout.strip(),
                "image": IMAGE,
                "ports": [17474, 17687],
                "volume": "upf-agent-v8-neo4j-data",
            }
        )
    )


if __name__ == "__main__":
    start()
