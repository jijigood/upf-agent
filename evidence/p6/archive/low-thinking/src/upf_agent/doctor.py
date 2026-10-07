"""Read-only readiness probes; never stop or claim other processes' GPUs."""

import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from upf_agent.config import GenerationModel, load_model_config


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def gpu_snapshot() -> tuple[list[dict[str, int | str]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return [], str(error)
    cards: list[dict[str, int | str]] = []
    for line in completed.stdout.splitlines():
        index, name, total, used, free, driver = [item.strip() for item in line.split(",", 5)]
        cards.append(
            {
                "index": int(index),
                "name": name,
                "total_mib": int(total),
                "used_mib": int(used),
                "free_mib": int(free),
                "driver": driver,
            }
        )
    return cards, None


def check_model_shards(root: Path) -> tuple[bool, str]:
    if not root.is_dir():
        return False, f"model directory missing: {root}"
    index = root / "model.safetensors.index.json"
    if index.is_file():
        expected = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
        missing = sorted(name for name in expected if not (root / name).is_file())
        if missing:
            return False, f"missing {len(missing)} indexed shards"
        if not expected:
            return False, "index contains no shards"
        return True, f"{len(expected)} indexed shards present; verify hashes against model manifest"
    return False, "model.safetensors.index.json missing; shard completeness unknown"


def local_endpoint_status(model: GenerationModel) -> Check:
    parsed = urlparse(model.endpoint)
    assert parsed.hostname is not None and parsed.port is not None
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=0.5):
            pass
    except OSError:
        return Check(f"endpoint_{model.served_name}", "pending", "dedicated port is not serving")
    try:
        with httpx.Client(timeout=2, trust_env=False) as client:
            result = client.get(f"{model.endpoint}/models")
            result.raise_for_status()
            ids = {item["id"] for item in result.json().get("data", [])}
    except (httpx.HTTPError, ValueError, KeyError) as error:
        return Check(f"endpoint_{model.served_name}", "fail", f"unexpected service: {error}")
    if model.served_name not in ids:
        return Check(f"endpoint_{model.served_name}", "fail", f"served model mismatch: {ids}")
    return Check(f"endpoint_{model.served_name}", "pass", "metadata endpoint responds")


def bf16_probe(python: Path, gpu_index: int) -> Check:
    if not python.is_file():
        return Check("bf16_cuda", "pending", f"inference Python missing: {python}")
    code = (
        "import json,torch; "
        "x=torch.ones((64,64),device='cuda',dtype=torch.bfloat16); "
        "y=x@x; torch.cuda.synchronize(); "
        "print(json.dumps({'torch':torch.__version__, 'cuda':torch.version.cuda, "
        "'capability':torch.cuda.get_device_capability(), 'result':float(y[0,0])}))"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["VLLM_NO_USAGE_STATS"] = "1"
    try:
        result = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return Check("bf16_cuda", "fail", str(error))
    if result.returncode != 0:
        return Check("bf16_cuda", "fail", result.stderr[-400:])
    try:
        value = json.loads(result.stdout.splitlines()[-1])
    except (ValueError, IndexError):
        return Check("bf16_cuda", "fail", "no parseable probe response")
    return Check("bf16_cuda", "pass", json.dumps(value, sort_keys=True))


def run_doctor(project_root: Path, probe_bf16: bool = False) -> dict[str, object]:
    config = load_model_config(project_root / "config" / "models.yaml")
    checks: list[Check] = []
    cards, error = gpu_snapshot()
    if error:
        checks.append(Check("gpu_query", "fail", error))
    else:
        checks.append(Check("gpu_query", "pass", f"{len(cards)} cards; snapshot only"))
        for card in cards:
            # 68 GiB is a planning threshold for one BF16 generator, not a hard reservation.
            threshold_mib = 68 * 1024
            enough = int(card["free_mib"]) >= threshold_mib
            status = "pass" if enough else "blocked"
            checks.append(
                Check(
                    f"gpu_{card['index']}_generation_headroom",
                    status,
                    f"free {card['free_mib']} MiB; initial planning threshold {threshold_mib} MiB",
                )
            )
    for name, model in config.generation.items():
        present, detail = check_model_shards(model.path)
        checks.append(Check(f"weights_{name}", "pass" if present else "pending", detail))
        checks.append(local_endpoint_status(model))
    for name in ("embedding", "reranker"):
        model = getattr(config, name)
        present, detail = check_model_shards(model.path)
        checks.append(Check(f"weights_{name}", "pass" if present else "pending", detail))
    proxies = [name for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY") if os.getenv(name)]
    checks.append(
        Check(
            "local_http_proxy",
            "pass",
            (
                f"proxy variables present: {','.join(proxies) or 'none'}; "
                "model client uses trust_env=False"
            ),
        )
    )
    if probe_bf16:
        available = [card for card in cards if int(card["free_mib"]) >= 4096]
        if available:
            python = Path("/home/huangxiaolin/dify_vllm_uv310/bin/python")
            checks.append(bf16_probe(python, int(available[0]["index"])))
        else:
            checks.append(Check("bf16_cuda", "blocked", "no card has 4 GiB free"))
    return {
        "captured_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "checks": [check.as_dict() for check in checks],
        "gpus": cards,
    }
