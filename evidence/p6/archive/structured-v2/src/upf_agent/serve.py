"""Launch an isolated, guarded, local vLLM service."""

import os
from pathlib import Path
from urllib.parse import urlparse

from upf_agent.config import load_model_config
from upf_agent.doctor import check_model_shards, gpu_snapshot


class ServiceLaunchBlocked(RuntimeError):
    pass


def launch_local_model(
    project_root: Path,
    profile: str,
    gpu_index: int,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
) -> None:
    config = load_model_config(project_root / "config" / "models.yaml")
    if profile not in config.generation:
        raise ServiceLaunchBlocked(f"unknown model profile: {profile}")
    model = config.generation[profile]
    ready, detail = check_model_shards(model.path)
    if not ready:
        raise ServiceLaunchBlocked(f"model is incomplete: {detail}")
    cards, error = gpu_snapshot()
    if error:
        raise ServiceLaunchBlocked(f"GPU query failed: {error}")
    card = next((item for item in cards if item["index"] == gpu_index), None)
    if card is None:
        raise ServiceLaunchBlocked(f"GPU {gpu_index} not found")
    minimum_free_mib = 68 * 1024
    if int(card["free_mib"]) < minimum_free_mib:
        raise ServiceLaunchBlocked(
            f"GPU {gpu_index} free {card['free_mib']} MiB < planning threshold "
            f"{minimum_free_mib} MiB; shared processes remain untouched"
        )
    environment_path = project_root / ".venv-inference" / "bin" / "vllm"
    if not environment_path.is_file():
        raise ServiceLaunchBlocked(f"isolated inference environment missing: {environment_path}")
    endpoint = urlparse(model.endpoint)
    assert endpoint.port is not None
    chosen_length = max_model_len or model.max_model_len
    chosen_sequences = max_num_seqs if max_num_seqs is not None else model.max_num_seqs
    if chosen_sequences not in {1, 2}:
        raise ServiceLaunchBlocked("local evaluation accepts concurrency 1 or 2")
    if chosen_length not in {16384, 32768}:
        raise ServiceLaunchBlocked("initial smoke accepts only 16K or 32K context")
    args = [
        str(environment_path),
        "serve",
        str(model.path),
        "--host",
        "127.0.0.1",
        "--port",
        str(endpoint.port),
        "--served-model-name",
        model.served_name,
        "--tensor-parallel-size",
        "1",
        "--dtype",
        model.dtype,
        "--max-model-len",
        str(chosen_length),
        "--max-num-seqs",
        str(chosen_sequences),
        "--gpu-memory-utilization",
        str(model.gpu_memory_utilization),
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        model.tool_parser,
    ]
    if model.reasoning_parser:
        args += ["--reasoning-parser", model.reasoning_parser]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
        }
    )
    os.execve(str(environment_path), args, environment)
