"""Verify the live local generation profile before consuming a one-use evaluation."""

from pathlib import Path
from urllib.parse import urlparse

import httpx

from upf_agent.config import GenerationModel, load_model_config


def _flag(args: list[str], name: str) -> str | None:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return None


def expected_configuration(model: GenerationModel) -> dict:
    return {
        "model_path": str(model.path),
        "served_name": model.served_name,
        "context": model.max_model_len,
        "max_num_seqs": model.max_num_seqs,
        "dtype": model.dtype,
        "gpu_memory_utilization": model.gpu_memory_utilization,
        "tool_parser": model.tool_parser,
        "reasoning_parser": model.reasoning_parser,
        "eager": True,
    }


def command_configuration(model: GenerationModel, proc_root: Path = Path("/proc")) -> dict:
    endpoint = urlparse(model.endpoint)
    expected = expected_configuration(model)
    matches = []
    for path in proc_root.glob("[0-9]*/cmdline"):
        try:
            args = path.read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if str(model.path) not in args or not any(Path(arg).name == "vllm" for arg in args):
            continue

        if _flag(args, "--port") != str(endpoint.port):
            continue
        actual = {
            "model_path": str(model.path),
            "served_name": _flag(args, "--served-model-name"),
            "context": int(_flag(args, "--max-model-len") or 0),
            "max_num_seqs": int(_flag(args, "--max-num-seqs") or 0),
            "dtype": _flag(args, "--dtype"),
            "gpu_memory_utilization": float(_flag(args, "--gpu-memory-utilization") or 0),
            "tool_parser": _flag(args, "--tool-call-parser"),
            "reasoning_parser": _flag(args, "--reasoning-parser"),
            "eager": "--enforce-eager" in args,
        }
        if _flag(args, "--host") not in {"127.0.0.1", "::1", "localhost"} or actual != expected:
            raise ValueError("Live generation arguments differ from the evaluation configuration")
        matches.append(actual)
    if len(matches) != 1:
        raise ValueError("Need exactly one inspectable local model service at the configured port")
    return matches[0]


def generation_preflight(root: Path, profile: str) -> dict:
    model = load_model_config(root / "config/models.yaml").generation[profile]
    configuration = command_configuration(model)
    endpoint = urlparse(model.endpoint)
    origin = f"{endpoint.scheme}://{endpoint.netloc}"
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            client.get(origin + "/health").raise_for_status()
            response = client.get(model.endpoint + "/models")
            response.raise_for_status()
            served = [r for r in response.json()["data"] if r["id"] == model.served_name]
        if len(served) != 1 or (
            served[0].get("root") != str(model.path)
            or served[0].get("max_model_len") != model.max_model_len
        ):
            raise ValueError("Model service metadata differs from the evaluation configuration")
    except (httpx.HTTPError, KeyError, TypeError) as error:
        raise RuntimeError("Local model service is not ready; heldout was not consumed") from error
    return configuration
