"""P1 behavior checks for strict contracts, local endpoints, and GPU guard."""

from datetime import datetime
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from upf_agent.config import load_model_config
from upf_agent.contracts import TimeWindow, ToolResult
from upf_agent.doctor import check_model_shards
from upf_agent.models import LocalChatClient
from upf_agent.serve import ServiceLaunchBlocked, launch_local_model

ROOT = Path(__file__).resolve().parents[1]


def test_time_window_requires_order_and_timezone() -> None:
    with pytest.raises(ValidationError):
        TimeWindow(
            start=datetime.fromisoformat("2025-05-14T00:00:00"),
            end=datetime.fromisoformat("2025-05-15T00:00:00"),
            timezone_source="unknown",
        )
    with pytest.raises(ValidationError):
        TimeWindow(
            start=datetime.fromisoformat("2025-05-15T00:00:00+08:00"),
            end=datetime.fromisoformat("2025-05-14T00:00:00+08:00"),
            timezone_source="input",
        )


def test_tool_result_accepts_nested_json_data() -> None:
    result = ToolResult.model_validate(
        {
            "status": "completed",
            "data": {"counts": {"PFCP": 14}, "examples": [{"line": 10, "state": "raised"}]},
            "elapsed_ms": 12,
        }
    )
    assert result.data["counts"] == {"PFCP": 14}


def test_model_client_uses_dedicated_loopback_endpoint_without_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    model = load_model_config(ROOT / "config/models.yaml").generation["baseline"]

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "127.0.0.1"
        assert request.url.port == 18000
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["content-type"] == "application/json"
        assert request.read().decode().count(model.served_name) == 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    with LocalChatClient(model, transport=httpx.MockTransport(respond)) as client:
        assert client.chat([{"role": "user", "content": "probe"}])["content"] == "ok"


def test_service_guard_refuses_insufficient_shared_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "upf_agent.serve.gpu_snapshot",
        lambda: ([{"index": 0, "free_mib": 40000}], None),
    )
    with pytest.raises(ServiceLaunchBlocked, match="shared processes remain untouched"):
        launch_local_model(ROOT, "baseline", 0)


def test_external_model_endpoint_is_rejected(tmp_path: Path) -> None:
    config = (ROOT / "config/models.yaml").read_text(encoding="utf-8")
    altered = tmp_path / "models.yaml"
    altered.write_text(config.replace("http://127.0.0.1:18000/v1", "https://example.com/v1"))
    with pytest.raises(ValidationError):
        load_model_config(altered)


def test_partial_sharded_download_is_not_declared_ready(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model-00001-of-00018.safetensors").write_bytes(b"partial")
    ready, detail = check_model_shards(tmp_path)
    assert not ready
    assert "index" in detail
