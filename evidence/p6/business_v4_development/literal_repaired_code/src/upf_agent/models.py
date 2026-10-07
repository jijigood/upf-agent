"""Local-only chat client and P1 inference smoke tests."""

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from upf_agent.config import GenerationModel, load_model_config


class LocalModelError(RuntimeError):
    pass


class LocalChatClient:
    def __init__(
        self,
        model: GenerationModel,
        transport: httpx.BaseTransport | None = None,
        chat_template_kwargs: dict[str, str] | None = None,
    ):
        self.model = model
        self.chat_template_kwargs = chat_template_kwargs
        self.client = httpx.Client(timeout=180, trust_env=False, transport=transport)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "LocalChatClient":
        return self

    def __exit__(self, *_errors: object) -> None:
        self.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 256,
        chat_template_kwargs: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model.served_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if tools is not None:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if response_format is not None:
            body["response_format"] = response_format
        selected_template_kwargs = chat_template_kwargs or self.chat_template_kwargs
        if selected_template_kwargs is not None:
            body["chat_template_kwargs"] = selected_template_kwargs
        try:
            response = self.client.post(f"{self.model.endpoint}/chat/completions", json=body)
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as error:
            raise LocalModelError(f"local model request failed: {error}") from error


def run_smoke(project_root: Path, profile: str = "baseline") -> dict[str, object]:
    config = load_model_config(project_root / "config" / "models.yaml")
    model = config.generation[profile]
    results: dict[str, object] = {"model": model.id, "endpoint": model.endpoint}
    timings_ms: dict[str, int] = {}
    total_start = perf_counter()
    template_kwargs = {"reasoning_effort": "low"} if profile == "target" else None
    if template_kwargs is not None:
        results["chat_template_kwargs"] = template_kwargs
    with LocalChatClient(model, chat_template_kwargs=template_kwargs) as client:
        start = perf_counter()
        text_reply = client.chat([{"role": "user", "content": "请只回答：本地模型已连接"}])
        timings_ms["text"] = round((perf_counter() - start) * 1000)
        results["text"] = {
            "pass": bool(text_reply.get("content")),
            "content": text_reply.get("content"),
        }

        format_spec = {
            "type": "json_schema",
            "json_schema": {
                "name": "model_probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}, "name": {"type": "string"}},
                    "required": ["ok", "name"],
                    "additionalProperties": False,
                },
            },
        }
        start = perf_counter()
        structured = client.chat(
            [{"role": "user", "content": "按JSON格式回答：ok=true，name=UPF。"}],
            response_format=format_spec,
        )
        timings_ms["json"] = round((perf_counter() - start) * 1000)
        parsed = json.loads(structured.get("content") or "")
        results["json"] = {"pass": parsed == {"ok": True, "name": "UPF"}, "result": parsed}

        tool_spec = {
            "type": "function",
            "function": {
                "name": "lookup_local_version",
                "description": "Get a locally registered product version",
                "parameters": {
                    "type": "object",
                    "properties": {"product": {"type": "string"}},
                    "required": ["product"],
                    "additionalProperties": False,
                },
            },
        }
        question = {"role": "user", "content": "请调用工具查询UPF版本，再告诉我结果。"}
        start = perf_counter()
        tool_reply = client.chat([question], tools=[tool_spec])
        timings_ms["tool_call"] = round((perf_counter() - start) * 1000)
        calls = tool_reply.get("tool_calls") or []
        if not calls:
            results["tool_roundtrip"] = {"pass": False, "error": "model emitted no tool call"}
        else:
            call = calls[0]
            arguments = json.loads(call["function"]["arguments"])
            if call["function"]["name"] != "lookup_local_version":
                raise LocalModelError("model selected unexpected tool")
            if arguments.get("product") != "UPF":
                raise LocalModelError(f"model tool parameters wrong: {arguments}")
            start = perf_counter()
            final = client.chat(
                [
                    question,
                    {"role": "assistant", "tool_calls": calls},
                    {"role": "tool", "tool_call_id": call["id"], "content": '{"version":"5.0"}'},
                ]
            )
            timings_ms["tool_result"] = round((perf_counter() - start) * 1000)
            results["tool_roundtrip"] = {
                "pass": "5.0" in (final.get("content") or ""),
                "tool_name": call["function"]["name"],
                "arguments": arguments,
                "final_content": final.get("content"),
            }
        if profile == "target":
            start = perf_counter()
            reasoning = client.chat(
                [{"role": "user", "content": "先思考，再回答：1+1等于几？最终回答只写数字。"}],
                max_tokens=512,
                chat_template_kwargs={"reasoning_effort": "medium"},
            )
            timings_ms["reasoning"] = round((perf_counter() - start) * 1000)
            reasoning_content = reasoning.get("reasoning") or reasoning.get("reasoning_content")
            final_content = reasoning.get("content")
            results["reasoning"] = {
                "pass": bool(reasoning_content) and "2" in (final_content or ""),
                "reasoning_content_present": bool(reasoning_content),
                "reasoning_field": (
                    "reasoning" if reasoning.get("reasoning") else "reasoning_content"
                ),
                "final_content": final_content,
            }
    timings_ms["total"] = round((perf_counter() - total_start) * 1000)
    results["timings_ms"] = timings_ms
    return results
