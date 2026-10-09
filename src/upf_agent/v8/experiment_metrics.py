"""Common dispatch-boundary counters; missing token/source telemetry stays null."""

from contextlib import contextmanager
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import Field

from upf_agent.v8.models import Model, digest


class InvocationEvent(Model):
    schema_version: str = "v8-dispatch-metrics-1"
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    invocation_id: str
    kind: Literal["tool", "model", "retriever", "source"]
    operation: str
    status: Literal["attempted", "committed", "reused", "error", "blocked"]
    request_sha256: str
    duration_seconds: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    error_type: str | None = None


def summarize_events(events):
    """Attempts and reuse are separate; error may accompany a committed failure receipt."""
    items = [InvocationEvent.model_validate(e) for e in events]
    if len({e.event_id for e in items}) != len(items):
        raise ValueError("duplicate metric event IDs")
    by_kind = {}
    for kind in sorted({e.kind for e in items} | {"tool", "model"}):
        selected = [e for e in items if e.kind == kind]
        by_kind[kind] = {
            status: sum(e.status == status for e in selected)
            for status in ("attempted", "committed", "reused", "error", "blocked")
        }
    model_attempts = [e for e in items if e.kind == "model" and e.status == "attempted"]
    model_receipts = [e for e in items if e.kind == "model" and e.status == "committed"]
    known = [
        e for e in model_receipts if e.input_tokens is not None and e.output_tokens is not None
    ]
    complete = len(known) == len(model_attempts) == len(model_receipts)
    return {
        "schema_version": "v8-dispatch-metrics-1",
        "measurement_boundary": "dispatcher (not physical network requests or source reads)",
        "invocations": by_kind,
        "token_usage": {
            "input": sum(e.input_tokens for e in known) if complete else None,
            "output": sum(e.output_tokens for e in known) if complete else None,
            "known_model_receipts": len(known),
            "model_attempts": len(model_attempts),
        },
        "source_reads": None,
        "cache_hits": None,
    }


class MetricCollector:
    """Adapter-independent event collector with an optional durable sink."""

    def __init__(self, run_id, sink=None):
        self.run_id = run_id
        self.events = []
        self.sink = sink

    def emit(self, invocation_id, kind, operation, status, request, **kwargs):
        value = InvocationEvent(
            run_id=self.run_id,
            invocation_id=invocation_id,
            kind=kind,
            operation=operation,
            status=status,
            request_sha256=digest(request),
            **kwargs,
        ).model_dump(mode="json")
        if self.sink:
            self.sink(value)
        self.events.append(value)
        return value

    @contextmanager
    def measure(self, kind, operation, request):
        invocation = uuid4().hex
        self.emit(invocation, kind, operation, "attempted", request)
        started = perf_counter()
        receipt = {}
        try:
            yield receipt
        except Exception as error:
            self.emit(
                invocation,
                kind,
                operation,
                "error",
                request,
                duration_seconds=perf_counter() - started,
                error_type=type(error).__name__,
            )
            raise
        else:
            self.emit(
                invocation,
                kind,
                operation,
                "committed",
                request,
                duration_seconds=perf_counter() - started,
                **receipt,
            )
            if receipt.get("error_type"):
                self.emit(
                    invocation,
                    kind,
                    operation,
                    "error",
                    request,
                    error_type=receipt["error_type"],
                )

    def summary(self):
        return summarize_events(self.events)


class MeasuredBackend:
    """Wrap P5 or another backend without changing planner or tool-selection behavior."""

    def __init__(self, backend, collector):
        self.backend, self.collector = backend, collector

    def tool(self, task, call, known, timeout):
        request = call.model_dump(mode="json")
        with self.collector.measure("tool", call.name, request) as receipt:
            result = self.backend.tool(task, call, known, timeout)
            if getattr(result, "error_code", None):
                receipt["error_type"] = result.error_code
            return result

    def model(self, operation, data, timeout, allowance, repair=None):
        request = {"operation": operation, "run_id": data.task.run_id, "allowance": allowance}
        with self.collector.measure("model", operation, request) as receipt:
            result = self.backend.model(operation, data, timeout, allowance, repair)
            usage = result.get("usage", {})
            receipt["input_tokens"] = usage.get("prompt_tokens")
            receipt["output_tokens"] = usage.get("completion_tokens")
            return result
