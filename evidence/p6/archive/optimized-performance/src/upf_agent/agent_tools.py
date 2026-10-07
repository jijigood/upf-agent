"""Read-only tool registry with validation before dispatch and task-scoped provenance."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, JsonValue, ValidationError

from upf_agent.agent_contracts import AgentTask, SourceFact, ToolCall, ToolPacket, VerifiedEvidence
from upf_agent.contracts import ResultStatus, StrictModel, TimeWindow, ToolResult
from upf_agent.log_correlation import correlate_events
from upf_agent.log_parser import file_sha
from upf_agent.log_tools import LogStore
from upf_agent.p3_answer import source_lookup
from upf_agent.p3_index import load_active_index
from upf_agent.p3_search import search
from upf_agent.topology import lookup_topology


class SearchArgs(StrictModel):
    query: str = Field(min_length=1, max_length=2000)
    product: str | None = None
    version: str | None = None
    top_k: int = Field(default=6, ge=1, le=8)


class ManualArgs(StrictModel):
    chunk_id: str = Field(min_length=1, max_length=100)
    source_id: str | None = None


class LogArgs(StrictModel):
    log_id: str = Field(pattern=r"^[0-9a-f]{20}$")
    window: TimeWindow | None = None
    device_id: str | None = None


class AnalyzeArgs(LogArgs):
    event_type: str | None = None
    bucket_minutes: int = Field(default=60, ge=1, le=1440)


class FindArgs(LogArgs):
    event_type: str | None = None
    event_state: Literal["raised", "cleared", "observed", "unknown"] | None = None
    protocol: str | None = None
    peer_ip: str | None = None
    seid: str | None = None
    teid: str | None = None
    limit: int = Field(default=10, ge=1, le=50)


class ContextArgs(StrictModel):
    log_id: str = Field(pattern=r"^[0-9a-f]{20}$")
    event_id: str = Field(min_length=1, max_length=100)
    before: int = Field(default=2, ge=0, le=5)
    after: int = Field(default=2, ge=0, le=5)


class TopologyArgs(StrictModel):
    topology_id: str = Field(pattern=r"^[0-9a-f]{20}$")
    device_id: str = Field(min_length=1, max_length=100)
    interface_type: Literal["N3", "N4", "N6", "N9"] | None = None
    at: datetime | None = None


class CompareArgs(StrictModel):
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class CorrelateArgs(LogArgs):
    horizon_seconds: int = Field(default=86400, ge=1, le=604800)
    limit: int = Field(default=100, ge=1, le=200)


REGISTRY: dict[str, type[StrictModel]] = {
    "search_manual": SearchArgs,
    "read_manual_block": ManualArgs,
    "analyze_log_window": AnalyzeArgs,
    "find_log_events": FindArgs,
    "read_log_context": ContextArgs,
    "lookup_topology": TopologyArgs,
    "compare_evidence": CompareArgs,
    "correlate_events": CorrelateArgs,
}


def validate_call(
    task: AgentTask, call: ToolCall, known: list[VerifiedEvidence]
) -> tuple[StrictModel | None, str | None]:
    schema = REGISTRY.get(call.name)
    if schema is None:
        return None, "unknown_tool"
    try:
        args = schema.model_validate_json(json.dumps(call.arguments))
    except ValidationError:
        return None, "invalid_arguments"
    native_ids = {item.native_id for item in known}
    ids = {item.evidence_id for item in known}
    devices = set(task.device_ids) | {
        str(item.metadata["device_id"]) for item in known if item.metadata.get("device_id")
    }
    if isinstance(args, (LogArgs, ContextArgs)) and args.log_id != task.log_generation:
        return None, "scope_denied"
    if isinstance(args, LogArgs):
        if args.window is not None and args.window != task.time_window:
            return None, "scope_denied"
        args.window = task.time_window
        if args.device_id is not None and args.device_id not in devices:
            return None, "scope_denied"
        if len(task.device_ids) == 1:
            args.device_id = task.device_ids[0]
        elif len(task.device_ids) > 1:
            return None, "scope_denied"  # CLI creates separate tasks for distinct device scopes.
    if isinstance(args, SearchArgs):
        if (task.product and args.product not in (None, task.product)) or (
            task.version and args.version not in (None, task.version)
        ):
            return None, "scope_denied"
        args.product, args.version = task.product or args.product, task.version or args.version
    if isinstance(args, (ManualArgs, ContextArgs)):
        native = args.chunk_id if isinstance(args, ManualArgs) else args.event_id
        if native not in native_ids:
            return None, "scope_denied"
    if isinstance(args, TopologyArgs):
        if args.topology_id != task.topology_generation or args.device_id not in devices:
            return None, "scope_denied"
        if args.at is not None:
            if args.at.utcoffset() is None or (
                task.time_window and not task.time_window.start <= args.at < task.time_window.end
            ):
                return None, "scope_denied"
    if isinstance(args, CompareArgs) and not set(args.evidence_ids) <= ids:
        return None, "scope_denied"
    return args, None


def failure(code: str, message: str, retryable: bool = False) -> ToolPacket:
    status = ResultStatus.MISSING_SOURCE if code == "missing_source" else ResultStatus.FAILED
    return ToolPacket(
        result=ToolResult(status=status, warnings=[message], elapsed_ms=0),
        gaps=[message],
        error_code=code,
        retryable=retryable,
    )


def evidence(
    task: AgentTask,
    native: str,
    source_id: str,
    sha: str,
    locator: str,
    kind: Literal["manual", "log", "topology", "tool"],
    excerpt: str,
    metadata: dict[str, JsonValue] | None = None,
) -> VerifiedEvidence:
    return VerifiedEvidence(
        evidence_id=f"{task.run_id}:{native}",
        native_id=native,
        source_id=source_id,
        source_sha256=sha,
        locator=locator,
        kind=kind,
        excerpt=excerpt[:12000],
        tool_version="p5-tools-v1",
        metadata=metadata or {},
        applicability_note="来源可核查；现场适用性与因果支持未由引用检查证明。",
    )


def execute_tool(
    root: Path, task: AgentTask, call: ToolCall, known: list[VerifiedEvidence]
) -> ToolPacket:
    args, error = validate_call(task, call, known)
    if error:
        return failure(error, f"工具请求被拒绝：{error}，{call.name}。")
    try:
        return _dispatch(root, task, call.name, args, known)
    except FileNotFoundError as error:
        return failure("missing_source", str(error))
    except (ValueError, KeyError, json.JSONDecodeError) as error:
        return failure("corrupt_source", str(error))
    except Exception as error:
        return failure("tool_failed", f"{type(error).__name__}: {error}")


def _dispatch(
    root: Path, task: AgentTask, name: str, args: StrictModel, known: list[VerifiedEvidence]
) -> ToolPacket:
    if isinstance(args, SearchArgs):
        _, manifest = load_active_index(root)
        if manifest["index_generation"] != task.manual_generation:
            raise ValueError(
                "pinned manual generation differs from active index; explicit new task required"
            )
        retrieval = search(root, args.query, args.product, args.version, task.retrieval_device)
        retrieval["evidence"] = retrieval["evidence"][: args.top_k]
        entries = []
        items = [
            item
            for parent in retrieval["evidence"]
            for item in [parent, *parent.get("neighbors", [])]
        ]
        seen = set()
        for item in items:
            if item["chunk_id"] in seen:
                continue
            seen.add(item["chunk_id"])
            original = source_lookup(root, str(task.manual_generation), item["chunk_id"])
            if not original["text"].startswith(item["text"]):
                raise ValueError("manual excerpt differs from pinned source")
            entries.append(
                evidence(
                    task,
                    item["chunk_id"],
                    original["source_id"],
                    original["source_sha256"],
                    ", ".join(original["locators"]),
                    "manual",
                    item["text"],
                    {
                        "product": original["product"],
                        "version": original["version"],
                        "generation": task.manual_generation,
                    },
                )
            )
        result = ToolResult(
            status=ResultStatus.COMPLETED if entries else ResultStatus.MISSING_SOURCE,
            data={"retrieval": retrieval},
            evidence_ids=[item.evidence_id for item in entries],
            warnings=["检索命中是候选原文，不等于已经回答问题。"],
            elapsed_ms=0,
        )
        return ToolPacket(
            result=result, evidence=entries, gaps=[] if entries else ["没有相符手册证据。"]
        )
    if isinstance(args, ManualArgs):
        original = source_lookup(root, str(task.manual_generation), args.chunk_id)
        if args.source_id and original["source_id"] != args.source_id:
            raise ValueError("source/chunk mismatch")
        entry = evidence(
            task,
            args.chunk_id,
            original["source_id"],
            original["source_sha256"],
            ", ".join(original["locators"]),
            "manual",
            original["text"],
            {
                "product": original["product"],
                "version": original["version"],
                "generation": task.manual_generation,
            },
        )
        return ToolPacket(
            result=ToolResult(
                status=ResultStatus.COMPLETED,
                data={},
                evidence_ids=[entry.evidence_id],
                elapsed_ms=0,
            ),
            evidence=[entry],
        )
    if isinstance(args, CompareArgs):
        selected = [item for item in known if item.evidence_id in args.evidence_ids]
        rows = [
            {
                "evidence_id": item.evidence_id,
                "source_sha256": item.source_sha256,
                "kind": item.kind,
                "scope": item.metadata,
            }
            for item in selected
        ]
        return ToolPacket(
            result=ToolResult(
                status=ResultStatus.COMPLETED,
                data={"comparisons": rows, "is_causal_proof": False},
                evidence_ids=args.evidence_ids,
                warnings=["只比较已登记证据的范围、版本、标识；不推导因果。"],
                elapsed_ms=0,
            )
        )
    if isinstance(args, TopologyArgs):
        result = lookup_topology(
            root / "data/p4/topology" / args.topology_id,
            args.device_id,
            args.interface_type,
            args.at,
        )
        entries, facts = [], []
        for edge in cast(list[dict[str, JsonValue]], result.data.get("edges", [])):
            entry = evidence(
                task,
                str(edge["edge_id"]),
                str(edge["source_id"]),
                str(edge["source_sha256"]),
                f"{edge['source_sheet'] or 'csv'}:row={edge['source_row']}",
                "topology",
                json.dumps(edge, ensure_ascii=False),
                edge,
            )
            entries.append(entry)
            prefix = "合成测试拓扑" if edge["synthetic"] else "导入拓扑"
            facts.append(
                SourceFact(
                    text=f"{prefix}记载 {edge['device_id']}:{edge['device_port']} → "
                    f"{edge['peer_device_id']}:{edge['peer_port']}，"
                    f"接口 {edge['interface_type']}；仅作检查对象。",
                    evidence_ids=[entry.evidence_id],
                )
            )
        return ToolPacket(
            result=result,
            evidence=entries,
            facts=facts,
            gaps=[] if entries else ["没有适用的拓扑边。"],
        )
    log_id = cast(LogArgs | ContextArgs, args).log_id
    store = LogStore(root / "data/p4/logs", log_id)
    source = Path(str(store.manifest["source_path"]))
    if file_sha(source) != store.manifest["source_sha256"]:
        raise ValueError("registered log source hash changed")
    entries, facts = [], []
    if isinstance(args, AnalyzeArgs):
        result = store.analyze(args.window, args.event_type, args.device_id, args.bucket_minutes)
        if result.status in (ResultStatus.COMPLETED, ResultStatus.PARTIAL):
            identity = hashlib.sha256(args.model_dump_json().encode()).hexdigest()[:16]
            entry = evidence(
                task,
                f"aggregate:{log_id}:{identity}",
                str(store.manifest["source_id"]),
                str(store.manifest["source_sha256"]),
                "full selected scope; " + args.model_dump_json(),
                "tool",
                json.dumps(result.data, ensure_ascii=False),
                {
                    "generation": log_id,
                    "scope": result.data["scope"],
                    "device_id": args.device_id,
                    "synthetic": store.manifest["synthetic"],
                },
            )
            entries.append(entry)
            if args.device_id and store.manifest.get("device_id") != args.device_id:
                result.warnings.append(
                    "日志没有可靠记录所请求设备身份；筛选无匹配不代表该设备没有故障。"
                )
            facts.append(
                SourceFact(
                    text=f"所选范围包含 {result.data['total_events']} 条日志事件，"
                    f"覆盖 {result.data['physical_lines']} 行；计数单位为消息，不是独立故障。",
                    evidence_ids=[entry.evidence_id],
                )
            )
            for row in cast(list[dict[str, JsonValue]], result.data["counts"]):
                facts.append(
                    SourceFact(
                        text=f"{row['event_type']} / {row['event_state']}："
                        f"{row['messages']} 条消息。",
                        evidence_ids=[entry.evidence_id],
                    )
                )
    elif isinstance(args, FindArgs):
        result = store.find(
            args.window,
            args.limit,
            args.event_type,
            args.event_state,
            args.protocol,
            args.device_id,
            args.peer_ip,
            args.seid,
            args.teid,
        )
        for row in cast(list[dict[str, JsonValue]], result.data.get("events", [])):
            entries.append(
                evidence(
                    task,
                    str(row["event_id"]),
                    str(row["source_id"]),
                    str(row["source_sha256"]),
                    f"lines={row['line_start']}-{row['line_end']};offset={row['byte_offset']}",
                    "log",
                    str(row["raw_excerpt"]),
                    row,
                )
            )
    elif isinstance(args, ContextArgs):
        result = store.context(args.event_id, args.before, args.after)
        if result.data:
            row = cast(dict[str, JsonValue], result.data["event"])
            entries.append(
                evidence(
                    task,
                    args.event_id,
                    str(row["source_id"]),
                    str(row["source_sha256"]),
                    f"lines={row['line_start']}-{row['line_end']};offset={row['byte_offset']}",
                    "log",
                    str(result.data["raw_text"]),
                    row,
                )
            )
    elif isinstance(args, CorrelateArgs):
        if args.device_id is not None:
            return failure("scope_denied", "关联工具尚不支持设备过滤，不能扩大已限定设备范围。")
        directory = (
            root / "data/p4/topology" / task.topology_generation
            if task.topology_generation
            else None
        )
        result = correlate_events(store, args.window, directory, args.horizon_seconds, args.limit)
        gaps = list(result.warnings)
        if result.data.get("identity_gap_messages"):
            gaps.append("脱敏或缺少真实标识，无法证明告警发生与解除属于同一设备/对端。")
        return ToolPacket(result=result, gaps=gaps)
    else:
        raise ValueError("unhandled registered tool")
    return ToolPacket(
        result=result,
        evidence=entries,
        facts=facts,
        gaps=list(result.warnings)
        if result.status in (ResultStatus.NEED_INPUT, ResultStatus.MISSING_SOURCE)
        else [],
    )
