"""Root-isolated legacy tools and strictly sourced structured read-only observations."""

import hashlib
import json
from pathlib import Path

from upf_agent.agent_contracts import AgentTask, ToolCall
from upf_agent.agent_tools import validate_call
from upf_agent.contracts import TaskType
from upf_agent.v8.compiler import permitted
from upf_agent.v8.models import (
    EvidenceBinding,
    EvidenceContract,
    OrchestrationTask,
    QueryStep,
    SourcePointer,
    digest,
)


def source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    data_root = (root / "data").resolve()
    if not data_root.is_relative_to(root.resolve()) or not path.is_relative_to(data_root):
        raise ValueError("source must be inside the isolated data root")
    return path


def verify_binding(root: Path, binding: EvidenceBinding) -> bool:
    if not binding.source.relative_path:
        return False
    try:
        path = source_path(root, binding.source.relative_path)
        return hashlib.sha256(path.read_bytes()).hexdigest() == binding.source.sha256
    except (OSError, ValueError):
        return False


def verify_path(root: Path, task: OrchestrationTask) -> bool:
    if not task.path.source_path:
        return False
    try:
        path = source_path(root, task.path.source_path)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != task.path.source_sha256:
            return False
        body = json.loads(raw)
        data = body["path"]
        return (
            body.get("source_id") == task.path.source_id
            and body.get("synthetic", False) == task.path.synthetic
            and data["service_id"] == task.path.service_id
            and data["version"] == task.path.version
            and data["objects"] == [p.object_id for p in task.path.positions]
            and data["domains"] == [p.domain for p in task.path.positions]
        )
    except (OSError, ValueError, KeyError):
        return False


def read_observation(
    root: Path, task: OrchestrationTask, contract: EvidenceContract, step: QueryStep, receipt: str
) -> tuple[list[EvidenceBinding], list[str]]:
    bindings, gaps = [], []
    for relative in step.arguments["files"]:
        if relative not in task.observation_files:
            raise ValueError("observation source outside the declared task")
        path = source_path(root, relative)
        raw = path.read_bytes()
        body = json.loads(raw)
        # Provenance cannot be supplied through arbitrary EvidenceBinding JSON.
        synthetic = body.get("synthetic") is True
        for i, record in enumerate(body.get("observations", [])):
            obj = task.aliases.get(record.get("object_id"), record.get("object_id"))
            if obj != step.object_id:
                continue
            for req in contract.requirements:
                if (
                    req.requirement_id not in step.requirement_ids
                    or record.get("category") != req.kind
                ):
                    continue
                if req.kind == "P" and record.get("peer_object_id") is not None:
                    if record["peer_object_id"] not in req.objects:
                        continue
                if not permitted(task, obj, req.fields):
                    gaps.append(f"permission_denied:{req.requirement_id}")
                    continue
                source = SourcePointer(
                    source_id=body.get("source_id", path.name),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    locator=f"/observations/{i}",
                    relative_path=relative,
                )
                identity = digest([task.run_id, req.requirement_id, obj, relative, i])[:24]
                bindings.append(
                    EvidenceBinding(
                        evidence_id=f"{task.run_id}:{identity}",
                        native_id=identity,
                        task_id=task.run_id,
                        contract_id=contract.contract_id,
                        requirement_id=req.requirement_id,
                        object_id=obj,
                        category=req.kind,
                        fields={
                            f: v
                            for f, v in record.get("fields", {}).items()
                            if f in task.permission.readable_fields
                        },
                        source=source,
                        product=record.get("product"),
                        purposes=record.get("purposes", []),
                        version=record.get("version"),
                        config_version=record.get("config_version"),
                        observed=record.get("observed"),
                        valid=record.get("valid"),
                        receipt_id=receipt,
                        synthetic=synthetic,
                        source_verified=True,
                        corroborated=body.get("corroborated", False),
                    )
                )
    return bindings, gaps


def legacy_task(task: OrchestrationTask, obj: str) -> AgentTask:
    return AgentTask(
        run_id=task.run_id,
        question=task.question,
        task_type=TaskType.MANUAL if task.template == "manual_operation" else TaskType.DIAGNOSE,
        device_ids=[obj],
        product=task.product,
        version=task.version,
        time_window=task.time_window,
        log_generation=task.log_generation,
        topology_generation=task.topology_generation,
        manual_generation=task.manual_generation,
        retrieval_device=task.retrieval_device,
        mode="deterministic",
        budget=task.budget,
    )


def collect(
    root: Path,
    task: OrchestrationTask,
    contract: EvidenceContract,
    step: QueryStep,
    receipt: str,
    timeout: float = 180,
) -> dict:
    if step.object_id not in task.objects or step.object_id not in task.permission.objects:
        raise ValueError("query object outside authorized scope")
    if step.tool == "read_observation":
        bindings, gaps = read_observation(root, task, contract, step, receipt)
        return {"bindings": [b.model_dump(mode="json") for b in bindings], "gaps": gaps}
    scoped = legacy_task(task, step.object_id)
    call = ToolCall(name=step.tool, arguments=step.arguments)
    _, error = validate_call(scoped, call, [])
    if error:
        return {"bindings": [], "gaps": [error]}
    from upf_agent.agent_runtime import LocalBackend

    packet = LocalBackend(root).tool(scoped, call, [], timeout)
    bindings = []
    for evidence in packet.evidence:
        for req in contract.requirements:
            if req.requirement_id not in step.requirement_ids or req.kind not in {"K", "R"}:
                continue
            meta = evidence.metadata
            observed = None
            locator = evidence.locator
            obj = step.object_id if req.kind == "K" else meta.get("device_id")
            if req.kind == "K":
                fields = {"instruction": evidence.excerpt}
                # A manual is mapped to the requested object only after product/version checks.
                from upf_agent.p3_index import load_active_index, load_chunks

                index, _ = load_active_index(root)
                chunk = load_chunks(index).get(evidence.native_id)
                relative = f"data/raw/{chunk.relative_path}" if chunk else None
                product, version = (chunk.product, chunk.version) if chunk else (None, None)
                if chunk:
                    locator = ";".join(chunk.locators) or chunk.chunk_id
            else:
                fields = {
                    "event_type": meta.get("event_type"),
                    "interface": meta.get("interface_type"),
                    "state": meta.get("event_state"),
                }
                product, version = None, None
                relative = "data/raw/upfd_masked.log"
                # A parsed timestamp without an explicit observation interval cannot satisfy R.
            binding = EvidenceBinding(
                evidence_id=f"{evidence.evidence_id}:{req.requirement_id}",
                native_id=evidence.native_id,
                task_id=task.run_id,
                contract_id=contract.contract_id,
                requirement_id=req.requirement_id,
                object_id=obj,
                category=req.kind,
                fields=fields,
                source=SourcePointer(
                    source_id=evidence.source_id,
                    sha256=evidence.source_sha256,
                    locator=locator,
                    relative_path=relative,
                ),
                product=product,
                version=version,
                observed=observed,
                receipt_id=receipt,
                synthetic=meta.get("synthetic") is True,
            )
            binding.source_verified = verify_binding(root, binding)
            if binding.category == "K":
                from upf_agent.v8.knowledge_applicability import annotate_knowledge

                annotate_knowledge(root, binding)
            bindings.append(binding)
    return {
        "bindings": [b.model_dump(mode="json") for b in bindings],
        "gaps": packet.gaps + ([packet.error_code] if packet.error_code else []),
        "legacy_packet": packet.model_dump(mode="json"),
    }
