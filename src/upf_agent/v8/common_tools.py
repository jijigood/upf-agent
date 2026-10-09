"""Public synthetic source adapter. No Contract states or Gold enter P5 tool responses."""

import hashlib
import json
from pathlib import Path

from upf_agent.agent_contracts import ToolPacket
from upf_agent.agent_tools import evidence, failure, validate_call
from upf_agent.contracts import ResultStatus, ToolResult
from upf_agent.v8.adapters import source_path
from upf_agent.v8.compiler import permitted
from upf_agent.v8.knowledge_applicability import annotate_knowledge
from upf_agent.v8.models import EvidenceBinding, SourcePointer


class CommonEvidenceTools:
    version = "v8-common-synthetic-tools-2"

    def __init__(self, root: Path, task, contract, budget):
        if not task.synthetic:
            raise ValueError("public synthetic adapter requires a synthetic task")
        self.root, self.task, self.contract, self.budget = root, task, contract, budget
        self.bindings = {}
        self.source_hashes = {
            relative: hashlib.sha256(source_path(root, relative).read_bytes()).hexdigest()
            for relative in task.observation_files
        }

    def _rows(self):
        for relative in self.task.observation_files:
            path = source_path(self.root, relative)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.source_hashes[relative]:
                raise ValueError("frozen source changed during method run")
            body = json.loads(raw)
            if body.get("synthetic") is not True:
                raise ValueError("synthetic tool adapter requires explicit synthetic source")
            for index, record in enumerate(body.get("observations", [])):
                obj = record.get("object_id")
                if not permitted(self.task, obj, []):
                    continue
                visible = {
                    **record,
                    "fields": {
                        k: v
                        for k, v in record.get("fields", {}).items()
                        if k in self.task.permission.readable_fields
                    },
                }
                yield relative, body, index, visible, hashlib.sha256(raw).hexdigest()

    def tool(self, task, call, known, timeout):
        args, error = validate_call(task, call, known)
        if error:
            return failure(error, error)
        request = call.model_dump(mode="json")
        self.budget.check("tool", request)
        with self.budget.collector.measure("tool", call.name, request):
            if call.name == "compare_evidence":
                return ToolPacket(
                    result=ToolResult(
                        status=ResultStatus.COMPLETED,
                        elapsed_ms=0,
                        data={
                            "comparisons": [
                                e.model_dump(mode="json")
                                for e in known
                                if e.evidence_id in args.evidence_ids
                            ]
                        },
                    )
                )
            categories = {
                "search_manual": {"K"},
                "read_manual_block": {"K"},
                "analyze_log_window": {"R"},
                "find_log_events": {"R"},
                "read_log_context": {"R"},
                "lookup_topology": {"P"},
                "correlate_events": {"R"},
            }
            if call.name not in categories:
                return failure("unknown_tool", call.name)
            obj = getattr(args, "device_id", None)
            if obj is not None and not permitted(self.task, obj, []):
                return failure("scope_denied", "object outside experiment scope")
            entries = []
            for relative, body, index, record, sha in self._rows():
                if record.get("category") not in categories[call.name] or (
                    obj and record["object_id"] != obj
                ):
                    continue
                native = f"{Path(relative).stem}:observation:{index}"
                if call.name == "read_manual_block" and (
                    native != args.chunk_id
                    or (args.source_id and args.source_id != body["source_id"])
                ):
                    continue
                if call.name == "read_log_context" and native != args.event_id:
                    continue
                if (
                    getattr(args, "event_type", None)
                    and record["fields"].get("event_type") != args.event_type
                ):
                    continue
                limit = getattr(args, "top_k", getattr(args, "limit", 50))
                if len(entries) >= limit:
                    break
                entry = evidence(
                    task,
                    native,
                    body["source_id"],
                    sha,
                    f"/observations/{index}",
                    "manual"
                    if record["category"] == "K"
                    else "log"
                    if record["category"] == "R"
                    else "topology",
                    json.dumps(record, ensure_ascii=False),
                    {
                        "device_id": record["object_id"],
                        "relative_path": relative,
                        "synthetic": True,
                        "record": record,
                    },
                )
                entries.append(entry)
                for req in self.contract.requirements:
                    if req.kind != record["category"] or record["object_id"] not in req.objects:
                        continue
                    b = EvidenceBinding(
                        evidence_id=f"{entry.evidence_id}:{req.requirement_id}",
                        native_id=native,
                        task_id=self.task.run_id,
                        contract_id=self.contract.contract_id,
                        requirement_id=req.requirement_id,
                        object_id=record["object_id"],
                        category=req.kind,
                        fields=record["fields"],
                        source=SourcePointer(
                            source_id=body["source_id"],
                            sha256=sha,
                            locator=f"/observations/{index}",
                            relative_path=relative,
                        ),
                        # Knowledge documents use their validity/version, not runtime
                        # observation windows. This matches graph document projection.
                        observed=record.get("observed") if req.kind in {"R", "P"} else None,
                        valid=record.get("valid"),
                        product=record.get("product"),
                        version=record.get("version"),
                        config_version=record.get("config_version"),
                        receipt_id=entry.evidence_id,
                        synthetic=True,
                        source_verified=True,
                        purposes=[],
                        corroborated=body.get("corroborated") is True,
                    )
                    if b.category == "K":
                        annotate_knowledge(self.root, b)
                    self.bindings[b.evidence_id] = b
            return ToolPacket(
                result=ToolResult(
                    status=ResultStatus.COMPLETED if entries else ResultStatus.MISSING_SOURCE,
                    elapsed_ms=0,
                    evidence_ids=[e.evidence_id for e in entries],
                ),
                evidence=entries,
                gaps=[] if entries else ["no matching permitted source"],
            )
