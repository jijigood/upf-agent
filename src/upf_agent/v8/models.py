"""Serializable contracts shared by the compiler, evidence ledger and graph backends."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from upf_agent.agent_contracts import AgentBudget
from upf_agent.contracts import TimeWindow


def canonical(value: Any) -> str:
    def encode(obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"cannot canonicalize {type(obj).__name__}")

    return orjson.dumps(value, default=encode, option=orjson.OPT_SORT_KEYS).decode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Code(StrEnum):
    SATISFIED = "satisfied"
    MISSING = "missing"
    UNKNOWN = "unknown"
    STALE = "stale"
    CONFLICT = "conflict"
    NOT_APPLICABLE = "not_applicable"
    DENIED = "denied"


class PathPosition(Model):
    object_id: str = Field(min_length=1)
    domain: str | None = None
    interface: Literal["N3", "N4", "N6", "N9"] | None = None


class BusinessPath(Model):
    service_id: str | None = None
    positions: list[PathPosition] = Field(default_factory=list)
    source_id: str | None = None
    source_sha256: str | None = None
    source_path: str | None = None
    version: str | None = None
    resolution: Literal["resolved", "candidate", "unknown"] = "unknown"
    candidates: list[list[str]] = Field(default_factory=list)
    synthetic: bool = False


class PermissionScope(Model):
    objects: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    readable_fields: list[str] = Field(default_factory=list)
    revision: str = "1"
    # Read permission is independent of an execution authorization.
    execute: Literal[False] = False


class OrchestrationTask(Model):
    run_id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[0-9a-f]{32}$")
    question: str = Field(min_length=1, max_length=2000)
    template: Literal["upf_n4_fault", "upf_n3_fault", "manual_operation", "log_observation"]
    path: BusinessPath = Field(default_factory=BusinessPath)
    objects: list[str] = Field(min_length=1)
    product: str | None = None
    version: str | None = None
    time_window: TimeWindow | None = None
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))
    permission: PermissionScope = Field(default_factory=PermissionScope)
    aliases: dict[str, str] = Field(default_factory=dict)
    synthetic: bool = False
    budget: AgentBudget = Field(default_factory=AgentBudget)
    log_generation: str | None = None
    topology_generation: str | None = None
    manual_generation: str | None = None
    observation_files: list[str] = Field(default_factory=list)
    model_profile: Literal["baseline", "target"] = "baseline"
    model_enabled: bool = False
    retrieval_device: Literal["cpu", "cuda:0", "cuda:1"] = "cuda:1"
    interrupt_before_collect: bool = False
    graph_backend: Literal["memory", "neo4j"] = "memory"
    graph_enabled: bool = True
    execution_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("time_window", mode="before")
    @classmethod
    def decode_window(cls, value):
        return (
            TimeWindow.model_validate_json(canonical(value)) if isinstance(value, dict) else value
        )

    @model_validator(mode="after")
    def scope(self) -> OrchestrationTask:
        if self.as_of.utcoffset() is None:
            raise ValueError("as_of requires an explicit timezone")
        if len(set(self.objects)) != len(self.objects):
            raise ValueError("duplicate objects")
        if self.path.synthetic and not self.synthetic:
            raise ValueError("a real task cannot declare a synthetic resolved path")
        return self


class Requirement(Model):
    requirement_id: str
    kind: Literal["K", "R", "P", "A", "C"]
    objects: list[str]
    domains: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    upstream: list[str] = Field(default_factory=list)
    mandatory: bool = True
    freshness_seconds: int | None = 300
    version_required: bool = True
    source_priority: list[str] = Field(default_factory=list)
    direction_map: dict[str, str] = Field(default_factory=dict)
    window_rule: Literal["equal", "intersection"] = "equal"
    boundary: bool = False
    purpose: str | None = None


class EvidenceContract(Model):
    contract_id: str
    version: str = "1"
    task_id: str
    template: str
    task_fingerprint: str
    requirements: list[Requirement]
    gaps: list[str] = Field(default_factory=list)


class EvaluationContext(Model):
    """Trusted evaluation inputs; batch_id certifies reuse, not recomputation."""

    batch_id: str = Field(min_length=1)
    as_of: datetime
    task_fingerprint: str
    contract_fingerprint: str
    permission_fingerprint: str
    path_fingerprint: str
    policy_version: str = "v8-evaluation-2"

    @model_validator(mode="after")
    def aware_time(self):
        if self.as_of.utcoffset() is None:
            raise ValueError("evaluation time requires an explicit timezone")
        return self

    @classmethod
    def from_task(cls, task, contract, batch, now=None):
        return cls(
            batch_id=batch,
            as_of=now if now is not None else task.as_of,
            task_fingerprint=digest(task),
            contract_fingerprint=digest(contract),
            permission_fingerprint=digest(task.permission),
            path_fingerprint=digest(task.path),
        )

    @property
    def fingerprint(self):
        return digest(self)


class Capability(Model):
    name: str
    tool: str
    kinds: list[str]
    fields: list[str]
    objects: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    source: str


class QueryStep(Model):
    step_id: str
    requirement_ids: list[str]
    tool: str
    object_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    capability: str


class QueryPlan(Model):
    contract_id: str
    steps: list[QueryStep]
    gaps: list[str]


class SourcePointer(Model):
    source_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locator: str = Field(min_length=1)
    relative_path: str | None = None


class EvidenceBinding(Model):
    evidence_id: str
    native_id: str
    task_id: str
    contract_id: str
    requirement_id: str
    object_id: str | None = None
    category: Literal["K", "R", "P"]
    fields: dict[str, Any]
    source: SourcePointer
    revision: int = Field(default=1, ge=1)
    observed: TimeWindow | None = None
    valid: TimeWindow | None = None
    product: str | None = None
    version: str | None = None
    config_version: str | None = None
    role: str = "collector"
    knowledge_applicability: dict[str, Any] | None = None
    purposes: list[str] = Field(default_factory=list)
    readable: bool = True
    receipt_id: str
    source_verified: bool = False
    synthetic: bool = False
    corroborated: bool = True
    deleted: bool = False

    @field_validator("observed", "valid", mode="before")
    @classmethod
    def decode_window(cls, value):
        return (
            TimeWindow.model_validate_json(canonical(value)) if isinstance(value, dict) else value
        )


class RequirementState(Model):
    requirement_id: str
    code: Code
    reasons: list[str]
    supporting: list[str] = Field(default_factory=list)
    opposing: list[str] = Field(default_factory=list)
    basis: dict[str, Any] = Field(default_factory=dict)
    state_version: str
    valid: bool
    batch_id: str
    pending: bool = False
    computed_batch_id: str | None = None
    evaluation_as_of: datetime | None = None
    context_fingerprint: str | None = None

    def semantic(self) -> dict:
        return self.model_dump(
            mode="json",
            exclude={
                "batch_id",
                "state_version",
                "computed_batch_id",
                "evaluation_as_of",
                "context_fingerprint",
            },
        )


class FieldPredicate(Model):
    object_id: str
    category: Literal["K", "R", "P"] = "R"
    field: str
    operator: Literal["eq", "ge", "present"] = "present"
    value: Any = None


class ActionTemplate(Model):
    action_id: str
    name: Literal["CHECK_N4_CONFIGURATION", "CHECK_INTERFACE_STATUS", "VERIFY_PEER_MAPPING"]
    target: str
    parameters: dict[str, Any]
    required_states: list[str]
    predicates: list[FieldPredicate]
    allowed_relations: list[str] = Field(default_factory=lambda: ["CARRIES", "DEPENDS_ON"])
    cutoff_relations: list[str] = Field(default_factory=lambda: ["ADMIN_BOUNDARY"])
    prerequisite_objects: list[str] = Field(default_factory=list)


class ActionDecision(Model):
    action_id: str
    outcome: Literal["suggest_allowed", "re_evidence", "human_review", "blocked"]
    reasons: list[str]
    state_ids: list[str]
    fields: dict[str, Any]
    impact_objects: list[str]
    path_intervals: list[list[int]]
    batch_id: str
    execute_authorized: Literal[False] = False
    computed_batch_id: str | None = None
    evaluation_as_of: datetime | None = None
    context_fingerprint: str | None = None


class Gap(Model):
    requirement_id: str
    route: Literal[
        "knowledge_search", "runtime_refresh", "missing_side", "bilateral_review", "human"
    ]
    objects: list[str]
    reasons: list[str]


class OrchestrationData(Model):
    task: OrchestrationTask
    contract: EvidenceContract | None = None
    query_plan: QueryPlan | None = None
    bindings: list[EvidenceBinding] = Field(default_factory=list)
    states: dict[str, RequirementState] = Field(default_factory=dict)
    actions: list[ActionDecision] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    batch_id: str = ""
    rounds: int = 0
    status: str = "created"
    stop_reason: str | None = None
    path_vector: list[dict[str, Any]] = Field(default_factory=list)
