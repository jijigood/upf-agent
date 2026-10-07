"""Strict bounded task, tool, evidence and graph state contracts for P5."""

from typing import Literal

from pydantic import Field, JsonValue

from upf_agent.contracts import (
    AnswerReport,
    Evidence,
    StrictModel,
    TaskSpec,
    TimeWindow,
    ToolResult,
)

GENERATION = r"^[0-9a-f]{20}$"
RUN_ID = r"^[0-9a-f]{32}$"


class AgentBudget(StrictModel):
    max_rounds: int = Field(default=3, ge=1, le=3)
    max_tool_calls: int = Field(default=8, ge=1, le=8)
    max_model_calls: int = Field(default=8, ge=1, le=8)
    timeout_seconds: float = Field(default=180.0, gt=0, le=180)


class AgentTask(TaskSpec):
    run_id: str = Field(pattern=RUN_ID)
    question: str = Field(min_length=1, max_length=2000)
    log_generation: str | None = Field(default=None, pattern=GENERATION)
    topology_generation: str | None = Field(default=None, pattern=GENERATION)
    manual_generation: str | None = Field(default=None, pattern=GENERATION)
    model_profile: Literal["baseline", "target"] = "baseline"
    retrieval_device: Literal["cpu", "cuda:0", "cuda:1"] = "cuda:1"
    mode: Literal["agent", "deterministic"] = "agent"
    session_id: str | None = Field(default=None, min_length=1, max_length=80)
    budget: AgentBudget = Field(default_factory=AgentBudget)
    execution_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ToolCall(StrictModel):
    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class VerifiedEvidence(Evidence):
    native_id: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SourceFact(StrictModel):
    text: str = Field(min_length=1, max_length=10000)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class ToolPacket(StrictModel):
    result: ToolResult
    evidence: list[VerifiedEvidence] = Field(default_factory=list)
    facts: list[SourceFact] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    error_code: (
        Literal[
            "unknown_tool",
            "invalid_arguments",
            "scope_denied",
            "missing_source",
            "timeout",
            "corrupt_source",
            "tool_failed",
            "budget_exhausted",
        ]
        | None
    ) = None
    retryable: bool = False


class Candidate(StrictModel):
    text: str = Field(min_length=1, max_length=1000)
    supporting_evidence_ids: list[str] = Field(min_length=1, max_length=4)
    opposing_evidence_ids: list[str] = Field(default_factory=list, max_length=4)


class InvestigationPlan(StrictModel):
    actions: list[ToolCall] = Field(default_factory=list, max_length=3)
    done: bool = False
    questions: list[str] = Field(default_factory=list, max_length=3)
    candidates: list[Candidate] = Field(default_factory=list, max_length=3)


class Receipt(StrictModel):
    key: str
    call: ToolCall
    packet: ToolPacket
    cached: bool = False


class AgentData(StrictModel):
    task: AgentTask
    route: Literal["manual", "diagnose", "missing_source"] = "manual"
    receipts: list[Receipt] = Field(default_factory=list)
    evidence: list[VerifiedEvidence] = Field(default_factory=list)
    facts: list[SourceFact] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    actions: list[ToolCall] = Field(default_factory=list)
    rounds: int = 0
    repairs: int = 0
    degraded: bool = False
    terminal: bool = False
    resume_count: int = 0
    user_answers: list[str] = Field(default_factory=list)
    report: AnswerReport | None = None


class ResumeInput(StrictModel):
    answer: str = Field(default="", max_length=2000)
    log_generation: str | None = Field(default=None, pattern=GENERATION)
    topology_generation: str | None = Field(default=None, pattern=GENERATION)
    device_ids: list[str] | None = None
    time_window: TimeWindow | None = None
