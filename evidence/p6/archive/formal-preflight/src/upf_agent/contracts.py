"""Shared contracts for all later migration stages."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TaskType(StrEnum):
    MANUAL = "manual"
    DIAGNOSE = "diagnose"
    NETWORK_STATUS = "network_status"
    OTHER = "other"


class ResultStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    NEED_INPUT = "need_input"
    MISSING_SOURCE = "missing_source"
    FAILED = "failed"


class TimeWindow(StrictModel):
    start: datetime
    end: datetime
    timezone_source: str

    @model_validator(mode="after")
    def check_window(self) -> "TimeWindow":
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("time window requires explicit timezone offsets")
        if self.start >= self.end:
            raise ValueError("time window must be half-open with start < end")
        return self


class TaskSpec(StrictModel):
    run_id: str
    question: str = Field(min_length=1)
    task_type: TaskType
    attachment_ids: list[str] = Field(default_factory=list)
    product: str | None = None
    version: str | None = None
    device_ids: list[str] = Field(default_factory=list)
    time_window: TimeWindow | None = None
    interactive: bool = False

    @field_validator("task_type", mode="before")
    @classmethod
    def parse_task_type(cls, value: str | TaskType) -> TaskType:
        return TaskType(value)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question cannot be blank")
        return value


class SourceRef(StrictModel):
    source_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    relative_path: str
    role: Literal["manual", "log", "paper", "topology", "fixture"]
    product: str | None = None
    version: str | None = None


class DocumentBlock(StrictModel):
    source_id: str
    block_id: str
    parent_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    kind: Literal["paragraph", "table", "table_row", "textbox", "image", "image_ocr", "code"]
    raw_text: str
    normalized_text: str
    locator: str
    extraction_method: str
    quality_flags: list[str] = Field(default_factory=list)
    ordinal: int = 0
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CommandRecord(StrictModel):
    command_id: str
    source_id: str
    family: Literal["MML", "CLI", "VPP", "other"]
    syntax: str
    parameter_block_ids: list[str] = Field(default_factory=list)
    precondition_block_ids: list[str] = Field(default_factory=list)
    example_block_ids: list[str] = Field(default_factory=list)
    product: str | None = None
    version: str | None = None
    syntax_block_id: str | None = None
    parameter_tokens: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)


class LogEvent(StrictModel):
    event_id: str
    source_id: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    byte_offset: int = Field(ge=0)
    byte_length: int = Field(default=0, ge=0)
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    timestamp_original: str | None = None
    timestamp_utc: datetime | None = None
    timezone_status: Literal["explicit", "assumed", "unknown"] = "unknown"
    device_id: str | None = None
    protocol: str | None = None
    event_type: str | None = None
    event_state: Literal["raised", "cleared", "observed", "unknown"] = "unknown"
    severity_raw: str | None = None
    peer_ip: str | None = None
    session: str | None = None
    seid: str | None = None
    teid: str | None = None
    imsi: str | None = None
    interface_type: str | None = None
    port: str | None = None
    rule_id: str | None = None
    classification_basis: str | None = None
    template_id: str | None = None
    parse_status: Literal["parsed", "blank", "quarantined"] = "parsed"
    quality_flags: list[str] = Field(default_factory=list)
    masked_fields: list[str] = Field(default_factory=list)
    raw_excerpt: str
    parser_version: str

    @model_validator(mode="after")
    def check_lines(self) -> "LogEvent":
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        if self.timestamp_utc is not None and self.timestamp_utc.utcoffset() is None:
            raise ValueError("timestamp_utc must have a timezone")
        return self


class TopologyEdge(StrictModel):
    edge_id: str
    device_id: str
    device_port: str
    peer_device_id: str
    peer_port: str
    interface_type: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_id: str
    source_sheet: str | None = None
    source_row: int | None = None
    synthetic: bool = False
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    vlan: str | None = None
    purpose: str | None = None
    quality_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_validity(self) -> "TopologyEdge":
        for value in (self.valid_from, self.valid_to):
            if value is not None and value.utcoffset() is None:
                raise ValueError("topology validity requires explicit timezone")
        if self.valid_from is not None and self.valid_to is not None:
            if self.valid_from >= self.valid_to:
                raise ValueError("topology validity must be half-open")
        return self


class Evidence(StrictModel):
    evidence_id: str
    source_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locator: str
    kind: Literal["manual", "log", "topology", "tool"]
    excerpt: str
    tool_version: str
    applicability_note: str | None = None


class ToolResult(StrictModel):
    status: ResultStatus
    data: dict[str, JsonValue] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    elapsed_ms: int = Field(ge=0)

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, value: str | ResultStatus) -> ResultStatus:
        return ResultStatus(value)


class AnswerReport(StrictModel):
    run_id: str
    status: ResultStatus
    facts: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    opposing_evidence_ids: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    model_id: str | None = None
    corpus_generation: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, value: str | ResultStatus) -> ResultStatus:
        return ResultStatus(value)
