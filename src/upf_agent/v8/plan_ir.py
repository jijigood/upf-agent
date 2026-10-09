"""Bounded plan and independent reference schemas; model fields are untrusted."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from upf_agent.v8.models import Model

PLAN_VERSION = "v8-plan-ir-1"
REGISTRY_VERSION = "v8-action-registry-1"


class EvidenceReference(Model):
    evidence_id: str
    revision: int = Field(ge=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locator: str = Field(min_length=1)


class Precondition(Model):
    predicate: Literal["field_equals", "field_present"]
    evidence_id: str
    field: str
    expected: Any = None


class PlanStep(Model):
    step_id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=4000)
    action_type: str
    target: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    preconditions: list[Precondition] = Field(default_factory=list)
    on_unknown: Literal["request_evidence", "human_review", "stop"] = "stop"
    expected_observations: list[str] = Field(default_factory=list)
    permission_kind: Literal["read", "suggest", "human"] = "read"
    effect_kind: Literal["read", "suggestion", "network_write"] = "read"
    claimed_executed: bool = False
    non_executable: bool = True


class PlanIR(Model):
    schema_version: Literal["v8-plan-ir-1"] = PLAN_VERSION
    registry_version: Literal["v8-action-registry-1"] = REGISTRY_VERSION
    context_fingerprint: str
    scope_fingerprint: str
    batch_id: str
    evaluation_as_of: datetime
    target_objects: list[str]
    steps: list[PlanStep] = Field(min_length=1, max_length=24)
    stop_conditions: list[str] = Field(min_length=1)
    goal_achieved: bool = False

    @model_validator(mode="after")
    def timezone(self):
        if self.evaluation_as_of.utcoffset() is None:
            raise ValueError("plan evaluation time requires timezone")
        return self


class RawPlanOutput(Model):
    method: str
    model_id: str
    content: str
    content_sha256: str
    request_sha256: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None


class StepSpan(Model):
    step_id: str
    start: int
    end: int
    text_start: int
    text_end: int


class TranslationReceipt(Model):
    translator_version: str = "v8-json-translator-1"
    raw_sha256: str
    plan: PlanIR | None = None
    spans: list[StepSpan] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    cost_model_calls: int = 0


class ActionPattern(Model):
    pattern_id: str
    action_type: str
    target: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class GoldBranch(Model):
    branch_id: str
    required: list[ActionPattern] = Field(min_length=1)
    optional: list[ActionPattern] = Field(default_factory=list)

    @model_validator(mode="after")
    def ids(self):
        ids = [p.pattern_id for p in self.required]
        if len(ids) != len(set(ids)) or any(set(p.depends_on) - set(ids) for p in self.required):
            raise ValueError("invalid Gold dependency references")
        return self


class CaseGold(Model):
    schema_version: str = "v8-independent-gold-1"
    case_id: str
    source_group: str
    structure_family: str
    source_level: Literal["D1", "D2", "D3", "D4"]
    split: Literal["dev", "validation", "heldout"] = "dev"
    synthetic: bool
    reviewed: bool = False
    reviewers: list[str] = Field(default_factory=list)
    review_version: str = "unreviewed"
    solvability: Literal["solvable", "needs_evidence", "human_review"]
    objects: list[str]
    version: str | None
    as_of: datetime
    permission_fingerprint: str
    visible_evidence: list[str]
    source_hashes: dict[str, str]
    any_of: list[GoldBranch] = Field(min_length=1)
    forbidden_actions: list[str] = Field(default_factory=lambda: ["EXECUTE_CHANGE"])
    stop_conditions: list[str] = Field(min_length=1)
    root_cause_truth: Literal["unknown", "expert_verified", "experimental_injection"] = "unknown"

    @model_validator(mode="after")
    def reviewed_basis(self):
        if self.reviewed and (len(set(self.reviewers)) < 2 or self.review_version == "unreviewed"):
            raise ValueError("reviewed Gold requires two independent reviewers and version")
        if self.as_of.utcoffset() is None:
            raise ValueError("Gold time requires timezone")
        return self
