"""Offline validation independent of DependencyIndex and Gold; trust comes from the runner."""

from pathlib import Path
from typing import Literal

import networkx as nx
from pydantic import Field, ValidationError

from upf_agent.v8.action_registry import ACTIONS
from upf_agent.v8.compiler import permitted
from upf_agent.v8.knowledge_applicability import annotate_knowledge
from upf_agent.v8.models import Model, digest
from upf_agent.v8.plan_ir import PlanIR
from upf_agent.v8.plan_source_verifier import verify_plan_source


class ValidationIssue(Model):
    code: str
    step_id: str | None = None
    severity: Literal["hard", "unknown"] = "hard"


class PlanValidationReport(Model):
    validator_version: str = "v8-plan-validator-1"
    validation_scope: str = "machine constraints; natural text semantics require independent review"
    raw_plan_validity: bool = False
    published_plan_validity: bool = False
    issues: list[ValidationIssue] = Field(default_factory=list)
    retained_steps: int = 0
    rejected_steps: int = 0
    rewritten_steps: int = 0
    preconditions: list[dict] = Field(default_factory=list)
    published_plan: PlanIR | None = None
    execute_authorized: Literal[False] = False


def validate_plan(plan, bindings, task, contract, context, root: Path):
    report = PlanValidationReport()

    def issue(code, step=None, severity="hard"):
        report.issues.append(ValidationIssue(code=code, step_id=step, severity=severity))

    if plan is None:
        issue("parse_failure", severity="unknown")
        return report
    from upf_agent.v8.evaluation_context import resolve_context

    resolve_context(task, contract, context.batch_id, context=context)
    if (
        plan.context_fingerprint != context.fingerprint
        or plan.batch_id != context.batch_id
        or plan.evaluation_as_of != context.as_of
        or plan.scope_fingerprint != digest(task.permission)
    ):
        issue("context_mismatch")
    if len(plan.target_objects) != len(set(plan.target_objects)):
        issue("duplicate_target_object")
    if set(plan.target_objects) != set(task.objects):
        issue("target_scope_mismatch")
    if plan.goal_achieved:
        # Planned future observations cannot certify actual diagnostic completion.
        issue("unverified_goal_claim")
    steps = {s.step_id: s for s in plan.steps}
    if len(steps) != len(plan.steps):
        issue("duplicate_step_id")
    graph = nx.DiGraph()
    graph.add_nodes_from(steps)
    for step in plan.steps:
        for predecessor in step.depends_on:
            if predecessor not in steps:
                issue("unknown_dependency", step.step_id)
            graph.add_edge(predecessor, step.step_id)
    if not nx.is_directed_acyclic_graph(graph):
        issue("dependency_cycle")
    evidence = {b.evidence_id: b for b in bindings if not b.deleted}
    eligibility = {}
    for identifier, b in evidence.items():
        reasons = []
        if b.task_id != task.run_id or b.contract_id != contract.contract_id:
            reasons.append("cross_task_evidence")
        if not verify_plan_source(root, b, task) or not b.corroborated:
            reasons.append("unverified_source")
        if not b.readable or not permitted(task, b.object_id, list(b.fields)):
            reasons.append("evidence_permission_denied")
        if b.synthetic and not task.synthetic:
            reasons.append("synthetic_in_real_task")
        if not b.version or b.version != task.version:
            reasons.append("evidence_version_mismatch")
        if b.valid and not b.valid.start <= context.as_of < b.valid.end:
            reasons.append("evidence_outside_validity")
        if b.category in {"R", "P"}:
            if not b.observed or b.observed.end > context.as_of:
                reasons.append("unknown_or_future_observation")
            elif (context.as_of - b.observed.end).total_seconds() > 300:
                reasons.append("stale_observation")
        if b.category == "K":
            checked = b.model_copy(deep=True)
            if b.knowledge_applicability is not None:
                annotate_knowledge(root, checked)
            if checked.product != task.product or task.template not in checked.purposes:
                reasons.append("unknown_knowledge_applicability")
        eligibility[identifier] = reasons
    for step in plan.steps:
        sid = step.step_id
        if not permitted(task, step.target, []):
            issue("object_permission_denied", sid)
        if step.claimed_executed or not step.non_executable or step.effect_kind == "network_write":
            issue("execution_claim_or_write", sid)
        spec = ACTIONS.get(step.action_type)
        if spec is None:
            issue("unknown_action", sid)
            continue
        schema, permission, effect, fields, needs_knowledge = spec
        if step.permission_kind != permission or step.effect_kind != effect:
            issue("permission_or_effect_mismatch", sid)
        peer = None
        try:
            params = schema.model_validate(step.parameters)
            if hasattr(params, "peer_object_id"):
                peer = params.peer_object_id
                if peer == step.target or not permitted(task, peer, fields):
                    issue("peer_permission_denied", sid)
            if hasattr(params, "field"):
                fields = [*fields, params.field]
            if hasattr(params, "interface"):
                expected = {"upf_n4_fault": "N4", "upf_n3_fault": "N3"}.get(task.template)
                if expected and params.interface != expected:
                    issue("wrong_interface_parameter", sid)
        except ValidationError:
            issue("invalid_parameters", sid)
        if not permitted(task, step.target, fields):
            issue("field_permission_denied", sid)
        refs = {ref.evidence_id for ref in step.evidence}
        good_knowledge = False
        for ref in step.evidence:
            b = evidence.get(ref.evidence_id)
            if b is None:
                issue("unknown_evidence", sid)
                continue
            if (
                ref.revision != b.revision
                or ref.source_sha256 != b.source.sha256
                or ref.locator != b.source.locator
            ):
                issue("evidence_reference_mismatch", sid)
            for reason in eligibility[b.evidence_id]:
                issue(reason, sid)
            if b.object_id != step.target and not (
                step.action_type == "VERIFY_PEER_MAPPING"
                and b.category == "P"
                and b.object_id == peer
            ):
                issue("evidence_target_mismatch", sid)
            elif b.category == "K" and not eligibility[b.evidence_id]:
                good_knowledge = True
        if needs_knowledge and not good_knowledge:
            issue("required_knowledge_missing", sid)
        for precondition in step.preconditions:
            b = evidence.get(precondition.evidence_id)
            value = "unknown"
            if precondition.evidence_id not in refs:
                issue("precondition_without_reference", sid)
            elif b and not eligibility.get(b.evidence_id) and precondition.field in b.fields:
                actual = b.fields[precondition.field]
                if actual is not None:
                    value = (
                        "true"
                        if (
                            precondition.predicate == "field_present"
                            or actual == precondition.expected
                        )
                        else "false"
                    )
            report.preconditions.append(
                {"step_id": sid, "predicate": precondition.model_dump(), "value": value}
            )
            if value == "false":
                issue("false_precondition", sid)
            # Unknown is conditional: the step's finite on_unknown outlet applies before proceeding.
    if not report.issues:
        report.raw_plan_validity = report.published_plan_validity = True
        report.retained_steps = len(plan.steps)
        report.published_plan = plan
    else:
        # Do not repair or reorder a failed plan. The entire original remains in raw output.
        report.rejected_steps = len(plan.steps)
    return report
