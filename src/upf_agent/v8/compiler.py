"""Deterministic evidence templates and capability-bounded query compilation."""

from urllib.parse import quote

from upf_agent.v8.models import (
    Capability,
    EvidenceContract,
    Gap,
    OrchestrationTask,
    QueryPlan,
    QueryStep,
    Requirement,
    digest,
)


def slot(object_id: str, kind: str, field: str) -> str:
    return "field:" + digest([object_id, kind, field])


def object_token(object_id: str) -> str:
    return quote(object_id, safe="")


def permitted(task: OrchestrationTask, object_id: str, fields: list[str]) -> bool:
    domain = next((p.domain for p in task.path.positions if p.object_id == object_id), None)
    return (
        object_id in task.objects
        and object_id in task.permission.objects
        and (not task.permission.domains or domain in task.permission.domains)
        and set(fields) <= set(task.permission.readable_fields)
    )


def compile_contract(task: OrchestrationTask) -> EvidenceContract:
    reqs: list[Requirement] = []
    gaps: list[str] = []
    fault = task.template in {"upf_n4_fault", "upf_n3_fault"}
    if fault and (
        task.path.resolution != "resolved"
        or not task.path.source_id
        or not task.path.source_sha256
        or not task.path.version
        or not task.path.service_id
        or set(p.object_id for p in task.path.positions) != set(task.objects)
        or any(p.domain is None for p in task.path.positions)
    ):
        gaps.append("business_path_unresolved")
    for obj in task.objects:
        domain = next((p.domain for p in task.path.positions if p.object_id == obj), None)
        parts = []
        if task.template != "log_observation":
            rid = f"K:{object_token(obj)}"
            reqs.append(
                Requirement(
                    requirement_id=rid,
                    kind="K",
                    objects=[obj],
                    domains=[domain] if domain else [],
                    fields=["instruction"],
                    purpose=task.template,
                    freshness_seconds=None,
                )
            )
            parts.append(rid)
        if task.template != "manual_operation":
            fields = ["interface", "state"] if fault else ["event_type"]
            if task.template == "upf_n4_fault":
                fields.append("n4_enabled")
            elif task.template == "upf_n3_fault":
                fields.append("mtu")
            rid = f"R:{object_token(obj)}"
            reqs.append(
                Requirement(
                    requirement_id=rid,
                    kind="R",
                    objects=[obj],
                    domains=[domain] if domain else [],
                    fields=fields,
                )
            )
            parts.append(rid)
        reqs.append(
            Requirement(
                requirement_id=f"A:{object_token(obj)}",
                kind="A",
                objects=[obj],
                domains=[domain] if domain else [],
                fields=[f for r in reqs if r.objects == [obj] for f in r.fields],
                freshness_seconds=None,
                version_required=False,
            )
        )
        reqs.append(
            Requirement(
                requirement_id=f"C:{object_token(obj)}",
                kind="C",
                objects=[obj],
                upstream=parts,
                freshness_seconds=None,
            )
        )
    if fault:
        for left, right in zip(task.path.positions, task.path.positions[1:], strict=False):
            reqs.append(
                Requirement(
                    requirement_id=f"P:{object_token(left.object_id)}:{object_token(right.object_id)}",
                    kind="P",
                    objects=[left.object_id, right.object_id],
                    domains=[d for d in (left.domain, right.domain) if d],
                    fields=[
                        "service_id",
                        "local_endpoint",
                        "peer_endpoint",
                        "direction",
                        "mapping_version",
                    ],
                    direction_map={
                        "tx": "forward",
                        "rx": "reverse",
                        "egress": "forward",
                        "ingress": "reverse",
                    },
                    window_rule="intersection",
                    boundary=left.domain != right.domain,
                )
            )
    return EvidenceContract(
        contract_id=digest({"task": task, "templates": "1"})[:24],
        task_id=task.run_id,
        template=task.template,
        task_fingerprint=digest(task),
        requirements=reqs,
        gaps=gaps,
    )


def capabilities(task: OrchestrationTask) -> list[Capability]:
    result = []
    if task.manual_generation:
        result.append(
            Capability(
                name="manual",
                tool="search_manual",
                kinds=["K"],
                fields=["instruction"],
                source="manual",
            )
        )
    if task.log_generation:
        result.append(
            Capability(
                name="logs",
                tool="find_log_events",
                kinds=["R"],
                fields=["event_type", "interface", "state"],
                source="log",
            )
        )
    if task.observation_files:
        result.append(
            Capability(
                name="observation",
                tool="read_observation",
                kinds=["K", "R", "P"],
                fields=[
                    "instruction",
                    "event_type",
                    "interface",
                    "state",
                    "n4_enabled",
                    "mtu",
                    "service_id",
                    "local_endpoint",
                    "peer_endpoint",
                    "direction",
                    "mapping_version",
                ],
                source="structured_receipt",
            )
        )
    return result


def compile_query_plan(
    task: OrchestrationTask, contract: EvidenceContract, registry: list[Capability] | None = None
) -> QueryPlan:
    registry = capabilities(task) if registry is None else registry
    steps: dict[str, QueryStep] = {}
    gaps = list(contract.gaps)
    for req in contract.requirements:
        if req.kind not in {"K", "R", "P"}:
            continue
        for obj in req.objects:
            if not permitted(task, obj, req.fields):
                gaps.append(f"permission_denied:{req.requirement_id}:{obj}")
                continue
            matches = [
                c
                for c in registry
                if req.kind in c.kinds
                and (not c.objects or obj in c.objects)
                and (not c.domains or set(req.domains) <= set(c.domains))
            ]
            complete = [c for c in matches if set(req.fields) <= set(c.fields)]
            if not complete:
                absent = sorted(set(req.fields) - {f for c in matches for f in c.fields})
                gaps.append(f"capability_gap:{req.requirement_id}:{obj}:{','.join(absent)}")
            complete.sort(key=lambda c: c.tool != "read_observation")
            chosen = complete[:1] or matches[:1]
            for cap in chosen:
                key = f"{cap.tool}:{obj}"
                if key in steps:
                    steps[key].requirement_ids.append(req.requirement_id)
                    continue
                args = {}
                if cap.tool == "search_manual":
                    args = {
                        "query": task.question,
                        "product": task.product,
                        "version": task.version,
                        "top_k": 6,
                    }
                elif cap.tool == "find_log_events":
                    args = {
                        "log_id": task.log_generation,
                        "device_id": obj,
                        "window": task.time_window.model_dump(mode="json")
                        if task.time_window
                        else None,
                        "limit": 10,
                    }
                elif cap.tool == "read_observation":
                    args = {"files": task.observation_files}
                steps[key] = QueryStep(
                    step_id=digest(key)[:20],
                    requirement_ids=[req.requirement_id],
                    tool=cap.tool,
                    object_id=obj,
                    arguments=args,
                    capability=cap.name,
                )
    return QueryPlan(contract_id=contract.contract_id, steps=list(steps.values()), gaps=gaps)


def compile_gap_plan(
    task: OrchestrationTask,
    contract: EvidenceContract,
    gaps: list[Gap],
    attempted: set[tuple[str, str]],
) -> QueryPlan:
    """Try only relevant, unattempted capabilities; missing P targets only its missing side."""
    steps, unavailable = [], []
    requirements = {r.requirement_id: r for r in contract.requirements}
    registry = capabilities(task)
    for gap in sorted(gaps, key=lambda g: (g.route == "human", g.requirement_id)):
        if gap.route == "human":
            continue
        req = requirements[gap.requirement_id]
        for obj in gap.objects:
            allowed = [c for c in registry if (c.tool, obj) not in attempted]
            scoped = req.model_copy(update={"objects": [obj]})
            sub = contract.model_copy(update={"requirements": [scoped]})
            plan = compile_query_plan(task, sub, allowed)
            steps.extend(plan.steps)
            unavailable.extend(plan.gaps)
    return QueryPlan(contract_id=contract.contract_id, steps=steps, gaps=unavailable)
