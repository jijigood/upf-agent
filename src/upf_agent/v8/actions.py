"""Read-only action templates, propagation scopes and value-sensitive admission gate."""

import networkx as nx

from upf_agent.v8.compiler import object_token
from upf_agent.v8.evaluation_context import bind_snapshot, certify, resolve_context
from upf_agent.v8.evaluator import EvidenceSnapshot, TrackedStates, record_basis, select
from upf_agent.v8.models import (
    ActionDecision,
    ActionTemplate,
    Code,
    EvidenceContract,
    FieldPredicate,
    OrchestrationTask,
    RequirementState,
    canonical,
    digest,
)


def action_templates(task: OrchestrationTask, contract: EvidenceContract) -> list[ActionTemplate]:
    result = []
    for obj in task.objects:
        if task.template == "upf_n4_fault":
            name, field, operator, value = "CHECK_N4_CONFIGURATION", "n4_enabled", "eq", True
        elif task.template == "upf_n3_fault":
            name, field, operator, value = "CHECK_INTERFACE_STATUS", "mtu", "ge", 1280
        elif task.template == "manual_operation":
            name, field, operator, value = "CHECK_INTERFACE_STATUS", "instruction", "present", None
        else:
            name, field, operator, value = "CHECK_INTERFACE_STATUS", "event_type", "present", None
        result.append(
            ActionTemplate(
                action_id=f"action:{object_token(obj)}",
                name=name,
                target=obj,
                parameters={
                    "interface": next(
                        (p.interface for p in task.path.positions if p.object_id == obj), None
                    ),
                    "mode": "read_only",
                },
                required_states=[
                    r.requirement_id
                    for r in contract.requirements
                    if obj in r.objects and r.kind in {"C", "P", "A"}
                ],
                predicates=[
                    FieldPredicate(
                        object_id=obj,
                        category="K" if field == "instruction" else "R",
                        field=field,
                        operator=operator,
                        value=value,
                    )
                ],
            )
        )
    if task.template in {"upf_n4_fault", "upf_n3_fault"}:
        for req in contract.requirements:
            if req.kind == "P":
                result.append(
                    ActionTemplate(
                        action_id=f"action:{req.requirement_id}",
                        name="VERIFY_PEER_MAPPING",
                        target=req.objects[0],
                        parameters={"peer": req.objects[1], "mode": "read_only"},
                        required_states=[req.requirement_id]
                        + [f"A:{object_token(o)}" for o in req.objects],
                        predicates=[
                            FieldPredicate(
                                object_id=o,
                                category="P",
                                field="service_id",
                                operator="eq",
                                value=task.path.service_id,
                            )
                            for o in req.objects
                        ],
                    )
                )
    return result


def propagation_graph(task: OrchestrationTask) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(task.objects)
    for a, b in zip(task.path.positions, task.path.positions[1:], strict=False):
        relation = "CARRIES" if a.domain == b.domain else "ADMIN_BOUNDARY"
        graph.add_edge(a.object_id, b.object_id, relation=relation)
        graph.add_edge(b.object_id, a.object_id, relation=relation)
    return graph


def impact_scope(
    action: ActionTemplate, task: OrchestrationTask, graph: nx.DiGraph
) -> tuple[list[str], list[list[int]]]:
    found, queue = {action.target}, [action.target]
    while queue:
        obj = queue.pop()
        if obj not in graph:
            continue
        for other in graph.successors(obj):
            relation = graph[obj][other].get("relation")
            if relation in action.allowed_relations and relation not in action.cutoff_relations:
                if other not in found:
                    found.add(other)
                    queue.append(other)
    found.update(action.prerequisite_objects)
    positions = [i for i, p in enumerate(task.path.positions) if p.object_id in found]
    intervals: list[list[int]] = []
    for i in positions:
        if intervals and intervals[-1][1] == i:
            intervals[-1][1] = i + 1
        else:
            intervals.append([i, i + 1])
    return sorted(found), intervals


def gate(
    action: ActionTemplate,
    task: OrchestrationTask,
    contract: EvidenceContract,
    states: dict[str, RequirementState],
    snap: EvidenceSnapshot,
    batch: str,
    graph: nx.DiGraph | None = None,
    *,
    now=None,
    context=None,
) -> ActionDecision:
    context = resolve_context(task, contract, batch, now, context)
    return _gate(action, task, contract, states, snap, batch, graph, context)


def _gate(action, task, contract, states, snap, batch, graph, evaluation):
    bind_snapshot(snap, evaluation)
    objects, intervals = impact_scope(
        action, task, graph if graph is not None else propagation_graph(task)
    )
    if not hasattr(snap, "contract_lookup"):
        by_object = {}
        for req in contract.requirements:
            for obj in req.objects:
                by_object.setdefault(obj, []).append(req)
        snap.contract_lookup = by_object
    needed = set(action.required_states)
    for obj in objects:
        needed.update(r.requirement_id for r in snap.contract_lookup.get(obj, []) if r.mandatory)
    reasons = list(contract.gaps)
    human = bool(contract.gaps)
    blocked = False
    values = {}
    # One EvidenceSnapshot belongs to one immutable contract/task/batch.
    if not hasattr(snap, "gate_requirements"):
        snap.gate_requirements = {r.requirement_id: r for r in contract.requirements}
        snap.gate_context = {
            "contract": contract.contract_id,
            "contract_semantics": digest(contract),
            "permission": digest(task.permission),
            "path": digest(task.path),
        }
    by_id, context = snap.gate_requirements, snap.gate_context
    if not hasattr(snap, "basis_cache"):
        snap.basis_cache = {}
    for obj in objects:
        if obj not in task.objects or obj not in task.permission.objects:
            reasons.append(f"outside_authorized_scope:{obj}")
            blocked = True
        if not any(r.kind == "A" for r in snap.contract_lookup.get(obj, [])):
            reasons.append(f"prerequisite_contract_missing:{obj}")
            human = True
    for rid in sorted(needed):
        state = states.get(rid)
        if state is None or state.pending or state.batch_id != batch:
            reasons.append(f"pending_or_mixed_batch:{rid}")
            human = True
        elif (
            state.context_fingerprint != evaluation.fingerprint
            or state.evaluation_as_of != evaluation.as_of
        ):
            reasons.append(f"stale_evaluation_context_basis:{rid}")
            human = True
        elif state.code != Code.SATISFIED or not state.valid:
            reasons.append(f"unsatisfied:{rid}:{state.code.value}")
            blocked |= state.code == Code.DENIED
            human |= state.code in {Code.CONFLICT, Code.UNKNOWN}
        elif any(state.basis.get(k) != v for k, v in context.items()):
            reasons.append(f"stale_context_basis:{rid}")
            human = True
        elif rid in by_id and by_id[rid].kind in {"K", "R", "P"}:
            req = by_id[rid]
            if rid not in snap.basis_cache:
                current = {}
                for obj in req.objects:
                    selected, excluded = select(req, obj, task, contract, snap, evaluation.as_of)
                    current[obj] = {
                        "eligible": [record_basis(b, req) for b in selected],
                        "excluded": excluded,
                    }
                snap.basis_cache[rid] = current
            if state.basis.get("records") != snap.basis_cache[rid]:
                reasons.append(f"evidence_revision_basis_mismatch:{rid}")
                human = True
    for pred in action.predicates:
        if not hasattr(snap, "permission_inputs"):
            snap.permission_inputs = (
                set(task.objects),
                set(task.permission.objects),
                set(task.permission.domains),
                set(task.permission.readable_fields),
                {p.object_id: p.domain for p in task.path.positions},
            )
        declared, readable, domains, fields, domain_map = snap.permission_inputs
        if not (
            pred.object_id in declared
            and pred.object_id in readable
            and pred.field in fields
            and (not domains or domain_map.get(pred.object_id) in domains)
        ):
            reasons.append(f"predicate_read_permission_denied:{pred.object_id}:{pred.field}")
            blocked = True
        snap.get(pred.object_id, pred.category, pred.field)
        reqs = [r for r in snap.contract_lookup.get(pred.object_id, []) if r.kind == pred.category]
        candidates = []
        for req in reqs:
            selected, _ = select(req, pred.object_id, task, contract, snap, evaluation.as_of)
            candidates.extend(selected)
        field_values = {
            canonical(b.fields[pred.field]): b.fields[pred.field]
            for b in candidates
            if pred.field in b.fields and b.fields[pred.field] is not None
        }
        values[f"{pred.object_id}:{pred.category}:{pred.field}"] = sorted(field_values)
        passed = len(field_values) == 1
        if passed:
            actual = next(iter(field_values.values()))
            if pred.operator == "eq":
                passed = canonical(actual) == canonical(pred.value)
            elif pred.operator == "ge":
                passed = type(actual) in {int, float} and actual >= pred.value
            else:
                passed = actual != ""
        if not passed:
            reasons.append(f"predicate_failed:{pred.object_id}:{pred.field}:{pred.operator}")
    outcome = (
        "blocked"
        if blocked
        else "human_review"
        if human
        else ("re_evidence" if reasons else "suggest_allowed")
    )
    return certify(
        ActionDecision(
            action_id=action.action_id,
            outcome=outcome,
            reasons=reasons,
            state_ids=sorted(needed),
            fields=values,
            impact_objects=objects,
            path_intervals=intervals,
            batch_id=batch,
        ),
        evaluation,
        computed=True,
    )


def full_actions(
    task: OrchestrationTask,
    contract: EvidenceContract,
    bindings,
    states: dict[str, RequirementState],
    batch: str,
    actions: list[ActionTemplate] | None = None,
    *,
    now=None,
    context=None,
    read_sets: dict[str, set[str]] | None = None,
) -> dict[str, ActionDecision]:
    snap = EvidenceSnapshot(bindings)
    graph = propagation_graph(task)
    context = resolve_context(task, contract, batch, now, context)
    result = {}
    for action in actions if actions is not None else action_templates(task, contract):
        snap.reads.clear()
        result[action.action_id] = _gate(
            action,
            task,
            contract,
            TrackedStates(states, snap.reads) if read_sets is not None else states,
            snap,
            batch,
            graph,
            context,
        )
        if read_sets is not None:
            read_sets[action.action_id] = set(snap.reads)
    return result
