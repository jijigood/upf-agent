"""Independent deterministic K/R/P/A rules; no dependency index is consulted here."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from upf_agent.v8.compiler import slot
from upf_agent.v8.evaluation_context import bind_snapshot, certify, resolve_context
from upf_agent.v8.models import (
    Code,
    EvidenceBinding,
    EvidenceContract,
    Gap,
    OrchestrationTask,
    Requirement,
    RequirementState,
    canonical,
    digest,
)


class TrackedStates(dict):
    def __init__(self, states, reads):
        super().__init__(states)
        self.reads = reads

    def __getitem__(self, key):
        self.reads.add(f"state:{key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.reads.add(f"state:{key}")
        return super().get(key, default)

    def __contains__(self, key):
        self.reads.add(f"state:{key}")
        return super().__contains__(key)


class EvidenceSnapshot:
    def __init__(self, bindings: list[EvidenceBinding]):
        self.bindings = bindings
        self.by_object: dict[tuple[str, str, str], list[EvidenceBinding]] = defaultdict(list)
        self.values: dict[str, list[Any]] = defaultdict(list)
        self.reads: set[str] = set()
        self._fingerprints = None
        for b in bindings:
            if not b.deleted:
                self.by_object[b.object_id or "?", b.category, b.requirement_id].append(b)

    @property
    def fingerprints(self):
        if self._fingerprints is None:
            for b in self.bindings:
                if b.deleted:
                    continue
                obj = b.object_id or "?"
                record = b.model_dump(mode="json", exclude={"fields"})
                self.values[slot(obj, b.category, f"__record__:{b.requirement_id}")].append(record)
                for field, value in b.fields.items():
                    self.values[slot(obj, b.category, field)].append(
                        [b.evidence_id, b.revision, value]
                    )
            self._fingerprints = {
                k: digest(sorted(v, key=canonical)) for k, v in self.values.items()
            }
        return self._fingerprints

    def get(self, obj: str, kind: str, field: str) -> list[Any]:
        self.reads.add(slot(obj, kind, field))
        return self.values.get(slot(obj, kind, field), [])

    def records(self, obj: str, kind: str, requirement_id: str) -> list[EvidenceBinding]:
        self.get(obj, kind, f"__record__:{requirement_id}")
        return self.by_object.get((obj, kind, requirement_id), [])


def _state(
    req: Requirement,
    code: Code,
    reasons: list[str],
    basis: dict,
    batch: str,
    support: list[str] | None = None,
    oppose: list[str] | None = None,
) -> RequirementState:
    semantic = {
        "code": code.value,
        "reasons": reasons,
        "basis": basis,
        "supporting": sorted(support or []),
        "opposing": sorted(oppose or []),
    }
    return RequirementState(
        requirement_id=req.requirement_id,
        code=code,
        reasons=reasons,
        supporting=semantic["supporting"],
        opposing=semantic["opposing"],
        basis=basis,
        state_version=digest(semantic),
        valid=code == Code.SATISFIED,
        batch_id=batch,
    )


def _eligibility(
    b: EvidenceBinding,
    req: Requirement,
    task: OrchestrationTask,
    contract: EvidenceContract,
    now: datetime,
    authorized: bool,
) -> tuple[Code, str] | None:
    if b.task_id != task.run_id or b.contract_id != contract.contract_id:
        return Code.UNKNOWN, "cross_task_or_contract"
    if b.requirement_id != req.requirement_id:
        return Code.UNKNOWN, "wrong_requirement"
    if b.object_id not in req.objects:
        return Code.UNKNOWN, "unknown_identity"
    if not b.source_verified or not b.corroborated:
        return Code.UNKNOWN, "unverified_source_or_relation"
    if b.synthetic and not task.synthetic:
        return Code.NOT_APPLICABLE, "synthetic_in_real_task"
    if not b.readable or not authorized:
        return Code.DENIED, "permission_denied"
    if req.kind == "K" and (not task.product or not b.product):
        return Code.UNKNOWN, "unknown_product"
    if req.kind == "K" and task.product != b.product:
        return Code.NOT_APPLICABLE, "wrong_product"
    if req.kind == "K" and req.purpose:
        if not b.purposes:
            return Code.UNKNOWN, "unknown_knowledge_purpose"
        if req.purpose not in b.purposes:
            return Code.NOT_APPLICABLE, "wrong_knowledge_purpose"
    if req.kind == "R" and "interface" in req.fields:
        expected = {"upf_n4_fault": "N4", "upf_n3_fault": "N3"}.get(task.template)
        if expected and b.fields.get("interface") not in (None, expected):
            return Code.NOT_APPLICABLE, "wrong_interface"
    if req.version_required:
        if not task.version or not b.version or b.version == "unknown":
            return Code.UNKNOWN, "unknown_version"
        if b.version != task.version:
            return Code.NOT_APPLICABLE, "wrong_version"
    if b.valid and not b.valid.start <= now < b.valid.end:
        return Code.STALE, "outside_valid_window"
    if req.kind in {"R", "P"}:
        if b.observed is None:
            return Code.UNKNOWN, "unknown_observation_window"
        if b.observed.start > now:
            return Code.UNKNOWN, "future_observation"
        if (
            req.freshness_seconds is not None
            and (now - b.observed.end).total_seconds() > req.freshness_seconds
        ):
            return Code.STALE, "freshness_expired"
        if task.time_window and max(b.observed.start, task.time_window.start) >= min(
            b.observed.end, task.time_window.end
        ):
            return Code.NOT_APPLICABLE, "outside_task_window"
    return None


def select(
    req: Requirement,
    obj: str,
    task: OrchestrationTask,
    contract: EvidenceContract,
    snap: EvidenceSnapshot,
    now: datetime,
) -> tuple[list[EvidenceBinding], list[dict]]:
    if not hasattr(snap, "permission_inputs"):
        snap.permission_inputs = (
            set(task.objects),
            set(task.permission.objects),
            set(task.permission.domains),
            set(task.permission.readable_fields),
            {p.object_id: p.domain for p in task.path.positions},
        )
    declared, readable, domains, fields, domain_map = snap.permission_inputs
    authorized = (
        obj in declared
        and obj in readable
        and (not domains or domain_map.get(obj) in domains)
        and set(req.fields) <= fields
    )
    records = snap.records(obj, req.kind, req.requirement_id)
    for f in req.fields:
        snap.get(obj, req.kind, f)
    eligible, rejected = [], []
    for b in records:
        if b.requirement_id != req.requirement_id:
            continue
        rejection = _eligibility(b, req, task, contract, now, authorized)
        if rejection:
            rejected.append(
                {"id": b.evidence_id, "code": rejection[0].value, "reason": rejection[1]}
            )
        else:
            eligible.append(b)
    if eligible and req.source_priority:
        priorities = {v: i for i, v in enumerate(req.source_priority)}
        rank = min(priorities.get(b.source.source_id, len(priorities)) for b in eligible)
        selected = [
            b for b in eligible if priorities.get(b.source.source_id, len(priorities)) == rank
        ]
        rejected.extend(
            {
                "id": b.evidence_id,
                "code": "not_applicable",
                "reason": "lower_declared_source_priority",
            }
            for b in eligible
            if b not in selected
        )
        eligible = selected
    return sorted(eligible, key=lambda b: b.evidence_id), sorted(rejected, key=canonical)


def record_basis(binding: EvidenceBinding, requirement: Requirement) -> dict:
    record = binding.model_dump(mode="json")
    record["fields"] = {
        field: binding.fields[field] for field in requirement.fields if field in binding.fields
    }
    return record


def _evaluate_requirement(
    req: Requirement,
    task: OrchestrationTask,
    contract: EvidenceContract,
    snap: EvidenceSnapshot,
    states: dict[str, RequirementState],
    batch: str,
    now: datetime,
) -> RequirementState:
    if not hasattr(snap, "context"):
        snap.context = {
            "contract": contract.contract_id,
            "contract_semantics": digest(contract),
            "permission": digest(task.permission),
            "path": digest(task.path),
        }
    context = snap.context
    if req.kind == "A":
        if not hasattr(snap, "permission_inputs"):
            snap.permission_inputs = (
                set(task.objects),
                set(task.permission.objects),
                set(task.permission.domains),
                set(task.permission.readable_fields),
                {p.object_id: p.domain for p in task.path.positions},
            )
        declared, readable, domains, fields, domain_map = snap.permission_inputs
        ok = all(
            o in declared
            and o in readable
            and (not domains or domain_map.get(o) in domains)
            and set(req.fields) <= fields
            for o in req.objects
        )
        return _state(
            req,
            Code.SATISFIED if ok else Code.DENIED,
            ["read_permission_granted" if ok else "permission_denied"],
            context,
            batch,
        )
    if req.kind == "C":
        inputs = {rid: states[rid].semantic() if rid in states else None for rid in req.upstream}
        ok = all(
            v is not None and v["code"] == "satisfied" and not v["pending"] for v in inputs.values()
        )
        return _state(
            req,
            Code.SATISFIED if ok else Code.MISSING,
            ["complete" if ok else "upstream_incomplete"],
            {**context, "inputs": inputs},
            batch,
        )
    sides = {o: select(req, o, task, contract, snap, now) for o in req.objects}
    basis = {
        **context,
        "records": {
            obj: {"eligible": [record_basis(b, req) for b in records], "excluded": rejected}
            for obj, (records, rejected) in sides.items()
        },
    }
    supporting = [b.evidence_id for records, _ in sides.values() for b in records]
    opposing = [v["id"] for _, rejected in sides.values() for v in rejected]
    for obj, (records, rejected) in sides.items():
        if not records:
            codes = [Code(r["code"]) for r in rejected]
            code = next(
                (
                    c
                    for c in [
                        Code.DENIED,
                        Code.CONFLICT,
                        Code.STALE,
                        Code.NOT_APPLICABLE,
                        Code.UNKNOWN,
                    ]
                    if c in codes
                ),
                Code.MISSING,
            )
            return _state(
                req,
                code,
                [f"missing_side:{obj}"] + [r["reason"] for r in rejected],
                basis,
                batch,
                supporting,
                opposing,
            )
        missing = sorted(
            {
                f
                for b in records
                for f in req.fields
                if f not in b.fields or b.fields[f] is None or b.fields[f] == ""
            }
        )
        if missing:
            return _state(
                req,
                Code.MISSING,
                [f"missing_fields:{obj}:{','.join(missing)}"],
                basis,
                batch,
                supporting,
                opposing,
            )
        values = {canonical({f: b.fields[f] for f in req.fields}) for b in records}
        if len(values) != 1:
            return _state(
                req,
                Code.CONFLICT,
                [f"unresolved_values:{obj}"],
                basis,
                batch,
                supporting,
                supporting + opposing,
            )
        if len({canonical(b.observed) for b in records}) > 1 and req.window_rule == "equal":
            return _state(
                req,
                Code.UNKNOWN,
                ["window_normalization_unspecified"],
                basis,
                batch,
                supporting,
                opposing,
            )
    if req.kind == "P":
        if len(req.objects) != 2:
            return _state(req, Code.UNKNOWN, ["pair_requires_two_objects"], basis, batch)
        lefts, rights = sides[req.objects[0]][0], sides[req.objects[1]][0]
        if not req.direction_map:
            return _state(req, Code.UNKNOWN, ["direction_rule_missing"], basis, batch)
        # Validate every retained candidate; one convenient pair cannot hide a conflict.
        for left in lefts:
            for right in rights:
                lf, rf = left.fields, right.fields
                if any(type(f[key]) is not str for f in (lf, rf) for key in req.fields):
                    return _state(
                        req,
                        Code.UNKNOWN,
                        ["pair_identity_field_type_unknown"],
                        basis,
                        batch,
                        supporting,
                        opposing,
                    )
                if task.path.version is None or any(
                    f["mapping_version"] != task.path.version for f in (lf, rf)
                ):
                    return _state(
                        req,
                        Code.NOT_APPLICABLE,
                        ["mapping_version_outside_contract"],
                        basis,
                        batch,
                        supporting,
                        opposing,
                    )
                if (
                    lf["service_id"] != task.path.service_id
                    or rf["service_id"] != task.path.service_id
                ):
                    return _state(
                        req,
                        Code.CONFLICT,
                        ["service_mismatch"],
                        basis,
                        batch,
                        supporting,
                        supporting,
                    )
                if lf["local_endpoint"] != rf["peer_endpoint"] or (
                    lf["peer_endpoint"] != rf["local_endpoint"]
                ):
                    return _state(
                        req,
                        Code.CONFLICT,
                        ["endpoint_mismatch"],
                        basis,
                        batch,
                        supporting,
                        supporting,
                    )
                directions = {
                    req.direction_map.get(lf["direction"]),
                    req.direction_map.get(rf["direction"]),
                }
                if None in directions:
                    return _state(req, Code.UNKNOWN, ["unknown_direction"], basis, batch)
                if directions != {"forward", "reverse"}:
                    return _state(req, Code.CONFLICT, ["direction_mismatch"], basis, batch)
                if not left.config_version or not right.config_version:
                    return _state(req, Code.UNKNOWN, ["unknown_config_version"], basis, batch)
                if left.config_version != right.config_version or (
                    lf["mapping_version"] != rf["mapping_version"]
                ):
                    return _state(req, Code.CONFLICT, ["pair_version_mismatch"], basis, batch)
                lw, rw = left.observed, right.observed
                if lw is None or rw is None:
                    return _state(req, Code.UNKNOWN, ["unknown_observation_window"], basis, batch)
                if max(lw.start, rw.start) >= min(lw.end, rw.end):
                    return _state(
                        req, Code.CONFLICT, ["half_open_windows_do_not_overlap"], basis, batch
                    )
                if req.window_rule == "equal" and lw != rw:
                    return _state(
                        req, Code.UNKNOWN, ["window_normalization_unspecified"], basis, batch
                    )
    return _state(req, Code.SATISFIED, ["contract_satisfied"], basis, batch, supporting, opposing)


def evaluate_requirement(req, task, contract, snap, states, batch, now, context=None):
    if context is None:
        context = resolve_context(task, contract, batch, now)
    elif context.batch_id != batch or context.as_of != now:
        raise ValueError("requirement evaluation context batch/time mismatch")
    bind_snapshot(snap, context)
    return certify(
        _evaluate_requirement(req, task, contract, snap, states, batch, now),
        context,
        computed=True,
    )


def full_evaluate(
    task: OrchestrationTask,
    contract: EvidenceContract,
    bindings: list[EvidenceBinding],
    batch: str,
    now: datetime | None = None,
    *,
    context=None,
    read_sets: dict[str, set[str]] | None = None,
) -> dict[str, RequirementState]:
    """Reference traversal of contract declarations, independent of the dependency index."""
    snap = EvidenceSnapshot(bindings)
    context = resolve_context(task, contract, batch, now, context)
    states: dict[str, RequirementState] = {}

    def evaluate(req):
        snap.reads.clear()
        result = evaluate_requirement(
            req,
            task,
            contract,
            snap,
            TrackedStates(states, snap.reads) if read_sets is not None else states,
            batch,
            context.as_of,
            context,
        )
        if read_sets is not None:
            read_sets[f"state:{req.requirement_id}"] = set(snap.reads)
        return result

    for req in contract.requirements:
        if req.kind != "C":
            states[req.requirement_id] = evaluate(req)
    pending = [r for r in contract.requirements if r.kind == "C"]
    for _ in range(len(pending) + 1):
        ready = [r for r in pending if set(r.upstream) <= states.keys()]
        for req in ready:
            states[req.requirement_id] = evaluate(req)
            pending.remove(req)
        if not pending or not ready:
            break
    for req in pending:
        states[req.requirement_id] = _state(req, Code.UNKNOWN, ["unresolved_cycle"], {}, batch)
        states[req.requirement_id].pending = True
        certify(states[req.requirement_id], context, computed=True)
        if read_sets is not None:
            read_sets[f"state:{req.requirement_id}"] = {f"state:{r}" for r in req.upstream}
    return states


def route_gaps(contract: EvidenceContract, states: dict[str, RequirementState]) -> list[Gap]:
    gaps = []
    for req in contract.requirements:
        state = states[req.requirement_id]
        if state.code == Code.SATISFIED:
            continue
        route = "human"
        if req.kind == "K" and state.code != Code.DENIED:
            route = "knowledge_search"
        elif req.kind == "R" and state.code != Code.DENIED:
            route = "runtime_refresh"
        elif req.kind == "P" and state.code in {Code.MISSING, Code.STALE}:
            route = "missing_side"
        elif req.kind == "P" and state.code == Code.CONFLICT:
            route = "bilateral_review"
        objects = req.objects
        if route == "missing_side":
            missing = [o for o in objects if any(r == f"missing_side:{o}" for r in state.reasons)]
            objects = missing or objects
        gaps.append(
            Gap(
                requirement_id=req.requirement_id,
                route=route,
                objects=objects,
                reasons=state.reasons,
            )
        )
    return gaps
