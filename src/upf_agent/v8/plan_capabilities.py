"""Generation-time admissibility, derived from visible sources, never reference answers.

This compiler is a shared generation prior, not an evaluator. The offline Validator
rechecks the untouched output and physical sources independently.
"""

from upf_agent.v8.compiler import compile_contract, permitted
from upf_agent.v8.knowledge_applicability import annotate_knowledge
from upf_agent.v8.models import canonical
from upf_agent.v8.plan_source_verifier import verify_plan_source

VERSION = "v8-plan-capabilities-1"


def compile_capabilities(task, context, bindings, root):
    admissible, rejected = [], {}
    contract_id = compile_contract(task).contract_id
    for original in bindings:
        b = original.model_copy(deep=True)
        reasons = []
        if b.deleted or b.task_id != task.run_id or b.contract_id != contract_id:
            reasons.append("deleted_or_cross_task")
        if not verify_plan_source(root, b, task) or not b.corroborated:
            reasons.append("source_unverified")
        if not b.readable or not permitted(task, b.object_id, list(b.fields)):
            reasons.append("permission_denied")
        if b.synthetic and not task.synthetic:
            reasons.append("synthetic_source")
        if not b.version or b.version != task.version:
            reasons.append("version_unknown_or_mismatched")
        if b.valid and not b.valid.start <= context.as_of < b.valid.end:
            reasons.append("outside_validity")
        if b.category in {"R", "P"}:
            if not b.observed or b.observed.end > context.as_of:
                reasons.append("observation_window_unknown_or_future")
            elif (context.as_of - b.observed.end).total_seconds() > 300:
                reasons.append("stale_observation")
        if b.category == "K":
            # Re-read the registered mapping, including revocation; a cached receipt is not trust.
            annotate_knowledge(root, b)
            if b.product != task.product or task.template not in b.purposes:
                reasons.append("knowledge_applicability_unknown")
        if reasons:
            rejected[b.evidence_id] = reasons
        else:
            admissible.append(b)

    fault = task.template in {"upf_n3_fault", "upf_n4_fault"}
    field = "mtu" if task.template == "upf_n3_fault" else "n4_enabled"
    required_kinds = ["K", "R", "P"] if fault else ["R"]
    if task.template == "manual_operation":
        required_kinds = ["K"]
    unresolved = []
    for obj in task.objects:
        for kind in required_kinds:
            rows = [b for b in admissible if b.object_id == obj and b.category == kind]
            if not rows:
                unresolved.append(f"{obj}:{kind}:missing_or_inadmissible")
            if kind == "R":
                fields = ["interface", "state", field] if fault else ["event_type"]
                for name in fields:
                    values = {canonical(b.fields[name]) for b in rows if name in b.fields}
                    if not permitted(task, obj, [name]):
                        unresolved.append(f"{obj}:{name}:permission_denied")
                    elif not values:
                        unresolved.append(f"{obj}:{name}:missing")
                    elif len(values) > 1:
                        unresolved.append(f"{obj}:{name}:conflicting_observations")
    if fault:
        pairs = {
            obj: [b for b in admissible if b.object_id == obj and b.category == "P"]
            for obj in task.objects
        }
        for left, right in zip(task.path.positions, task.path.positions[1:], strict=False):
            for a in pairs[left.object_id]:
                for b in pairs[right.object_id]:
                    af, bf = a.fields, b.fields
                    checks = {
                        "service_id": af.get("service_id")
                        == bf.get("service_id")
                        == task.path.service_id,
                        "mapping_version": af.get("mapping_version")
                        == bf.get("mapping_version")
                        == task.path.version,
                        "endpoints": af.get("local_endpoint") == bf.get("peer_endpoint")
                        and af.get("peer_endpoint") == bf.get("local_endpoint"),
                        "config_version": bool(a.config_version)
                        and a.config_version == b.config_version,
                        "observed_window": a.observed.start < b.observed.end
                        and b.observed.start < a.observed.end,
                    }
                    for name, ok in checks.items():
                        if not ok:
                            unresolved.append(f"{a.object_id}/{b.object_id}:{name}:mismatch")
                    for side in [a, b]:
                        runtime = [
                            r
                            for r in admissible
                            if r.object_id == side.object_id and r.category == "R"
                        ]
                        if any(r.config_version != side.config_version for r in runtime):
                            unresolved.append(
                                f"{side.object_id}:runtime_pair_config_version:mismatch"
                            )
    return {
        "version": VERSION,
        "admissible_ids": [b.evidence_id for b in admissible],
        "rejected": rejected,
        "unresolved": sorted(set(unresolved)),
        "identity_steps": {obj: f"id{i + 1}" for i, obj in enumerate(task.objects)},
        "interface": "N3" if task.template == "upf_n3_fault" else "N4",
        "configuration_field": field,
        "fault_investigation": fault,
        "manual_investigation": task.template == "manual_operation",
        "post_generation_repair": False,
        "uses_gold_or_contract_states": False,
    }
