from datetime import timedelta

import pytest

from upf_agent.v8.adapters import collect
from upf_agent.v8.compiler import compile_contract, compile_query_plan
from upf_agent.v8.models import EvaluationContext, EvidenceBinding
from upf_agent.v8.plan_capabilities import compile_capabilities
from upf_agent.v8.plan_cases import SPECS, prepare_case
from upf_agent.v8.plan_generator import generation_schema


def inputs(root, index=0):
    task, _ = prepare_case(root, index)
    contract = compile_contract(task)
    bindings = []
    for step in compile_query_plan(task, contract).steps:
        packet = collect(root, task, contract, step, step.step_id)
        bindings.extend(EvidenceBinding.model_validate(b) for b in packet["bindings"])
    context = EvaluationContext.from_task(task, contract, "capability-test")
    return task, context, bindings


@pytest.mark.parametrize("index", range(24))
def test_visible_sources_detect_all_missing_and_conflicting_development_conditions(tmp_path, index):
    root = tmp_path / "case"
    task, context, bindings = inputs(root, index)
    capabilities = compile_capabilities(task, context, bindings, root)
    mutation = SPECS[index][1]
    clear = mutation in {"normal", "disabled", "down", "small_mtu"}
    assert bool(capabilities["unresolved"]) is not clear
    assert not capabilities["uses_gold_or_contract_states"]
    assert not capabilities["post_generation_repair"]


@pytest.mark.parametrize("index", [4, 5, 13, 14, 15, 16])
def test_missing_or_unknown_knowledge_cannot_generate_configuration(tmp_path, index):
    root = tmp_path / "case"
    task, context, bindings = inputs(root, index)
    caps = compile_capabilities(task, context, bindings, root)
    schema = generation_schema(
        bindings, {"target_objects": task.objects}, capabilities=caps, task=task
    )
    actions = [
        b["properties"]["action_type"]["const"]
        for b in schema["$defs"]["PlanStep"]["anyOf"]
        if b["properties"]["target"]["const"] == "UPF-A"
    ]
    assert "CHECK_CONFIGURATION" not in actions
    assert "REQUEST_EVIDENCE" in actions and "HUMAN_REVIEW" in actions


def test_permission_denied_configuration_stays_blocked_without_rejecting_all_checks(tmp_path):
    root = tmp_path / "case"
    task, context, bindings = inputs(root, 19)
    caps = compile_capabilities(task, context, bindings, root)
    schema = generation_schema(
        bindings, {"target_objects": task.objects}, capabilities=caps, task=task
    )
    actions = [
        b["properties"]["action_type"]["const"] for b in schema["$defs"]["PlanStep"]["anyOf"]
    ]
    assert "CHECK_CONFIGURATION" not in actions
    assert "CHECK_INTERFACE" in actions
    assert any("permission_denied" in gap for gap in caps["unresolved"])


def test_action_reference_scope_and_identity_dependencies_are_constrained_before_generation(
    tmp_path,
):
    root = tmp_path / "case"
    task, context, bindings = inputs(root)
    caps = compile_capabilities(task, context, bindings, root)
    schema = generation_schema(
        bindings, {"target_objects": task.objects}, capabilities=caps, task=task
    )
    by_id = {b.evidence_id: b for b in bindings}
    for branch in schema["$defs"]["PlanStep"]["anyOf"]:
        props = branch["properties"]
        assert list(props)[:4] == ["action_type", "target", "step_id", "text"]
        action, target = props["action_type"]["const"], props["target"]["const"]
        expected = [] if action == "CHECK_IDENTITY" else [caps["identity_steps"][target]]
        assert props["depends_on"] == {"const": expected}
        refs = props["evidence"]
        if refs["maxItems"]:
            choices = schema["$defs"][refs["items"]["$ref"].split("/")[-1]]["anyOf"]
            for choice in choices:
                binding = by_id[choice["const"]["evidence_id"]]
                if action == "VERIFY_PEER_MAPPING":
                    assert binding.category == "P"
                else:
                    assert binding.object_id == target
                if action == "CHECK_CONFIGURATION":
                    assert binding.category == "K" and refs["minItems"] == 1


@pytest.mark.parametrize("index", [0, 4, 22])
def test_supplementation_is_limited_to_actual_visible_gaps(tmp_path, index):
    root = tmp_path / "case"
    task, context, bindings = inputs(root, index)
    caps = compile_capabilities(task, context, bindings, root)
    schema = generation_schema(
        bindings, {"target_objects": task.objects}, capabilities=caps, task=task
    )
    requests = [
        b
        for b in schema["$defs"]["PlanStep"]["anyOf"]
        if b["properties"]["action_type"]["const"] == "REQUEST_EVIDENCE"
    ]
    assert bool(requests) == bool(caps["unresolved"])
    for branch in requests:
        missing = branch["properties"]["parameters"]["properties"]["missing"]
        assert missing["items"]["enum"] == caps["unresolved"]
        assert missing["minItems"] == 1


@pytest.mark.parametrize("index", range(24))
def test_task_workflow_has_unique_anchors_and_one_actual_gap_outlet(tmp_path, index):
    root = tmp_path / "case"
    task, context, bindings = inputs(root, index)
    caps = compile_capabilities(task, context, bindings, root)
    schema = generation_schema(
        bindings, {"target_objects": task.objects}, capabilities=caps, task=task
    )
    steps = schema["properties"]["steps"]
    slots = steps["prefixItems"]
    assert steps["items"] is False
    assert steps["minItems"] == steps["maxItems"] == len(slots) <= 8
    assert [slot["properties"]["target"]["const"] for slot in slots[:2]] == task.objects
    assert all(slot["properties"]["action_type"]["const"] == "CHECK_IDENTITY" for slot in slots[:2])
    identifiers = []
    for slot in slots:
        for branch in slot.get("anyOf", [slot]):
            props = branch["properties"]
            identifiers.append(props["step_id"]["const"])
            if props["action_type"]["const"] == "VERIFY_PEER_MAPPING":
                assert props["depends_on"]["const"] == list(caps["identity_steps"].values())
    assert len(identifiers) == len(set(identifiers))
    if caps["unresolved"]:
        outlets = slots[-1]["anyOf"]
        assert {b["properties"]["action_type"]["const"] for b in outlets} == {
            "REQUEST_EVIDENCE",
            "HUMAN_REVIEW",
        }
        request = next(
            b for b in outlets if b["properties"]["action_type"]["const"] == "REQUEST_EVIDENCE"
        )
        assert request["properties"]["parameters"]["properties"]["missing"] == {
            "const": caps["unresolved"]
        }


@pytest.mark.parametrize("mutation", ["source", "receipt", "task", "contract", "future", "deleted"])
def test_generation_compiler_does_not_trust_forged_binding_metadata(tmp_path, mutation):
    root = tmp_path / "case"
    task, context, bindings = inputs(root)
    b = next(b for b in bindings if b.category == ("K" if mutation == "receipt" else "R"))
    if mutation == "source":
        b.fields["state"] = "forged"
    elif mutation == "receipt":
        (root / "data/v8/knowledge_applicability.json").write_text(
            '{"version":"revoked","annotations":[]}'
        )
    elif mutation == "task":
        b.task_id = "0" * 32
    elif mutation == "contract":
        b.contract_id = "other-contract"
    elif mutation == "future":
        b.observed.end = context.as_of + timedelta(seconds=1)
    elif mutation == "deleted":
        b.deleted = True
    caps = compile_capabilities(task, context, bindings, root)
    assert b.evidence_id not in caps["admissible_ids"]
    assert b.evidence_id in caps["rejected"]
