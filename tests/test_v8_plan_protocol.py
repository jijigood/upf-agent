import hashlib
import json

import pytest
from test_v8_rules import prepared

from upf_agent.v8.models import EvaluationContext, digest
from upf_agent.v8.plan_ir import ActionPattern, CaseGold, GoldBranch, PlanIR, PlanStep
from upf_agent.v8.plan_metrics import score_case, summarize_cases
from upf_agent.v8.plan_translator import translate_plan
from upf_agent.v8.plan_validator import validate_plan


def inputs(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    context = EvaluationContext.from_task(task, contract, "plan-batch")
    plan = PlanIR(
        context_fingerprint=context.fingerprint,
        scope_fingerprint=digest(task.permission),
        batch_id=context.batch_id,
        evaluation_as_of=context.as_of,
        target_objects=task.objects,
        steps=[
            PlanStep(
                step_id="identity",
                text="检查UPF-A身份",
                action_type="CHECK_IDENTITY",
                target="UPF-A",
            ),
            PlanStep(
                step_id="interface",
                text="核对N4接口状态",
                action_type="CHECK_INTERFACE",
                target="UPF-A",
                parameters={"interface": "N4"},
                depends_on=["identity"],
            ),
        ],
        stop_conditions=["未知时停止补证"],
    )
    return task, contract, bindings, context, plan


def validate(tmp_path, items):
    task, contract, bindings, context, plan = items
    return validate_plan(plan, bindings, task, contract, context, tmp_path)


def test_valid_read_check_does_not_require_its_future_result(tmp_path):
    items = inputs(tmp_path)
    report = validate(tmp_path, items)
    assert report.raw_plan_validity and report.retained_steps == 2
    assert not report.execute_authorized


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("cycle", "dependency_cycle"),
        ("unknown_dependency", "unknown_dependency"),
        ("duplicate", "duplicate_step_id"),
        ("batch", "context_mismatch"),
        ("context", "context_mismatch"),
        ("scope", "target_scope_mismatch"),
        ("target", "object_permission_denied"),
        ("execution", "execution_claim_or_write"),
        ("command", "invalid_parameters"),
        ("action", "unknown_action"),
        ("interface", "wrong_interface_parameter"),
        ("permission", "permission_or_effect_mismatch"),
        ("goal", "unverified_goal_claim"),
        ("missing_knowledge", "required_knowledge_missing"),
        ("field", "field_permission_denied"),
    ],
)
def test_hard_violations_are_rejected_without_repair(tmp_path, mutation, code):
    items = list(inputs(tmp_path))
    task, _, _, _, plan = items
    if mutation == "cycle":
        plan.steps[0].depends_on = ["interface"]
    if mutation == "unknown_dependency":
        plan.steps[1].depends_on = ["missing"]
    if mutation == "duplicate":
        plan.steps[1].step_id = "identity"
    if mutation == "batch":
        plan.batch_id = "stale"
    if mutation == "context":
        plan.context_fingerprint = "fake"
    if mutation == "scope":
        plan.target_objects = ["UPF-A"]
    if mutation == "target":
        plan.steps[1].target = "outside"
    if mutation == "execution":
        plan.steps[1].claimed_executed = True
    if mutation == "command":
        plan.steps[1].parameters["command"] = "sudo change"
    if mutation == "action":
        plan.steps[1].action_type = "EXECUTE_CHANGE"
    if mutation == "interface":
        plan.steps[1].parameters["interface"] = "N3"
    if mutation == "permission":
        plan.steps[1].permission_kind = "suggest"
    if mutation == "goal":
        plan.goal_achieved = True
        plan.steps[1].expected_observations = ["已修复"]
    if mutation == "missing_knowledge":
        plan.steps[1].action_type = "CHECK_CONFIGURATION"
        plan.steps[1].parameters = {"interface": "N4", "field": "n4_enabled"}
    if mutation == "field":
        task.permission.readable_fields.remove("state")
        items[3] = EvaluationContext.from_task(task, items[1], "plan-batch")
        plan.context_fingerprint = items[3].fingerprint
        plan.scope_fingerprint = digest(task.permission)
    report = validate(tmp_path, items)
    assert not report.raw_plan_validity
    assert code in [i.code for i in report.issues]
    assert report.published_plan is None and report.rewritten_steps == 0


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("id", "unknown_evidence"),
        ("revision", "evidence_reference_mismatch"),
        ("hash", "evidence_reference_mismatch"),
        ("locator", "evidence_reference_mismatch"),
        ("source", "unverified_source"),
        ("version", "evidence_version_mismatch"),
        ("future", "unknown_or_future_observation"),
        ("target", "evidence_target_mismatch"),
    ],
)
def test_evidence_is_checked_against_actual_snapshot_and_source(tmp_path, mutation, code):
    items = inputs(tmp_path)
    task, _, bindings, _, plan = items
    b = next(b for b in bindings if b.category == "R" and b.object_id == "UPF-A")
    ref = {
        "evidence_id": b.evidence_id,
        "revision": b.revision,
        "source_sha256": b.source.sha256,
        "locator": b.source.locator,
    }
    if mutation == "id":
        ref["evidence_id"] = "invented"
    if mutation == "revision":
        ref["revision"] += 1
    if mutation == "hash":
        ref["source_sha256"] = "0" * 64
    if mutation == "locator":
        ref["locator"] = "/invented"
    if mutation == "source":
        (tmp_path / b.source.relative_path).write_text("tampered")
    if mutation == "version":
        b.version = "4.0"
    if mutation == "future":
        from datetime import timedelta

        b.observed = b.observed.model_copy(update={"end": task.as_of + timedelta(seconds=1)})
    if mutation == "target":
        plan.steps[1].target = "SMF-B"
    from upf_agent.v8.plan_ir import EvidenceReference

    plan.steps[1].evidence = [EvidenceReference(**ref)]
    report = validate(tmp_path, items)
    assert code in [i.code for i in report.issues]


def test_translator_preserves_original_spans_and_unicode(tmp_path):
    *_, plan = inputs(tmp_path)
    raw = plan.model_dump_json(indent=2)
    result = translate_plan(raw)
    assert result.plan == plan and not result.errors
    assert result.raw_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    for span, step in zip(result.spans, plan.steps, strict=True):
        assert json.loads(raw[span.start : span.end]) == step.model_dump(mode="json")
        assert json.loads(raw[span.text_start : span.text_end]) == step.text


@pytest.mark.parametrize(
    "raw", ["not JSON", "{}", '{"x":1,"x":2}', '{"x":NaN}', "```json\n{}\n```"]
)
def test_translator_keeps_failures_without_silent_repair(raw):
    result = translate_plan(raw)
    assert result.plan is None and result.errors and result.cost_model_calls == 0


def test_unknown_fields_are_not_discarded(tmp_path):
    *_, plan = inputs(tmp_path)
    body = plan.model_dump(mode="json")
    body["model_source_verified"] = True
    result = translate_plan(json.dumps(body))
    assert result.plan is None and result.unknown_fields


def gold_for(task, plan):
    return CaseGold(
        case_id="case",
        source_group="group",
        structure_family="family",
        source_level="D1",
        synthetic=True,
        solvability="solvable",
        objects=task.objects,
        version=task.version,
        as_of=task.as_of,
        permission_fingerprint=digest(task.permission),
        visible_evidence=[],
        source_hashes={},
        any_of=[
            GoldBranch(
                branch_id="main",
                required=[
                    ActionPattern(pattern_id="id", action_type="CHECK_IDENTITY", target="UPF-A"),
                    ActionPattern(
                        pattern_id="if",
                        action_type="CHECK_INTERFACE",
                        target="UPF-A",
                        parameters={"interface": "N4"},
                        depends_on=["id"],
                    ),
                ],
            )
        ],
        stop_conditions=["unknown stop"],
    )


def test_unreviewed_reference_does_not_become_vpr(tmp_path):
    items = inputs(tmp_path)
    task, *_, plan = items
    metrics = score_case(plan, validate(tmp_path, items), gold_for(task, plan))
    assert metrics["reference_pass"] and metrics["vpr"] is None
    summary = summarize_cases([metrics])
    assert summary["vpr_denominator"] == 0 and summary["vpr"] is None


def test_gold_multi_solution_not_union_and_dependency_reachable(tmp_path):
    items = inputs(tmp_path)
    task, *_, plan = items
    gold = gold_for(task, plan)
    gold.any_of.append(
        GoldBranch(
            branch_id="alternative",
            required=[
                ActionPattern(pattern_id="review", action_type="HUMAN_REVIEW", target="SMF-B")
            ],
        )
    )
    assert score_case(plan, validate(tmp_path, items), gold)["reference_pass"]
    plan.steps[1].depends_on = []
    assert not score_case(plan, validate(tmp_path, items), gold)["reference_pass"]


def test_unknown_precondition_has_finite_stop_outlet(tmp_path):
    from upf_agent.v8.plan_ir import EvidenceReference, Precondition

    items = inputs(tmp_path)
    _, _, bindings, _, plan = items
    b = next(b for b in bindings if b.object_id == "UPF-A" and b.category == "R")
    plan.steps[1].evidence = [
        EvidenceReference(
            evidence_id=b.evidence_id,
            revision=b.revision,
            source_sha256=b.source.sha256,
            locator=b.source.locator,
        )
    ]
    plan.steps[1].preconditions = [
        Precondition(
            predicate="field_equals",
            evidence_id=b.evidence_id,
            field="unknown_field",
            expected=True,
        )
    ]
    report = validate(tmp_path, items)
    assert report.raw_plan_validity and report.preconditions[0]["value"] == "unknown"
    plan.steps[1].preconditions[0].field = "n4_enabled"
    plan.steps[1].preconditions[0].expected = False
    assert "false_precondition" in [i.code for i in validate(tmp_path, items).issues]


def test_parse_failure_is_unscorable_not_confirmed_hard_violation(tmp_path):
    task, contract, bindings, context, plan = inputs(tmp_path)
    report = validate_plan(None, bindings, task, contract, context, tmp_path)
    metrics = score_case(None, report, gold_for(task, plan))
    assert metrics["unscorable"] and not metrics["hard_constraint_violation"]
    assert not metrics["reference_pass"]


def test_duplicate_target_cannot_hide_in_scope_set(tmp_path):
    items = inputs(tmp_path)
    items[-1].target_objects.append("UPF-A")
    assert "duplicate_target_object" in [i.code for i in validate(tmp_path, items).issues]


def test_source_locator_and_field_values_are_independently_verified(tmp_path):
    from upf_agent.v8.plan_ir import EvidenceReference

    items = inputs(tmp_path)
    b = next(b for b in items[2] if b.category == "R" and b.object_id == "UPF-A")
    b.fields["n4_enabled"] = False
    items[-1].steps[1].evidence = [
        EvidenceReference(
            evidence_id=b.evidence_id,
            revision=b.revision,
            source_sha256=b.source.sha256,
            locator=b.source.locator,
        )
    ]
    assert "unverified_source" in [i.code for i in validate(tmp_path, items).issues]
    b.fields["n4_enabled"] = True
    b.source.locator = "/observations/999"
    items[-1].steps[1].evidence[0].locator = b.source.locator
    assert "unverified_source" in [i.code for i in validate(tmp_path, items).issues]


def test_peer_mapping_uses_two_authorized_objects_without_invented_pair_target(tmp_path):
    from upf_agent.v8.plan_ir import EvidenceReference

    items = inputs(tmp_path)
    plan = items[-1]
    pair = PlanStep(
        step_id="pair",
        text="核对双侧映射",
        action_type="VERIFY_PEER_MAPPING",
        target="UPF-A",
        parameters={"peer_object_id": "SMF-B"},
        depends_on=["identity"],
    )
    pair.evidence = [
        EvidenceReference(
            evidence_id=b.evidence_id,
            revision=b.revision,
            source_sha256=b.source.sha256,
            locator=b.source.locator,
        )
        for b in items[2]
        if b.category == "P"
    ]
    plan.steps.append(pair)
    assert validate(tmp_path, items).raw_plan_validity
    pair.parameters["peer_object_id"] = "outside"
    assert "peer_permission_denied" in [i.code for i in validate(tmp_path, items).issues]


def test_single_reviewer_and_mutated_reference_cannot_become_reviewed_gold(tmp_path):
    from upf_agent.v8.plan_review import GoldReview, apply_reviews, calibration_report

    task, *_, plan = inputs(tmp_path)
    gold = gold_for(task, plan)
    a = GoldReview(
        case_id=gold.case_id,
        reviewer="test-reviewer-a",
        reference_sha256=digest(gold),
        decision="approve",
        review_version="test-1",
        review_source="unit-test-only",
    )
    b = a.model_copy(update={"reviewer": "test-reviewer-b"})
    with pytest.raises(ValueError, match="two independent"):
        apply_reviews(gold, [a])
    reviewed = apply_reviews(gold, [a, b])
    assert reviewed.reviewed and len(reviewed.reviewers) == 2
    b.reference_sha256 = "wrong"
    with pytest.raises(ValueError, match="frozen"):
        apply_reviews(gold, [a, b])
    b.reference_sha256 = digest(gold)
    b.decision = "request_changes"
    with pytest.raises(ValueError, match="adjudication"):
        apply_reviews(gold, [a, b])
    report = calibration_report([a, b])
    assert report["paired_cases"] == 1 and report["agreement_count"] == 0
    assert report["adjudication_cases"] == [gold.case_id]


def test_source_or_fixture_family_cannot_leak_across_splits(tmp_path):
    from upf_agent.v8.plan_review import validate_splits

    task, *_, plan = inputs(tmp_path)
    gold = gold_for(task, plan)
    other = gold.model_copy(
        update={"case_id": "other", "source_group": "another", "split": "heldout"}
    )
    with pytest.raises(ValueError, match="structure_family"):
        validate_splits([gold, other])


def test_json_boolean_and_integer_are_not_the_same_source_value(tmp_path):
    from upf_agent.v8.plan_source_verifier import verify_plan_source

    task, _, bindings, _, _ = inputs(tmp_path)
    b = next(b for b in bindings if b.category == "R" and b.object_id == "UPF-A")
    b.fields["n4_enabled"] = 1
    assert not verify_plan_source(tmp_path, b, task)


@pytest.mark.parametrize("tampered", [False, True])
def test_non_json_manual_requires_pinned_chunk_archive_hash(tmp_path, monkeypatch, tampered):
    from types import SimpleNamespace

    from upf_agent.v8.plan_source_verifier import verify_plan_source

    task, _, bindings, _, _ = inputs(tmp_path)
    b = next(b for b in bindings if b.category == "K")
    path = tmp_path / "data/raw/pinned.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.fields["instruction"])
    b.source.relative_path = "data/raw/pinned.txt"
    b.source.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    b.source.locator = "page=1"
    b.native_id = "chunk"
    archive = tmp_path / "archive"
    archive.mkdir()
    chunks = archive / "chunks.jsonl"
    chunks.write_text("pinned archive\n")
    manifest = {"chunks_sha256": hashlib.sha256(chunks.read_bytes()).hexdigest()}
    if tampered:
        chunks.write_text("changed archive\n")
    monkeypatch.setattr("upf_agent.p3_index.load_active_index", lambda root: (archive, manifest))
    monkeypatch.setattr(
        "upf_agent.p3_index.load_chunks",
        lambda root: {
            "chunk": SimpleNamespace(
                chunk_id="chunk",
                source_id=b.source.source_id,
                relative_path="pinned.txt",
                product=b.product,
                version=b.version,
                source_sha256=b.source.sha256,
                text=b.fields["instruction"],
                locators=["page=1"],
            )
        },
    )
    assert verify_plan_source(tmp_path, b, task) is (not tampered)
