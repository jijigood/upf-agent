import hashlib
from copy import deepcopy

import pytest
from test_v8_rules import prepared

from upf_agent.v8.evaluator import full_evaluate
from upf_agent.v8.knowledge_applicability import (
    KnowledgeAnnotation,
    KnowledgeRegistry,
    annotate_knowledge,
)


def mapped(tmp_path):
    task, contract, bindings = prepared(tmp_path)
    b = next(b for b in bindings if b.category == "K")
    entry = KnowledgeAnnotation(
        annotation_id="annotation-1",
        source_sha256=b.source.sha256,
        content_sha256=hashlib.sha256(b.fields["instruction"].encode()).hexdigest(),
        locator=b.source.locator,
        product=b.product,
        version=b.version,
        object_ids=[b.object_id],
        purposes=[task.template],
        basis_kind="expert_review",
        basis_source="review-receipt-1",
        basis_version="1",
        reviewed=True,
        reviewer="reviewer-1",
    )
    return task, contract, bindings, b, entry


def test_reviewed_mapping_satisfies_knowledge(tmp_path):
    task, contract, bindings, b, entry = mapped(tmp_path)
    b.purposes = []
    annotate_knowledge(tmp_path, b, KnowledgeRegistry(version="1", annotations=[entry]))
    assert b.purposes == [task.template]
    assert b.knowledge_applicability["reviewed"]
    assert full_evaluate(task, contract, bindings, "batch")[b.requirement_id].valid


@pytest.mark.parametrize(
    "change",
    [
        "absent",
        "unreviewed",
        "revoked",
        "version",
        "object",
        "locator",
        "text",
        "hash",
        "source_changed",
        "conflict",
    ],
)
def test_unknown_wrong_and_changed_mapping_cannot_close_knowledge(tmp_path, change):
    task, contract, bindings, b, entry = mapped(tmp_path)
    entries = [entry]
    if change == "absent":
        entries = []
    if change == "unreviewed":
        entry.reviewed = False
    if change == "revoked":
        entry.revoked = True
    if change == "version":
        entry.version = "4.0"
    if change == "object":
        entry.object_ids = ["other"]
    if change == "locator":
        entry.locator += "/other"
    if change == "text":
        b.fields["instruction"] += " forged"
    if change == "hash":
        entry.source_sha256 = "0" * 64
    if change == "source_changed":
        (tmp_path / b.source.relative_path).write_text("tampered")
    if change == "conflict":
        other = deepcopy(entry)
        other.annotation_id = "other"
        other.purposes = ["manual_operation"]
        entries.append(other)
    annotate_knowledge(tmp_path, b, KnowledgeRegistry(version="1", annotations=entries))
    assert b.purposes == []
    assert not full_evaluate(task, contract, bindings, "batch")[b.requirement_id].valid


def test_rule_mapping_has_explicit_basis_without_claiming_expert_review(tmp_path):
    task, _, _, b, entry = mapped(tmp_path)
    entry.rule_id = "synthetic-fixture-1"
    entry.basis_kind = "trusted_rule"
    entry.reviewed = False
    annotate_knowledge(tmp_path, b, KnowledgeRegistry(version="1", annotations=[entry]))
    assert b.purposes == [task.template]
    assert not b.knowledge_applicability["reviewed"]


@pytest.mark.parametrize("multiple_locators", [False, True])
def test_normal_manual_adapter_uses_same_registry(tmp_path, monkeypatch, multiple_locators):
    from types import SimpleNamespace

    from upf_agent.agent_contracts import AgentTask, ToolPacket
    from upf_agent.agent_tools import evidence
    from upf_agent.contracts import ResultStatus, ToolResult
    from upf_agent.v8.adapters import collect
    from upf_agent.v8.models import QueryStep

    task, contract, _, b, annotation = mapped(tmp_path)
    if multiple_locators:
        b.source.locator = "page=1;page=2"
        annotation.locator = b.source.locator
    directory = tmp_path / "data/v8"
    directory.mkdir(parents=True)
    (directory / "knowledge_applicability.json").write_text(
        KnowledgeRegistry(version="1", annotations=[annotation]).model_dump_json()
    )
    relative = b.source.relative_path.removeprefix("data/raw/")
    monkeypatch.setattr("upf_agent.p3_index.load_active_index", lambda root: (tmp_path, {}))
    monkeypatch.setattr(
        "upf_agent.p3_index.load_chunks",
        lambda index: {
            "chunk": SimpleNamespace(
                relative_path=relative,
                product=b.product,
                version=b.version,
                locators=["page=1", "page=2"] if multiple_locators else [b.source.locator],
            )
        },
    )
    # Adapter's existing raw-source projection is retained. Use a root-isolated raw copy.
    path = tmp_path / "data/raw" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((tmp_path / b.source.relative_path).read_bytes())
    entry = evidence(
        AgentTask(run_id=task.run_id, question=task.question, task_type="manual"),
        "chunk",
        b.source.source_id,
        b.source.sha256,
        "page=1, page=2" if multiple_locators else b.source.locator,
        "manual",
        b.fields["instruction"],
    )
    monkeypatch.setattr(
        "upf_agent.agent_runtime.LocalBackend.tool",
        lambda *args: ToolPacket(
            result=ToolResult(status=ResultStatus.COMPLETED, elapsed_ms=0), evidence=[entry]
        ),
    )
    step = QueryStep(
        step_id="manual",
        requirement_ids=[b.requirement_id],
        tool="search_manual",
        object_id=b.object_id,
        arguments={"query": task.question},
        capability="manual",
    )
    packet = collect(tmp_path, task, contract, step, "receipt")
    actual = packet["bindings"][0]
    assert actual["purposes"] == [task.template]
    assert actual["knowledge_applicability"]["annotation_ids"] == [annotation.annotation_id]


def test_empty_knowledge_text_is_unknown_not_a_collector_crash(tmp_path):
    _, _, _, b, entry = mapped(tmp_path)
    b.fields["instruction"] = None
    annotate_knowledge(tmp_path, b, KnowledgeRegistry(version="1", annotations=[entry]))
    assert b.purposes == []
    assert b.knowledge_applicability["status"] == "invalid_content"
