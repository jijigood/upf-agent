import json
from pathlib import Path

import pytest

from upf_agent.v8.engine import OrchestrationEngine
from upf_agent.v8.fixtures import example
from upf_agent.v8.graph import (
    GraphEdge,
    GraphNode,
    GraphQueryPlan,
    MemoryGraphStore,
    Neo4jGraphStore,
    compile_cypher,
)
from upf_agent.v8.models import SourcePointer
from upf_agent.v8.store import BudgetStop, Ledger


def graph_fixture():
    source = SourcePointer(source_id="fixture", sha256="a" * 64, locator="/fixture")
    nodes = [
        GraphNode(
            node_id=str(i),
            kind="Observation",
            layer="runtime",
            source=source,
            object_ids=["A"],
            version="5.0",
        )
        for i in range(8)
    ]
    nodes[2].version = "4.0"
    nodes[3].version = None
    nodes[4].synthetic = True
    nodes[5].verified = False
    nodes[6].object_ids = ["outside"]
    edges = [
        GraphEdge(
            edge_id=f"edge-{i}",
            source_id="0",
            target_id=str(i),
            relation="OBSERVED_ON",
            source=source,
            corroborated=i != 7,
        )
        for i in range(1, 8)
    ]
    return nodes, edges


def test_memory_filters_limits_history_and_parameters(tmp_path):
    task = example(tmp_path)
    nodes, edges = graph_fixture()
    nodes[1].valid_to = task.as_of
    store = MemoryGraphStore()
    store.build(nodes, edges)
    plan = GraphQueryPlan(
        seed_ids=["0"],
        objects=["A"],
        relations=["OBSERVED_ON"],
        as_of=task.as_of,
        version="5.0",
        contract_id="c",
    )
    result = store.query(plan)
    assert [n.node_id for n in result.nodes] == ["0"]
    assert len(result.excluded) == 7
    query, params = compile_cypher(plan, ["0' MATCH (n) DETACH DELETE n //"])
    assert "DETACH" not in query
    assert "DETACH" in params["frontier"][0]
    assert "LIMIT $limit" in query
    with pytest.raises(ValueError):
        plan.model_copy().relations = ["free"]  # assignment isn't validated


def test_graph_validation_immutable_and_hops(tmp_path):
    task = example(tmp_path)
    nodes, edges = graph_fixture()
    store = MemoryGraphStore()
    store.build(nodes, edges)
    with pytest.raises(ValueError):
        store.build([nodes[0].model_copy(update={"version": "6.0"})], [])
    with pytest.raises(ValueError):
        GraphQueryPlan(
            seed_ids=[], objects=[], relations=["evil"], as_of=task.as_of, contract_id="c"
        )
    with pytest.raises(ValueError):
        GraphQueryPlan(
            seed_ids=[], objects=[], relations=[], as_of=task.as_of, contract_id="c", max_hops=3
        )
    assert MemoryGraphStore.restore(store.snapshot()).snapshot() == store.snapshot()


def test_engine_complete_interrupt_resume_and_scope(tmp_path, monkeypatch):
    monkeypatch.setattr("upf_agent.v8.engine.load_manual_graph", lambda _root: MemoryGraphStore())
    task = example(tmp_path)
    task.interrupt_before_collect = True
    engine = OrchestrationEngine(tmp_path)
    try:
        assert engine.run(task)["status"] == "interrupted"
        result = engine.resume(task.run_id)
        assert result["status"] == "completed"
        assert all(a["outcome"] == "suggest_allowed" for a in result["actions"])
        before = engine.ledger.inspect(task.run_id)["receipts"]
        assert engine.resume(task.run_id) == result
        assert engine.ledger.inspect(task.run_id)["receipts"] == before
        with pytest.raises(ValueError):
            engine.run(task.model_copy(update={"objects": ["new"]}))
        assert (tmp_path / "runs/v8" / task.run_id / "report.md").is_file()
    finally:
        engine.close()


def test_engine_missing_source_and_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("upf_agent.v8.engine.load_manual_graph", lambda _root: MemoryGraphStore())
    task = example(tmp_path)
    task.budget.max_tool_calls = 1
    engine = OrchestrationEngine(tmp_path)
    try:
        result = engine.run(task)
        assert result["stop_reason"] == "tool_budget_exhausted"
        assert all(a["outcome"] != "suggest_allowed" for a in result["actions"])
        assert len(engine.ledger.inspect(task.run_id)["receipts"]) == 1
    finally:
        engine.close()


def test_ledger_crash_receipts_revision_collision_and_cross_task(tmp_path):
    task = example(tmp_path)
    ledger = Ledger(tmp_path)
    ledger.create(task)
    try:
        assert ledger.charge(task, "x", "tool", {"object": "A"}) is None
        with pytest.raises(BudgetStop):
            ledger.charge(task, "x", "tool", {"object": "A"})
        with pytest.raises(ValueError):
            ledger.charge(task, "x", "tool", {"object": "B"})
        ledger.commit_receipt(task, "x", {"done": True}, [], 0.1)
        assert ledger.charge(task, "x", "tool", {"object": "A"}) == {"done": True}
    finally:
        ledger.close()


def test_neo4j_same_semantics():
    root = Path(__file__).resolve().parents[1] / "runtime/v8"
    config_path = root / "config/v8-neo4j.json"
    if not config_path.is_file():
        pytest.skip("run scripts/v8_neo4j.py to enable the real backend integration")
    config = json.loads(config_path.read_text())
    nodes, edges = graph_fixture()
    memory = MemoryGraphStore()
    memory.build(nodes, edges)
    neo = Neo4jGraphStore(**config, namespace="v8-semantic-tests-v1")
    try:
        neo.build(nodes, edges)
        assert neo.snapshot() == memory.snapshot()
        task = example(root, "backend-query")
        for cap in (1, 3, 100):
            for synthetic in (False, True):
                for hops in (0, 1, 2):
                    plan = GraphQueryPlan(
                        seed_ids=["0"],
                        objects=["A"],
                        relations=["OBSERVED_ON"],
                        as_of=task.as_of,
                        version="5.0",
                        contract_id="c",
                        max_nodes=cap,
                        allow_synthetic=synthetic,
                        max_hops=hops,
                    )
                    assert neo.query(plan) == memory.query(plan)
        with pytest.raises(ValueError):
            neo.build([nodes[0].model_copy(update={"version": "6.0"})], [])
    finally:
        neo.close()
