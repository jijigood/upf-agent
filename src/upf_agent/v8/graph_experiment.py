"""Same-P3-seed retrieval comparisons and a separately reported official baseline."""

import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter

from upf_agent.config import load_model_config
from upf_agent.p3_index import load_active_index
from upf_agent.p3_search import search
from upf_agent.retrieval import RetrievalWorker
from upf_agent.v8.builders import load_manual_graph
from upf_agent.v8.evaluation import write_json
from upf_agent.v8.graph import (
    GraphEdge,
    GraphNode,
    GraphQueryPlan,
    MemoryGraphStore,
    Neo4jGraphStore,
)
from upf_agent.v8.models import SourcePointer, digest

QUERIES = [
    "UPF 5.0 安装部署 检查网络接口",
    "UPF 5.0 N4 PFCP 配置",
    "UPF 5.0 启动 upfmgr upfd",
    "UPF 5.0 N3 GTP-U 网络接口",
]


class RecordingWorker:
    def __init__(self, worker):
        self.worker, self.vectors = worker, {}

    def request(self, body, **kwargs):
        value = self.worker.request(body, **kwargs)
        if body.get("role") == "query":
            self.vectors[body["texts"][0]] = value["vectors"][0]
        return value


def ordinary_expand(store, seed_ids, max_hops=2, max_nodes=100):
    selected, frontier = set(seed_ids[:max_nodes]), set(seed_ids[:max_nodes])
    for _ in range(max_hops):
        neighbors = sorted(
            {
                n
                for i in frontier
                for e in store.adjacency.get(i, [])
                for n in (e.source_id, e.target_id)
            }
            - selected
        )
        frontier = set(neighbors[: max_nodes - len(selected)])
        selected.update(frontier)
    return sorted(selected)


def graph_experiment(root: Path) -> dict:
    output = root / "experiments/v8/graph"
    output.mkdir(parents=True, exist_ok=True)
    config = load_model_config(root / "config/models.yaml")
    base = load_manual_graph(root)
    index, manifest = load_active_index(root)
    results, official = [], []
    with RetrievalWorker(root, "embed", config.embedding.path, "cuda:1") as embedder:
        with RetrievalWorker(root, "rerank", config.reranker.path, "cuda:1") as reranker:
            recording = RecordingWorker(embedder)
            for case_index, question in enumerate(QUERIES):
                started = perf_counter()
                retrieved = search(root, question, "UPF", "5.0", workers=(recording, reranker))
                write_json(output / f"p3-query-{case_index}.json", retrieved)
                seeds = [e["chunk_id"] for e in retrieved["evidence"]][:8]
                if not seeds:
                    raise ValueError("no P3 seeds for the declared graph experiment")
                p3_time = perf_counter() - started
                store = MemoryGraphStore.restore(base.snapshot())
                # Controlled, sourced synthetic facts test applicability separately from relevance.
                raw = {
                    "synthetic": True,
                    "case": case_index,
                    "required": "good",
                    "controls": [
                        "wrong_version",
                        "unknown_version",
                        "expired",
                        "unverified",
                        "wrong_object",
                        "unsupported_relation",
                    ],
                }
                fixture_path = root / f"data/fixtures/v8-graph/{case_index}.json"
                write_json(fixture_path, raw)
                source = SourcePointer(
                    source_id=f"graph-oracle-{case_index}",
                    sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                    locator="/controls",
                    relative_path=str(fixture_path.relative_to(root)),
                )
                from datetime import UTC, datetime, timedelta

                now = datetime(2026, 10, 7, tzinfo=UTC)
                nodes, edges = [], []
                for kind in ["good", *raw["controls"]]:
                    node = GraphNode(
                        node_id=f"control-{case_index}-{kind}",
                        kind="Observation",
                        layer="runtime",
                        object_ids=["UPF-A"],
                        source=source,
                        version="5.0",
                        synthetic=True,
                    )
                    if kind == "wrong_version":
                        node.version = "4.0"
                    if kind == "unknown_version":
                        node.version = None
                    if kind == "expired":
                        node.valid_to = now - timedelta(seconds=1)
                    if kind == "unverified":
                        node.verified = False
                    if kind == "wrong_object":
                        node.object_ids = ["outside"]
                    nodes.append(node)
                    edges.append(
                        GraphEdge(
                            edge_id=f"control-edge-{case_index}-{kind}",
                            source_id=seeds[0],
                            target_id=node.node_id,
                            relation="OBSERVED_ON",
                            source=source,
                            synthetic=True,
                            corroborated=kind != "unsupported_relation",
                        )
                    )
                store.build(nodes, edges)
                write_json(output / f"snapshot-{case_index}.json", store.snapshot())
                plan = GraphQueryPlan(
                    seed_ids=seeds,
                    objects=["UPF-A"],
                    relations=["CONTAINS", "OBSERVED_ON"],
                    as_of=now,
                    version="5.0",
                    allow_synthetic=True,
                    contract_id=f"graph-case-{case_index}",
                )
                started = perf_counter()
                ordinary = ordinary_expand(store, seeds)
                ordinary_time = perf_counter() - started
                started = perf_counter()
                constrained = store.query(plan)
                constrained_time = perf_counter() - started
                required = {f"control-{case_index}-good"}
                invalid = {f"control-{case_index}-{k}" for k in raw["controls"]}
                applicable = {
                    k
                    for k, n in store.nodes.items()
                    if n.layer == "knowledge" and n.version == "5.0"
                } | required
                methods = {
                    "Hybrid": seeds,
                    "FixedGraph": ordinary,
                    "ContractGraph": [n.node_id for n in constrained.nodes],
                }
                for method, returned in methods.items():
                    present = set(returned)
                    results.append(
                        {
                            "case": case_index,
                            "method": method,
                            "returned": returned,
                            "required_coverage": len(present & required) / len(required),
                            "applicability_precision": len(present & applicable) / len(present)
                            if present
                            else 0,
                            "invalid_accepted": sorted(present & invalid),
                            "source_traceable": all(
                                store.nodes[i].source.locator
                                and store.nodes[i].source.relative_path
                                for i in present
                            ),
                            "duration_seconds": p3_time
                            if method == "Hybrid"
                            else ordinary_time
                            if method == "FixedGraph"
                            else constrained_time,
                        }
                    )
                # The complete real database is tested, not mocked as a successful backend.
                neo_cfg = json.loads((root / "config/v8-neo4j.json").read_text())
                neo = Neo4jGraphStore(
                    **neo_cfg, namespace=f"graph-experiment-{digest(store.snapshot())[:16]}"
                )
                try:
                    neo.build(list(store.nodes.values()), list(store.edges.values()))
                    if neo.query(plan) != constrained:
                        raise AssertionError("Memory/Neo4j semantic mismatch")
                finally:
                    neo.close()
                write_json(
                    output / f"subgraph-{case_index}.json", constrained.model_dump(mode="json")
                )
                official.append({"question": question, "vector": recording.vectors[question]})
    request = {
        "root": str(root),
        "generation": manifest["index_generation"],
        "queries": official,
        "output": str(output / "official-hybrid-cypher.json"),
    }
    write_json(output / "official-request.json", request)
    project = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            str(project / ".venv-graphrag/bin/python"),
            str(project / "scripts/v8_hybrid_cypher.py"),
            str(output / "official-request.json"),
        ],
        check=True,
        timeout=180,
    )
    summary = {
        "cases": len(QUERIES),
        "results": results,
        "backend_semantics_equal": True,
        "source_traceable": all(r["source_traceable"] for r in results),
        "contract_invalid_accepted": sum(
            len(r["invalid_accepted"]) for r in results if r["method"] == "ContractGraph"
        ),
        "limitations": [
            "synthetic controls measure applicability, not human semantic relevance",
            "official HybridCypher uses a different retrieval substrate and is supplemental",
        ],
    }
    summary["pass"] = summary["source_traceable"] and summary["contract_invalid_accepted"] == 0
    write_json(output / "summary.json", summary)
    return summary
