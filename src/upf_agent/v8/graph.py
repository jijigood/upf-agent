"""Three sourced logical graphs with identical bounded Memory and Neo4j semantics."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Protocol

from neo4j import GraphDatabase
from pydantic import Field, model_validator

from upf_agent.v8.models import Model, OrchestrationTask, SourcePointer, digest

RELATIONS = frozenset(
    {
        "HAS_INTERFACE",
        "CONNECTS",
        "IN_DOMAIN",
        "CARRIES",
        "DEPENDS_ON",
        "APPLIES_TO",
        "CONTAINS",
        "DESCRIBES",
        "SUPPORTED_BY",
        "OBSERVED_ON",
        "RECEIPT_FOR",
        "ADMIN_BOUNDARY",
    }
)


class GraphNode(Model):
    node_id: str
    kind: str
    layer: str
    object_ids: list[str] = Field(default_factory=list)
    source: SourcePointer
    version: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    synthetic: bool = False
    verified: bool = True
    attributes: dict = Field(default_factory=dict)


class GraphEdge(Model):
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    source: SourcePointer
    corroborated: bool = True
    synthetic: bool = False
    attributes: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def whitelist(self):
        if self.relation not in RELATIONS:
            raise ValueError("relationship type is not allowlisted")
        return self


class GraphQueryPlan(Model):
    seed_ids: list[str] = Field(max_length=100)
    objects: list[str]
    domains: list[str] = Field(default_factory=list)
    relations: list[str]
    as_of: datetime
    version: str | None = None
    require_version: bool = True
    allow_synthetic: bool = False
    max_hops: int = Field(default=2, ge=0, le=2)
    max_nodes: int = Field(default=100, ge=1, le=100)
    max_edges: int = Field(default=200, ge=1, le=200)
    contract_id: str

    @model_validator(mode="after")
    def validate_query(self):
        if not set(self.relations) <= RELATIONS:
            raise ValueError("relationship type is not allowlisted")
        if self.as_of.utcoffset() is None:
            raise ValueError("query time must have a timezone")
        return self


class Subgraph(Model):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    excluded: list[dict]
    truncated: bool = False
    plan_fingerprint: str


def compile_graph_query(
    task: OrchestrationTask, contract_id: str, seeds: list[str], relations: list[str] | None = None
) -> GraphQueryPlan:
    return GraphQueryPlan(
        seed_ids=seeds[:100],
        objects=[o for o in task.objects if o in task.permission.objects],
        domains=task.permission.domains,
        relations=relations
        or ["APPLIES_TO", "DESCRIBES", "CONTAINS", "HAS_INTERFACE", "SUPPORTED_BY", "OBSERVED_ON"],
        as_of=task.as_of,
        version=task.version,
        allow_synthetic=task.synthetic,
        contract_id=contract_id,
    )


def exclusion(node: GraphNode, plan: GraphQueryPlan) -> str | None:
    if (
        node.kind == "EvidenceBinding"
        and node.attributes.get("binding", {}).get("contract_id") != plan.contract_id
    ):
        return "cross_task_or_contract"
    if not node.verified:
        return "unverified_source"
    if node.synthetic and not plan.allow_synthetic:
        return "synthetic_in_real_task"
    if node.object_ids and not set(node.object_ids) <= set(plan.objects):
        return "outside_object_scope"
    if plan.domains and node.attributes.get("domain") is not None:
        if node.attributes["domain"] not in plan.domains:
            return "outside_domain_scope"
    if node.layer in {"knowledge", "runtime"} and plan.require_version:
        if not plan.version or not node.version or node.version == "unknown":
            return "unknown_version"
        if node.version != plan.version:
            return "wrong_version"
    if node.valid_from and node.valid_from > plan.as_of:
        return "not_yet_valid"
    if node.valid_to and node.valid_to <= plan.as_of:
        return "expired"
    return None


class EvidenceGraphStore(Protocol):
    def build(self, nodes: list[GraphNode], edges: list[GraphEdge]) -> None: ...
    def snapshot(self) -> dict: ...
    def query(self, plan: GraphQueryPlan) -> Subgraph: ...


class MemoryGraphStore:
    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}
        self.adjacency: dict[str, list[GraphEdge]] = defaultdict(list)

    def build(self, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
        for node in nodes:
            previous = self.nodes.get(node.node_id)
            if previous is not None and previous != node:
                raise ValueError("graph facts are immutable; use a revision-specific node ID")
            self.nodes[node.node_id] = node
        for edge in edges:
            if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
                raise ValueError("edge has an unknown endpoint")
            if edge.edge_id in self.edges and self.edges[edge.edge_id] != edge:
                raise ValueError("graph edges are immutable")
            if edge.edge_id not in self.edges:
                self.edges[edge.edge_id] = edge
                self.adjacency[edge.source_id].append(edge)
                self.adjacency[edge.target_id].append(edge)

    def snapshot(self) -> dict:
        return {
            "nodes": [self.nodes[k].model_dump(mode="json") for k in sorted(self.nodes)],
            "edges": [self.edges[k].model_dump(mode="json") for k in sorted(self.edges)],
        }

    def _nodes(self, ids: list[str]) -> list[GraphNode]:
        return [self.nodes[i] for i in sorted(set(ids)) if i in self.nodes]

    def _neighbors(self, ids: list[str], plan: GraphQueryPlan) -> list[GraphEdge]:
        return sorted(
            {
                e.edge_id: e
                for i in ids
                for e in self.adjacency.get(i, [])
                if e.relation in plan.relations
            }.values(),
            key=lambda e: e.edge_id,
        )[: plan.max_edges + 1]

    def query(self, plan: GraphQueryPlan) -> Subgraph:
        accepted: dict[str, GraphNode] = {}
        used: dict[str, GraphEdge] = {}
        rejected: dict[str, dict] = {}
        truncated = False

        def admit(ids):
            nonlocal truncated
            for node in self._nodes(ids):
                reason = exclusion(node, plan)
                if reason:
                    rejected[node.node_id] = {
                        "id": node.node_id,
                        "reason": reason,
                        "source": node.source.model_dump(mode="json"),
                    }
                elif node.node_id not in accepted:
                    if len(accepted) < plan.max_nodes:
                        accepted[node.node_id] = node
                    else:
                        truncated = True

        admit(plan.seed_ids)
        frontier = sorted(accepted)
        seen = set(frontier)
        for _ in range(plan.max_hops):
            candidates = self._neighbors(frontier, plan)
            if len(candidates) > plan.max_edges:
                truncated = True
                candidates = candidates[: plan.max_edges]
            safe = []
            for edge in candidates:
                if not edge.corroborated or edge.synthetic and not plan.allow_synthetic:
                    rejected[edge.edge_id] = {
                        "id": edge.edge_id,
                        "reason": "unverified_relation",
                        "source": edge.source.model_dump(mode="json"),
                    }
                else:
                    safe.append(edge)
            next_ids = sorted({n for e in safe for n in (e.source_id, e.target_id)} - seen)
            admit(next_ids)
            for edge in safe:
                if edge.source_id in accepted and edge.target_id in accepted:
                    if len(used) < plan.max_edges or edge.edge_id in used:
                        used[edge.edge_id] = edge
                    else:
                        truncated = True
            frontier = sorted(set(accepted) - seen)
            seen.update(frontier)
            if not frontier:
                break
        return Subgraph(
            nodes=[accepted[k] for k in sorted(accepted)],
            edges=[used[k] for k in sorted(used)],
            excluded=[rejected[k] for k in sorted(rejected)],
            truncated=truncated,
            plan_fingerprint=digest(plan),
        )

    @classmethod
    def restore(cls, snapshot: dict) -> MemoryGraphStore:
        store = cls()
        store.build(
            [GraphNode.model_validate(n) for n in snapshot["nodes"]],
            [GraphEdge.model_validate(e) for e in snapshot["edges"]],
        )
        return store


def compile_cypher(plan: GraphQueryPlan, frontier: list[str]) -> tuple[str, dict]:
    # Structure is fixed. Labels, edge kinds and object values never come from LLM text.
    return (
        "MATCH (n:V8Fact)-[r:V8Relation]-(m:V8Fact) "
        "WHERE n.namespace=$namespace AND m.namespace=$namespace "
        "AND n.id IN $frontier AND r.relation IN $relations "
        "RETURN DISTINCT r.payload AS payload,r.id AS id ORDER BY id LIMIT $limit",
        {
            "namespace": "",
            "frontier": frontier,
            "relations": plan.relations,
            "limit": plan.max_edges + 1,
        },
    )


class Neo4jGraphStore(MemoryGraphStore):
    def __init__(self, uri: str, username: str, password: str, namespace: str = "v8"):
        super().__init__()
        self.namespace = namespace
        self.driver = GraphDatabase.driver(
            uri, auth=(username, password), connection_timeout=5, max_transaction_retry_time=5
        )
        self.driver.verify_connectivity()
        self.driver.execute_query(
            "CREATE CONSTRAINT v8_fact_identity IF NOT EXISTS "
            "FOR (n:V8Fact) REQUIRE (n.namespace,n.id) IS UNIQUE"
        )

    def close(self):
        self.driver.close()

    def build(self, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
        # Check collisions against the database, not just a local cache.
        prior = self._nodes([n.node_id for n in nodes])
        by_id = {n.node_id: n for n in prior}
        if any(n.node_id in by_id and by_id[n.node_id] != n for n in nodes):
            raise ValueError("immutable graph node collision")
        rows = [{"id": n.node_id, "payload": n.model_dump_json()} for n in nodes]
        self.driver.execute_query(
            "UNWIND $rows AS row MERGE (n:V8Fact {namespace:$namespace,id:row.id}) "
            "ON CREATE SET n.payload=row.payload",
            rows=rows,
            namespace=self.namespace,
        )
        known = {
            n.node_id
            for n in self._nodes(sorted({i for e in edges for i in (e.source_id, e.target_id)}))
        }
        if any(e.source_id not in known or e.target_id not in known for e in edges):
            raise ValueError("edge has an unknown endpoint")
        prior_edges = self.driver.execute_query(
            "MATCH (:V8Fact {namespace:$namespace})-[r:V8Relation]->() "
            "WHERE r.id IN $ids RETURN r.payload AS payload",
            namespace=self.namespace,
            ids=[e.edge_id for e in edges],
        ).records
        old = {
            e.edge_id: e for e in (GraphEdge.model_validate_json(r["payload"]) for r in prior_edges)
        }
        if any(e.edge_id in old and old[e.edge_id] != e for e in edges):
            raise ValueError("immutable graph edge collision")
        self.driver.execute_query(
            "UNWIND $rows AS row MATCH (a:V8Fact {namespace:$namespace,id:row.a}), "
            "(b:V8Fact {namespace:$namespace,id:row.b}) "
            "MERGE (a)-[r:V8Relation {id:row.id}]->(b) "
            "ON CREATE SET r.payload=row.payload,r.relation=row.relation",
            rows=[
                {
                    "id": e.edge_id,
                    "a": e.source_id,
                    "b": e.target_id,
                    "payload": e.model_dump_json(),
                    "relation": e.relation,
                }
                for e in edges
            ],
            namespace=self.namespace,
        )

    def _nodes(self, ids: list[str]) -> list[GraphNode]:
        records = self.driver.execute_query(
            "MATCH (n:V8Fact {namespace:$namespace}) WHERE n.id IN $ids "
            "RETURN n.payload AS payload ORDER BY n.id",
            namespace=self.namespace,
            ids=ids,
        ).records
        return [GraphNode.model_validate_json(r["payload"]) for r in records]

    def _neighbors(self, ids: list[str], plan: GraphQueryPlan) -> list[GraphEdge]:
        query, params = compile_cypher(plan, ids)
        params["namespace"] = self.namespace
        return [
            GraphEdge.model_validate_json(r["payload"])
            for r in self.driver.execute_query(query, **params).records
        ]

    def snapshot(self) -> dict:
        nodes = self.driver.execute_query(
            "MATCH (n:V8Fact {namespace:$namespace}) RETURN n.payload AS payload ORDER BY n.id",
            namespace=self.namespace,
        ).records
        edges = self.driver.execute_query(
            "MATCH (:V8Fact {namespace:$namespace})-[r:V8Relation]->() "
            "RETURN r.payload AS payload ORDER BY r.id",
            namespace=self.namespace,
        ).records
        return {
            "nodes": [json.loads(r["payload"]) for r in nodes],
            "edges": [json.loads(r["payload"]) for r in edges],
        }


def save_snapshot(store: EvidenceGraphStore, path: Path) -> str:
    snapshot = store.snapshot()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    return digest(snapshot)
