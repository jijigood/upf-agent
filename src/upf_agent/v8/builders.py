"""Deterministic graph construction from validated chunks, paths and receipts."""

import hashlib
import json
from pathlib import Path

from upf_agent.p3_index import load_active_index, load_chunks
from upf_agent.v8.graph import GraphEdge, GraphNode, MemoryGraphStore
from upf_agent.v8.models import EvidenceBinding, OrchestrationTask, SourcePointer, digest


def manual_graph(root: Path) -> MemoryGraphStore:
    directory, manifest = load_active_index(root)
    if (
        hashlib.sha256((directory / "chunks.jsonl").read_bytes()).hexdigest()
        != manifest["chunks_sha256"]
    ):
        raise ValueError("chunk archive hash mismatch")
    from upf_agent.v8.knowledge_applicability import KnowledgeRegistry

    registry = KnowledgeRegistry.load(root)
    nodes, edges, sections = [], [], {}
    sources = {}
    for chunk in load_chunks(directory).values():
        source_file = (root / "data/raw" / chunk.relative_path).resolve()
        if not source_file.is_relative_to((root / "data/raw").resolve()):
            raise ValueError("source outside raw corpus")
        if chunk.source_id not in sources:
            sources[chunk.source_id] = hashlib.sha256(source_file.read_bytes()).hexdigest()
        if sources[chunk.source_id] != chunk.source_sha256:
            raise ValueError("manual source hash mismatch")
        source = SourcePointer(
            source_id=chunk.source_id,
            sha256=chunk.source_sha256,
            relative_path=f"data/raw/{chunk.relative_path}",
            locator=";".join(chunk.locators) or chunk.chunk_id,
        )
        node = GraphNode(
            node_id=chunk.chunk_id,
            kind="Command" if chunk.kind == "command" else "DocumentBlock",
            layer="knowledge",
            source=source,
            version=chunk.version,
            attributes={
                "text": chunk.text,
                "product": chunk.product,
                "block_ids": chunk.block_ids,
                "generation": manifest["index_generation"],
                "knowledge_registry_sha256": digest(registry),
            },
        )
        nodes.append(node)
        section_id = f"section:{chunk.source_id}:{digest(chunk.section_path)[:16]}"
        if section_id not in sections:
            sections[section_id] = GraphNode(
                node_id=section_id,
                kind="Section",
                layer="knowledge",
                source=source,
                version=chunk.version,
                attributes={"section_path": chunk.section_path},
            )
        edges.append(
            GraphEdge(
                edge_id=digest([section_id, chunk.chunk_id]),
                source_id=section_id,
                target_id=chunk.chunk_id,
                relation="CONTAINS",
                source=source,
            )
        )
        for interface in sorted(set(chunk.interface_tags) & {"N3", "N4", "N6", "N9"}):
            concept = f"concept:{chunk.source_id}:{chunk.version}:{interface}"
            if concept not in sections:
                sections[concept] = GraphNode(
                    node_id=concept,
                    kind="InterfaceConcept",
                    layer="knowledge",
                    source=source,
                    version=chunk.version,
                    attributes={"interface": interface, "physical": False},
                )
            edges.append(
                GraphEdge(
                    edge_id=digest([chunk.chunk_id, concept]),
                    source_id=chunk.chunk_id,
                    target_id=concept,
                    relation="DESCRIBES",
                    source=source,
                )
            )
    store = MemoryGraphStore()
    store.build(nodes + list(sections.values()), edges)
    return store


def task_graph(task: OrchestrationTask, bindings: list[EvidenceBinding]):
    nodes, edges = [], []
    path_source = SourcePointer(
        source_id=task.path.source_id or "unresolved-path",
        sha256=task.path.source_sha256 or "0" * 64,
        locator="/path",
        relative_path=task.path.source_path,
    )
    for position in task.path.positions:
        nodes.append(
            GraphNode(
                node_id=f"object:{position.object_id}",
                kind="Device",
                layer="static",
                object_ids=[position.object_id],
                source=path_source,
                version=task.path.version,
                synthetic=task.path.synthetic,
                verified=task.path.resolution == "resolved" and bool(task.path.source_sha256),
                attributes={
                    "domain": position.domain,
                    "interface": position.interface,
                    "service_id": task.path.service_id,
                },
            )
        )
        if position.domain is not None:
            domain_id = f"domain:{task.path.source_sha256}:{position.domain}"
            if not any(n.node_id == domain_id for n in nodes):
                nodes.append(
                    GraphNode(
                        node_id=domain_id,
                        kind="Domain",
                        layer="static",
                        source=path_source,
                        version=task.path.version,
                        synthetic=task.path.synthetic,
                        verified=task.path.resolution == "resolved",
                        attributes={"domain": position.domain},
                    )
                )
            edges.append(
                GraphEdge(
                    edge_id=digest([domain_id, position.object_id]),
                    source_id=f"object:{position.object_id}",
                    target_id=domain_id,
                    relation="IN_DOMAIN",
                    source=path_source,
                    synthetic=task.path.synthetic,
                    corroborated=task.path.resolution == "resolved",
                )
            )
    for left, right in zip(task.path.positions, task.path.positions[1:], strict=False):
        edges.append(
            GraphEdge(
                edge_id=digest([task.path.source_id, left.object_id, right.object_id]),
                source_id=f"object:{left.object_id}",
                target_id=f"object:{right.object_id}",
                relation="CARRIES" if left.domain == right.domain else "ADMIN_BOUNDARY",
                source=path_source,
                synthetic=task.path.synthetic,
                corroborated=task.path.resolution == "resolved",
            )
        )
    known_objects = {p.object_id for p in task.path.positions}
    for binding in bindings:
        if binding.deleted:
            continue
        node_id = f"binding:{binding.evidence_id}:{binding.revision}"
        nodes.append(
            GraphNode(
                node_id=node_id,
                kind="EvidenceBinding",
                layer="runtime",
                source=binding.source,
                object_ids=[binding.object_id] if binding.object_id else [],
                version=binding.version,
                synthetic=binding.synthetic,
                verified=binding.source_verified,
                valid_from=binding.valid.start if binding.valid else None,
                valid_to=binding.valid.end if binding.valid else None,
                attributes={"binding": binding.model_dump(mode="json")},
            )
        )
        if binding.object_id in known_objects:
            edges.append(
                GraphEdge(
                    edge_id=digest([node_id, binding.object_id]),
                    source_id=node_id,
                    target_id=f"object:{binding.object_id}",
                    relation="OBSERVED_ON",
                    source=binding.source,
                    synthetic=binding.synthetic,
                    corroborated=binding.corroborated,
                )
            )
        receipt_id = f"receipt:{binding.task_id}:{binding.receipt_id}"
        if not any(n.node_id == receipt_id for n in nodes):
            nodes.append(
                GraphNode(
                    node_id=receipt_id,
                    kind="ToolReceipt",
                    layer="runtime",
                    source=binding.source,
                    object_ids=[],
                    version=binding.version,
                    synthetic=binding.synthetic,
                    verified=binding.source_verified,
                )
            )
        edges.append(
            GraphEdge(
                edge_id=digest([receipt_id, node_id]),
                source_id=receipt_id,
                target_id=node_id,
                relation="RECEIPT_FOR",
                source=binding.source,
                synthetic=binding.synthetic,
                corroborated=binding.corroborated,
            )
        )
    return nodes, edges


def load_manual_graph(root: Path) -> MemoryGraphStore:
    path = root / "data/v8/manual_graph.json"
    if not path.exists():
        store = manual_graph(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store.snapshot(), ensure_ascii=False) + "\n")
        return store
    return MemoryGraphStore.restore(json.loads(path.read_text()))
