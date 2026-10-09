"""Import sourced topology candidates and bounded log events without inventing identity."""

import hashlib
import json
from pathlib import Path

from upf_agent.contracts import TopologyEdge
from upf_agent.log_tools import LogStore
from upf_agent.v8.builders import manual_graph
from upf_agent.v8.graph import GraphEdge, GraphNode
from upf_agent.v8.models import SourcePointer, digest


def source_graph(root: Path):
    store = manual_graph(root)
    for path in sorted((root / "data/p4/topology").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        raw = Path(manifest["source_path"]).resolve(strict=True)
        if not raw.is_relative_to((root / "data").resolve()):
            raise ValueError("topology source outside isolated root")
        if hashlib.sha256(raw.read_bytes()).hexdigest() != manifest["source_sha256"]:
            raise ValueError("topology source hash mismatch")
        records = json.loads((path.parent / "edges.json").read_text())
        nodes, edges = {}, []
        for record in records:
            edge = TopologyEdge.model_validate_json(json.dumps(record))
            source = SourcePointer(
                source_id=edge.source_id,
                sha256=manifest["source_sha256"],
                relative_path=str(raw.relative_to(root)),
                locator=f"{edge.source_sheet}:{edge.source_row}",
            )
            version = manifest["generation"]
            endpoints = [(edge.device_id, edge.device_port), (edge.peer_device_id, edge.peer_port)]
            ids = []
            for obj, port in endpoints:
                device_id = f"device:{version}:{obj}"
                interface_id = f"interface:{version}:{obj}:{port}"
                for node_id, kind in [(device_id, "Device"), (interface_id, "Interface")]:
                    nodes.setdefault(
                        node_id,
                        GraphNode(
                            node_id=node_id,
                            kind=kind,
                            layer="static",
                            object_ids=[obj],
                            source=source,
                            version=version,
                            synthetic=edge.synthetic,
                            valid_from=edge.valid_from,
                            valid_to=edge.valid_to,
                            attributes={
                                "interface": edge.interface_type,
                                "port": port,
                                "domain": None,
                                "quality_flags": edge.quality_flags,
                                "path_status": "candidate",
                            },
                        ),
                    )
                edges.append(
                    GraphEdge(
                        edge_id=digest([device_id, interface_id]),
                        source_id=device_id,
                        target_id=interface_id,
                        relation="HAS_INTERFACE",
                        source=source,
                        synthetic=edge.synthetic,
                    )
                )
                ids.append(interface_id)
            edges.append(
                GraphEdge(
                    edge_id=f"connection:{version}:{edge.edge_id}",
                    source_id=ids[0],
                    target_id=ids[1],
                    relation="CONNECTS",
                    source=source,
                    synthetic=edge.synthetic,
                    attributes={
                        "claim": "structured topology candidate; no business/domain inference"
                    },
                )
            )
        # A device can occur in more than one source row; keep one deterministic source pointer.
        unique = {e.edge_id: e for e in edges}
        store.build(list(nodes.values()), list(unique.values()))
    for manifest_path in sorted((root / "data/p4/logs").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        log = LogStore(manifest_path.parent.parent, manifest["generation"])
        events = log.find(limit=50)
        raw = Path(manifest["source_path"]).resolve(strict=True)
        if not raw.is_relative_to((root / "data").resolve()):
            raise ValueError("log source outside root")
        if hashlib.sha256(raw.read_bytes()).hexdigest() != manifest["source_sha256"]:
            raise ValueError("log source hash mismatch")
        nodes = []
        for record in events.data.get("events", []):
            source = SourcePointer(
                source_id=manifest["source_id"],
                sha256=manifest["source_sha256"],
                locator=f"bytes:{record.get('byte_offset')}:{record.get('byte_length')}",
                relative_path=str(raw.relative_to(root)),
            )
            nodes.append(
                GraphNode(
                    node_id=f"log:{manifest['generation']}:{record['event_id']}",
                    kind="LogEvent",
                    layer="runtime",
                    source=source,
                    object_ids=[record["device_id"]] if record.get("device_id") else [],
                    synthetic=manifest.get("synthetic", False),
                    version=None,
                    attributes={
                        "event": record,
                        "generation": manifest["generation"],
                        "limitations": [
                            "unknown configuration version",
                            "identity/time remain as parsed",
                        ],
                    },
                )
            )
        store.build(nodes, [])
    return store
