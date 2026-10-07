"""Drain3 offline training and immutable, declarative inference snapshots."""

import hashlib
import json
from pathlib import Path

from drain3 import TemplateMiner
from drain3.drain import LogCluster
from drain3.template_miner_config import TemplateMinerConfig

from upf_agent.log_parser import file_sha, message_body, stream_events
from upf_agent.log_tools import Row, write_json

TEMPLATE_VERSION = "drain3-0.9.11-upfd-v1"


def _config(max_clusters: int) -> TemplateMinerConfig:
    config = TemplateMinerConfig()
    config.drain_max_clusters = max_clusters
    config.drain_sim_th = 0.5
    config.snapshot_interval_minutes = 1000000
    return config


def train_templates(source: Path, root: Path, max_clusters: int = 1024) -> Row:
    if not 2 <= max_clusters <= 10000:
        raise ValueError("max_clusters must be in 2..10000")
    sha = file_sha(source)
    identity = {
        "version": TEMPLATE_VERSION,
        "code_sha256": file_sha(Path(__file__)),
        "source_sha256": sha,
        "max_clusters": max_clusters,
        "sim_th": 0.5,
        "message_body_version": file_sha(Path(__file__).with_name("log_parser.py")),
    }
    snapshot_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    miner = TemplateMiner(config=_config(max_clusters))
    seen = 0
    for event in stream_events(source, f"log-{sha[:16]}", sha):
        if event.parse_status == "parsed":
            miner.add_log_message(message_body(event.raw_excerpt))
            seen += 1
    if file_sha(source) != sha:
        raise ValueError("source changed during template training")
    templates = [
        {
            "cluster_id": cluster.cluster_id,
            "tokens": list(cluster.log_template_tokens),
            "training_messages": cluster.size,
        }
        for cluster in sorted(miner.drain.clusters, key=lambda item: item.cluster_id)
    ]
    directory = root / snapshot_id
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "templates.json"
    content = json.dumps(templates, ensure_ascii=False, indent=2) + "\n"
    if target.exists() and target.read_text() != content:
        raise ValueError("immutable template snapshot conflict")
    target.write_text(content, encoding="utf-8")
    evicted = miner.drain.clusters_counter - len(templates)
    manifest: Row = {
        **identity,
        "snapshot_id": snapshot_id,
        "templates_sha256": file_sha(target),
        "clusters": len(templates),
        "training_messages": seen,
        "evicted_clusters": evicted,
        "unknown_is_anomaly": False,
        "inference": "match_only_full_search_always",
        "warnings": ["LRU cluster limit caused eviction; retained templates are incomplete."]
        if evicted
        else [],
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


class FrozenTemplates:
    def __init__(self, directory: Path) -> None:
        self.manifest = json.loads((directory / "manifest.json").read_text())
        target = directory / "templates.json"
        if file_sha(target) != self.manifest["templates_sha256"]:
            raise ValueError("template snapshot hash mismatch")
        if self.manifest["version"] != TEMPLATE_VERSION:
            raise ValueError("unsupported Drain3 snapshot version")
        self.miner = TemplateMiner(config=_config(self.manifest["max_clusters"]))
        # Restore only primitive JSON token arrays, without deserializing executable objects.
        for row in json.loads(target.read_text()):
            cluster = LogCluster(row["tokens"], row["cluster_id"])
            cluster.size = row["training_messages"]
            self.miner.drain.id_to_cluster[cluster.cluster_id] = cluster
            self.miner.drain.add_seq_to_prefix_tree(self.miner.drain.root_node, cluster)

    def match(self, raw: str) -> str | None:
        cluster = self.miner.match(message_body(raw), full_search_strategy="always")
        return f"{self.manifest['snapshot_id']}:{cluster.cluster_id}" if cluster else None
