"""Read-only source audit and legacy baseline extraction for P0."""

import ast
import hashlib
import json
import platform
import re
import runpy
import subprocess
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import yaml

from upf_agent.contracts import SourceRef

WORD = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
ALLOWED_LEGACY_IMPORTS = {"re", "datetime", "collections"}
LOG_HEADER = re.compile(r"^(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d:\d\d[.,]\d+)")
COMMAND_TERMS = (
    "set-log-level",
    "upfmgr mml",
    "MML",
    "37100",
    "5002",
    "VPP",
    "PFCP",
    "GTP-U",
    "N3",
    "N4",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def docx_inventory(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as document:
        root = ElementTree.fromstring(document.read("word/document.xml"))
        text = "".join(node.text or "" for node in root.findall(".//w:t", WORD))
        media = [
            name
            for name in document.namelist()
            if name.startswith("word/media/") and not name.endswith("/")
        ]
        return {
            "paragraph_elements_including_nested": len(root.findall(".//w:p", WORD)),
            "tables": len(root.findall(".//w:tbl", WORD)),
            "table_rows": len(root.findall(".//w:tr", WORD)),
            "textboxes": len(root.findall(".//w:txbxContent", WORD)),
            "media_files": len(media),
            "term_occurrences_in_ooxml_text": {
                term: len(re.findall(re.escape(term), text, flags=re.IGNORECASE))
                for term in COMMAND_TERMS
            },
            "text_count_method": (
                "Each OOXML w:t element once; excludes image-only text; no gold labels"
            ),
        }


def log_inventory(path: Path) -> dict[str, object]:
    total = 0
    nonblank = 0
    dated = 0
    first_time: str | None = None
    last_time: str | None = None
    prior_time: str | None = None
    adjacent_out_of_order = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            total += 1
            if not line.strip():
                continue
            nonblank += 1
            match = LOG_HEADER.match(line)
            if match is None:
                continue
            dated += 1
            current_time = f"{match.group(1)} {match.group(2).replace(',', '.')}"
            if first_time is None:
                first_time = current_time
            if prior_time is not None and current_time < prior_time:
                adjacent_out_of_order += 1
            prior_time = current_time
            last_time = current_time
    return {
        "lines": total,
        "nonblank_lines": nonblank,
        "timestamped_lines": dated,
        "first_timestamp_in_file_order": first_time,
        "last_timestamp_in_file_order": last_time,
        "adjacent_out_of_order_pairs": adjacent_out_of_order,
        "timezone": "unknown",
    }


def sources_audit(corpus_config: Path, output: Path) -> list[dict[str, object]]:
    configuration = yaml.safe_load(corpus_config.read_text(encoding="utf-8"))
    source_root = Path(configuration["source_root"]).resolve()
    result: list[dict[str, object]] = []
    for item in configuration["sources"]:
        path = (source_root / item["path"]).resolve()
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError(f"missing or escaped source file: {item['path']}")
        checksum = sha256_file(path)
        record = SourceRef(
            source_id=f"{item['role']}-{checksum[:16]}",
            sha256=checksum,
            relative_path=item["path"],
            role=item["role"],
            product=item.get("product"),
            version=item.get("version"),
        ).model_dump()
        record["size_bytes"] = path.stat().st_size
        if path.suffix.lower() == ".docx":
            record["structure"] = docx_inventory(path)
        elif path.suffix.lower() == ".log":
            record["structure"] = log_inventory(path)
        elif path.suffix.lower() == ".pdf":
            pdfinfo = subprocess.run(
                ["pdfinfo", str(path)], capture_output=True, text=True, check=False, timeout=30
            )
            match = re.search(r"^Pages:\s+(\d+)", pdfinfo.stdout, flags=re.MULTILINE)
            record["structure"] = {"pdf_pages": int(match.group(1)) if match else None}
        result.append(record)
    write_json(output / "sources.json", result)
    return result


def models_audit(models_config: Path, output: Path) -> list[dict[str, object]]:
    configuration = yaml.safe_load(models_config.read_text(encoding="utf-8"))
    entries = [
        configuration["generation"]["baseline"],
        configuration["generation"]["target"],
        configuration["embedding"],
        configuration["reranker"],
    ]
    result: list[dict[str, object]] = []
    for entry in entries:
        root = Path(entry["path"])
        config_path = root / "config.json"
        index_path = root / "model.safetensors.index.json"
        weights = sorted(root.glob("*.safetensors")) if root.is_dir() else []
        indexed = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
        expected = sorted(set(indexed.get("weight_map", {}).values()))
        missing = [name for name in expected if not (root / name).is_file()]
        ready = bool(config_path.is_file() and index_path.is_file() and expected and not missing)
        if ready:
            ready = len(expected) == len(weights)
        result.append(
            {
                "model_id": entry["id"],
                "path": str(root),
                "exists": root.is_dir(),
                "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
                "indexed_shards": expected,
                "missing_shards": missing,
                "ready_for_weight_load": ready,
                "safetensors_files": [
                    {"name": path.name, "size_bytes": path.stat().st_size} for path in weights
                ],
                "total_weight_bytes": sum(path.stat().st_size for path in weights),
                "revision_status": "unverified_local_copy",
                "content_hash_status": "weights_not_hashed",
            }
        )
    write_json(output / "models.json", result)
    return result


def validate_legacy_module(source: str) -> None:
    allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.FunctionDef)
    for statement in ast.parse(source).body:
        if not isinstance(statement, allowed):
            raise ValueError(f"legacy source has unexpected top-level statement: {type(statement)}")
        if isinstance(statement, ast.Import):
            modules = [alias.name.split(".")[0] for alias in statement.names]
            if not set(modules) <= ALLOWED_LEGACY_IMPORTS:
                raise ValueError("legacy source imports an unapproved module")
        if isinstance(statement, ast.ImportFrom):
            if statement.module not in ALLOWED_LEGACY_IMPORTS:
                raise ValueError("legacy source imports from an unapproved module")


def workflow_audit(yaml_path: Path, log_path: Path, output: Path) -> dict[str, object]:
    workflow = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["workflow"]
    graph = workflow["graph"]
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}
    edges = graph["edges"]
    adjacency: dict[str, list[dict[str, str]]] = {}
    for edge in edges:
        adjacency.setdefault(edge["source"], []).append(
            {
                "target_id": edge["target"],
                "source_handle": str(edge.get("sourceHandle", "")),
                "target_type": node_by_id[edge["target"]]["data"]["type"],
            }
        )
    inventory: list[dict[str, object]] = []
    prompts: list[dict[str, object]] = []
    legacy_source: str | None = None
    for node in nodes:
        data = node["data"]
        node_id = node["id"]
        node_type = data["type"]
        inventory.append(
            {
                "id": node_id,
                "title": data.get("title", ""),
                "type": node_type,
                "model": data.get("model", {}).get("name"),
                "provider": data.get("model", {}).get("provider"),
                "dataset_ids": data.get("dataset_ids", []),
                "outgoing": adjacency.get(node_id, []),
                "migration_component": migration_component(node_type),
            }
        )
        if "prompt_template" in data:
            prompts.append({"node_id": node_id, "templates": data["prompt_template"]})
        if node_type == "code" and len(data.get("code", "")) > 5000:
            legacy_source = data["code"]
    if legacy_source is None:
        raise ValueError("expected legacy log code node not found")
    validate_legacy_module(legacy_source)
    legacy_file = output / "legacy_log.py"
    legacy_file.write_text(legacy_source, encoding="utf-8")
    legacy_main = runpy.run_path(str(legacy_file))["main"]
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        legacy_result = legacy_main(stream.read(), [], upf="", start_time="", end_time="")
    counts = legacy_result["category_counts"]
    if sum(counts.values()) != 32539:
        raise ValueError("legacy count no longer matches sampled input; inspect source/data")
    write_json(output / "workflow_nodes.json", inventory)
    write_json(output / "workflow_prompts.json", prompts)
    write_json(output / "workflow_edges.json", edges)
    write_json(
        output / "legacy_baseline.json",
        {
            "yaml_sha256": sha256_file(yaml_path),
            "legacy_code_sha256": hashlib.sha256(legacy_source.encode()).hexdigest(),
            "log_sha256": sha256_file(log_path),
            "category_counts": counts,
            "total_classified_records": sum(counts.values()),
            "known_bad_labels": [
                "PFCP heartbeat request classified as unreachable",
                "license alarm clear classified as raised",
                "masked FAR handover not matched",
            ],
            "interpretation": (
                "Historical output only; not ground truth for the new event classifier"
            ),
        },
    )
    summary = {
        "node_counts": dict(Counter(node["data"]["type"] for node in nodes)),
        "edge_count": len(edges),
        "llm_models": dict(
            Counter(
                node["data"].get("model", {}).get("name")
                for node in nodes
                if node["data"]["type"] == "llm"
            )
        ),
        "knowledge_dataset_ids": sorted(
            {dataset for node in nodes for dataset in node["data"].get("dataset_ids", [])}
        ),
        "conversation_variables": [item["name"] for item in workflow["conversation_variables"]],
    }
    write_json(output / "workflow_summary.json", summary)
    return summary


def migration_component(node_type: str) -> str:
    mapping = {
        "start": "TaskSpec and CLI",
        "question-classifier": "task router",
        "llm": "local generation adapter",
        "answer": "AnswerReport renderer",
        "knowledge-retrieval": "local retrieval",
        "code": "deterministic Python tools",
        "iteration": "bounded agent state loop",
        "iteration-start": "bounded agent state loop",
        "assigner": "request-scoped evidence store",
        "parameter-extractor": "typed parameter extraction",
        "document-extractor": "ingestion/parser",
    }
    return mapping[node_type]


def evaluation_slots(output: Path) -> None:
    categories = [
        ("command", 40),
        ("product", 20),
        ("manual_troubleshooting", 20),
        ("log", 20),
        ("topology_synthetic", 10),
        ("missing_or_conflict", 10),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for category, count in categories:
            for number in range(1, count + 1):
                item = {
                    "id": f"{category}-{number:03d}",
                    "category": category,
                    "split": "proposed_dev" if number % 2 else "proposed_heldout",
                    "question": None,
                    "gold_evidence_ids": [],
                    "gold_facts": [],
                    "forbidden_claims": [],
                    "label_status": "pending_human_review",
                    "synthetic": category == "topology_synthetic",
                    "source_group": None,
                }
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def hardware_snapshot(output: Path) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    write_json(
        output / "hardware.json",
        {
            "captured_utc": datetime.now(UTC).isoformat(),
            "platform": platform.platform(),
            "cpu_count_logical": __import__("os").cpu_count(),
            "gpu_query_exit_code": result.returncode,
            "gpu_csv": result.stdout.strip().splitlines(),
            "gpu_query_error": result.stderr.strip() if result.returncode else None,
            "resource_note": "Snapshot only; GPU free memory may change without notice",
        },
    )


def run_p0(project_root: Path) -> dict[str, object]:
    output = project_root / "data" / "p0"
    output.mkdir(parents=True, exist_ok=True)
    sources = sources_audit(project_root / "config" / "corpus.yaml", output)
    models = models_audit(project_root / "config" / "models.yaml", output)
    source_root = Path(
        yaml.safe_load((project_root / "config" / "corpus.yaml").read_text())["source_root"]
    )
    workflow = workflow_audit(source_root / "upftest.yml", source_root / "upfd_masked.log", output)
    evaluation_slots(project_root / "eval" / "slots.jsonl")
    hardware_snapshot(output)
    return {"sources": len(sources), "models": len(models), **workflow}
