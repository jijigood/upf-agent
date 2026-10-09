"""Create an isolated v8 root and rebuild a generation using verified P3 vectors."""

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import yaml
from qdrant_client import QdrantClient

from upf_agent.p3_chunks import build_chunks
from upf_agent.p3_index import _config, _publish, load_active_index, load_chunks
from upf_agent.v8.models import digest

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "runtime/v8"
ORIGINAL = Path("/home/huangxiaolin/upf-agent")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    fingerprints = {}
    for folder in ("src", "tests", "config"):
        for path in sorted((ORIGINAL / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                fingerprints[str(path.relative_to(ORIGINAL))] = sha(path)
    for path in sorted((ORIGINAL / "data/raw").iterdir()):
        if path.is_file():
            fingerprints[str(path.relative_to(ORIGINAL))] = sha(path)
    write(ROOT / "evidence/u0/original_fingerprints.json", fingerprints)
    for name in ("data/raw", "evidence/p1", "data/p2", "data/p4/fixtures", "data/p4/templates"):
        source, target = PROJECT / name, ROOT / name
        if source.exists():
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.sqlite3", "*-wal", "*-shm"),
            )
    target_db = ROOT / "data/p2/corpus.sqlite3"
    with sqlite3.connect(f"file:{PROJECT / 'data/p2/corpus.sqlite3'}?mode=ro", uri=True) as source:
        with sqlite3.connect(target_db) as target:
            source.backup(target)
    shutil.copytree(PROJECT / "config", ROOT / "config", dirs_exist_ok=True)
    for name, key, value in (
        ("corpus.yaml", "source_root", str(ROOT / "data/raw")),
        ("retrieval.yaml", "dictionary_path", str(ROOT / "config/telecom_dictionary.txt")),
    ):
        path = ROOT / "config" / name
        config = yaml.safe_load(path.read_text())
        config[key] = value
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    for name in ("tools", ".venv-retrieval"):
        link = ROOT / name
        if not link.exists():
            link.symlink_to(PROJECT / name, target_is_directory=True)
    # A path move changes the P3 identity. Verify the exact chunk/config/model equivalence,
    # then construct a new generation; never edit the archived baseline manifest.
    old_dir, old = load_active_index(PROJECT)
    config, fp = _config(ROOT)
    old_config, old_fp = _config(PROJECT)
    for key in fp:
        if key != "config_sha256" and fp[key] != old_fp[key]:
            raise ValueError(f"vector reuse rejected: {key}")
    if {k: v for k, v in config.items() if k != "dictionary_path"} != {
        k: v for k, v in old_config.items() if k != "dictionary_path"
    }:
        raise ValueError("retrieval semantics changed")
    p2, chunks = build_chunks(
        ROOT, Path(config["tokenizer_path"]), config["target_tokens"], config["max_tokens"]
    )
    previous = load_chunks(old_dir)

    def normalized(chunk):
        value = chunk.model_dump()
        value["interface_tags"] = sorted(value["interface_tags"])
        return value

    if {c.chunk_id: normalized(c) for c in chunks} != {
        k: normalized(c) for k, c in previous.items()
    }:
        raise ValueError("chunk mismatch: cannot reuse embeddings")
    identity = {"p2_generation": p2, **fp}
    generation = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    final = ROOT / "data/p3/indexes" / generation
    if not final.exists():
        shutil.copytree(old_dir, final)
        (final / "chunks.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in chunks))
        manifest = {
            **old,
            "index_generation": generation,
            "identity": identity,
            "chunks_sha256": sha(final / "chunks.jsonl"),
            "vector_reuse": {
                "from": old["index_generation"],
                "source_manifest_sha256": sha(old_dir / "manifest.json"),
                "verified_chunks": len(chunks),
            },
        }
        write(final / "manifest.json", manifest)
    _publish(final.parent, generation)
    export = ROOT / "data/v8/p3_vectors.jsonl"
    export.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    qdrant = QdrantClient(path=str(final / "qdrant"))
    with export.open("w") as out:
        offset = None
        while True:
            points, offset = qdrant.scroll(
                collection_name=config["collection"], limit=100, offset=offset, with_vectors=True
            )
            for point in points:
                out.write(
                    json.dumps(
                        {"id": str(point.id), "payload": point.payload, "vector": point.vector}
                    )
                    + "\n"
                )
                count += 1
            if offset is None:
                break
    qdrant.close()
    if count != len(chunks):
        raise ValueError("vector export incomplete")
    from upf_agent.log_tools import ingest_log
    from upf_agent.topology import import_topology

    logs = ingest_log(ROOT / "data/raw/upfd_masked.log", ROOT / "data/p4/logs")
    topologies = []
    for old_manifest in sorted((PROJECT / "data/p4/topology").glob("*/manifest.json")):
        manifest = json.loads(old_manifest.read_text())
        source = Path(manifest["source_path"])
        if source.is_relative_to(PROJECT):
            source = ROOT / source.relative_to(PROJECT)
        elif source.is_relative_to(ORIGINAL):
            source = ROOT / source.relative_to(ORIGINAL)
        if source.is_file():
            topologies.append(import_topology(source, ROOT / "data/p4/topology", synthetic=True))
    import upf_agent

    receipt = {
        "root": str(ROOT),
        "import": upf_agent.__file__,
        "p3_generation": generation,
        "log_generation": logs["generation"],
        "topology": topologies,
        "raw_files": len(list((ROOT / "data/raw").iterdir())),
        "vectors": count,
        "vector_export_sha256": sha(export),
        "identity": digest(identity),
    }
    if not str(upf_agent.__file__).startswith(str(PROJECT / "src")):
        raise ValueError("editable installation points outside the v8 copy")
    write(ROOT / "evidence/u0/isolation.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare()
