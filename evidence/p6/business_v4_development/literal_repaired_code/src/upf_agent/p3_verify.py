"""Validate P3 index consistency without loading large inference models."""

import hashlib
import json
from pathlib import Path

from qdrant_client import QdrantClient

from upf_agent.corpus_db import active_id, connect
from upf_agent.p3_index import _config, dense_search, load_active_index, load_chunks
from upf_agent.p3_lexical import TechnicalTokenizer, scope_allows, search_bm25, search_exact


def verify_index(root: Path, output: Path) -> dict:
    index_dir, manifest = load_active_index(root)
    config, fingerprint = _config(root)
    db = connect(root / "data/p2/corpus.sqlite3")
    try:
        p2_generation = active_id(db)
        sources = {
            sid: sha
            for sid, sha in db.execute(
                "SELECT source_id,sha256 FROM sources WHERE generation=? AND role='manual'",
                (p2_generation,),
            )
        }
    finally:
        db.close()
    if p2_generation != manifest["p2_generation"]:
        raise ValueError("P3 index refers to a different P2 generation")
    if any(manifest["identity"].get(key) != value for key, value in fingerprint.items()):
        raise ValueError("P3 index config/model fingerprint mismatch")
    chunks_file = index_dir / "chunks.jsonl"
    if hashlib.sha256(chunks_file.read_bytes()).hexdigest() != manifest["chunks_sha256"]:
        raise ValueError("Stored chunk file SHA mismatch")
    chunks = load_chunks(index_dir)
    if len(chunks) != manifest["stats"]["chunks"]:
        raise ValueError("Chunk count differs from manifest")
    for chunk in chunks.values():
        if chunk.source_sha256 != sources.get(chunk.source_id) or chunk.role != "manual":
            raise ValueError(f"Non-manual or stale source chunk: {chunk.chunk_id}")
        if chunk.token_count > config["max_tokens"]:
            raise ValueError(f"Oversized chunk: {chunk.chunk_id}")
    qdrant = QdrantClient(path=str(index_dir / "qdrant"))
    try:
        count = qdrant.count(manifest["collection"], exact=True).count
        dimension = qdrant.get_collection(manifest["collection"]).config.params.vectors.size
    finally:
        qdrant.close()
    if count != len(chunks) or dimension != fingerprint["embedding_dimension"]:
        raise ValueError("Qdrant point count or dimension mismatch")
    tokenizer = TechnicalTokenizer(Path(config["dictionary_path"]))
    bm25 = search_bm25(index_dir / "bm25", chunks, tokenizer, "show cp-node", "UPF", "5.0", 20)
    exact = search_exact(index_dir / "exact.json", "show cp-node", "UPF", "5.0", 20)
    dense = dense_search(
        index_dir, manifest["collection"], [1.0] + [0.0] * (dimension - 1), "UPF", "5.0", 20
    )
    for name, hits in (("bm25", bm25), ("exact", exact), ("dense", dense)):
        if not hits or any(
            not scope_allows(chunks[chunk_id], "UPF", "5.0") for chunk_id, _ in hits
        ):
            raise ValueError(f"{name} scoped search failed")
    report = {
        "index_generation": manifest["index_generation"],
        "p2_generation": p2_generation,
        "chunks": len(chunks),
        "qdrant_points": count,
        "dimension": dimension,
        "scoped_route_hits": {"bm25": len(bm25), "exact": len(exact), "dense": len(dense)},
        "redacted_chunks": sum(
            "sensitive_token_redacted" in c.quality_flags for c in chunks.values()
        ),
        "pass": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
