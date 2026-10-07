"""Atomic local Qdrant/BM25/command index generations over P2 manual chunks."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import yaml
from qdrant_client import QdrantClient, models

from upf_agent.config import load_model_config
from upf_agent.contracts import CommandRecord
from upf_agent.corpus_db import active_id, connect
from upf_agent.p3_chunks import CHUNKER_VERSION, RetrievalChunk, build_chunks
from upf_agent.p3_lexical import build_bm25, build_exact
from upf_agent.retrieval import RetrievalWorker


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_fingerprint(root: Path, profile: str) -> str:
    snapshot = json.loads((root / "evidence/p1/model-hashes.json").read_text(encoding="utf-8"))
    model = next(item for item in snapshot["models"] if item["profile"] == profile)
    files = sorted((item["name"], item["sha256"]) for item in model["files"])
    return hashlib.sha256(json.dumps(files).encode()).hexdigest()


def _config(root: Path) -> tuple[dict, dict]:
    config_path = root / "config/retrieval.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tokenizer = Path(config["tokenizer_path"])
    dictionary = Path(config["dictionary_path"])
    if not tokenizer.is_file() or not dictionary.is_file():
        raise FileNotFoundError("Local tokenizer or telecom dictionary is missing")
    models_config = load_model_config(root / "config/models.yaml")
    fingerprint = {
        "config_sha256": _sha(config_path),
        "dictionary_sha256": _sha(dictionary),
        "tokenizer_sha256": _sha(tokenizer),
        "embedding_id": models_config.embedding.id,
        "embedding_fingerprint": _model_fingerprint(root, "embedding"),
        "reranker_id": models_config.reranker.id,
        "reranker_fingerprint": _model_fingerprint(root, "reranker"),
        "embedding_dimension": models_config.embedding.dimension,
        "embedding_prompt_sha256": _sha(
            models_config.embedding.path / "config_sentence_transformers.json"
        ),
        "chunker_version": CHUNKER_VERSION,
    }
    return config, fingerprint


def _commands(root: Path, generation: str) -> list[CommandRecord]:
    db = connect(root / "data/p2/corpus.sqlite3")
    try:
        return [
            CommandRecord.model_validate_json(raw)
            for (raw,) in db.execute(
                "SELECT payload FROM commands WHERE generation=?", (generation,)
            )
        ]
    finally:
        db.close()


def load_active_index(root: Path) -> tuple[Path, dict]:
    base = root / "data/p3/indexes"
    pointer = base / "active.json"
    if not pointer.is_file():
        raise FileNotFoundError("P3 index is not built")
    active = json.loads(pointer.read_text(encoding="utf-8"))
    path = base / active["index_generation"]
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["index_generation"] != active["index_generation"]:
        raise ValueError("P3 index pointer/manifest mismatch")
    return path, manifest


def load_chunks(path: Path) -> dict[str, RetrievalChunk]:
    with (path / "chunks.jsonl").open(encoding="utf-8") as stream:
        chunks = [RetrievalChunk.model_validate_json(line) for line in stream if line.strip()]
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("duplicate chunks in stored index")
    return by_id


def build_index(root: Path, device: str = "cuda:1") -> dict:
    config, fingerprint = _config(root)
    p2_db = connect(root / "data/p2/corpus.sqlite3")
    try:
        p2_generation = active_id(p2_db)
    finally:
        p2_db.close()
    if p2_generation is None:
        raise ValueError("P2 corpus has no active generation")
    identity = {"p2_generation": p2_generation, **fingerprint}
    index_generation = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :20
    ]
    base = root / "data/p3/indexes"
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        final_dir = base / index_generation
        if (final_dir / "manifest.json").is_file():
            manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
            if manifest["identity"] == identity:
                _publish(base, index_generation)
                return {
                    "index_generation": index_generation,
                    "reused": True,
                    "stats": manifest["stats"],
                }
        stage = base / f".staging-{index_generation}-{os.getpid()}"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        try:
            parsed_generation, chunks = build_chunks(
                root, Path(config["tokenizer_path"]), config["target_tokens"], config["max_tokens"]
            )
            if parsed_generation != p2_generation or not chunks:
                raise ValueError("P2 generation changed or no chunks were produced")
            with (stage / "chunks.jsonl").open("w", encoding="utf-8") as output:
                for chunk in chunks:
                    output.write(chunk.model_dump_json() + "\n")
            bm25_stats = build_bm25(chunks, Path(config["dictionary_path"]), stage / "bm25")
            exact_stats = build_exact(_commands(root, p2_generation), chunks, stage / "exact.json")
            qdrant_dir = stage / "qdrant"
            qdrant = QdrantClient(path=str(qdrant_dir))
            try:
                dimension = int(fingerprint["embedding_dimension"])
                collection = config["collection"]
                qdrant.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    ),
                )
                model_config = load_model_config(root / "config/models.yaml")
                batch_size = int(config["embedding_batch_size"])
                with RetrievalWorker(
                    root, "embed", model_config.embedding.path, device=device
                ) as worker:
                    for start in range(0, len(chunks), batch_size):
                        batch = chunks[start : start + batch_size]
                        result = worker.request(
                            {"role": "document", "texts": [c.text for c in batch]}, timeout=300
                        )
                        if result["dimension"] != dimension or not result["normalized"]:
                            raise ValueError("embedding dimension or normalization mismatch")
                        points = [
                            models.PointStruct(
                                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                                vector=vector,
                                payload={
                                    "chunk_id": chunk.chunk_id,
                                    "source_id": chunk.source_id,
                                    "product": chunk.product,
                                    "version": chunk.version,
                                    "role": chunk.role,
                                    "kind": chunk.kind,
                                },
                            )
                            for chunk, vector in zip(batch, result["vectors"], strict=True)
                        ]
                        qdrant.upsert(collection_name=collection, points=points, wait=True)
                count = qdrant.count(collection_name=collection, exact=True).count
                vector_info = qdrant.get_collection(collection_name=collection)
                size = vector_info.config.params.vectors.size
                if count != len(chunks) or size != dimension:
                    raise ValueError(f"Qdrant count/dimension mismatch: {count}/{size}")
            finally:
                qdrant.close()
            stats = {
                "chunks": len(chunks),
                "by_kind": dict(Counter(c.kind for c in chunks)),
                "max_tokens": max(c.token_count for c in chunks),
                "sensitive_chunks_redacted": sum(
                    "sensitive_token_redacted" in c.quality_flags for c in chunks
                ),
                "bm25": bm25_stats,
                "exact": exact_stats,
                "qdrant_points": count,
                "dimension": dimension,
            }
            manifest = {
                "index_generation": index_generation,
                "identity": identity,
                "p2_generation": p2_generation,
                "collection": config["collection"],
                "embedding_device_used_for_build": device,
                "query_instruction": json.loads(
                    (model_config.embedding.path / "config_sentence_transformers.json").read_text()
                )["prompts"]["query"],
                "normalized_vectors": True,
                "stats": stats,
                "chunks_sha256": _sha(stage / "chunks.jsonl"),
            }
            (stage / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            p2_db = connect(root / "data/p2/corpus.sqlite3")
            try:
                if active_id(p2_db) != p2_generation:
                    raise ValueError("P2 generation changed during index build")
            finally:
                p2_db.close()
            os.replace(stage, final_dir)
            _publish(base, index_generation)
            return {"index_generation": index_generation, "reused": False, "stats": stats}
        except BaseException:
            if stage.exists():
                shutil.rmtree(stage)
            raise


def _publish(base: Path, generation: str) -> None:
    temporary = base / f".active-{os.getpid()}.json"
    temporary.write_text(json.dumps({"index_generation": generation}) + "\n", encoding="utf-8")
    os.replace(temporary, base / "active.json")


def dense_search(
    index_dir: Path,
    collection: str,
    vector: list[float],
    product: str | None,
    version: str | None,
    limit: int,
) -> list[tuple[str, float]]:
    conditions = [models.FieldCondition(key="role", match=models.MatchValue(value="manual"))]
    if product:
        conditions.append(
            models.FieldCondition(key="product", match=models.MatchValue(value=product))
        )
    if version:
        conditions.append(
            models.FieldCondition(key="version", match=models.MatchAny(any=[version, "unknown"]))
        )
    client = QdrantClient(path=str(index_dir / "qdrant"))
    try:
        points = client.query_points(
            collection_name=collection,
            query=vector,
            query_filter=models.Filter(must=conditions),
            with_payload=True,
            limit=limit,
        ).points
        return [(str(point.payload["chunk_id"]), float(point.score)) for point in points]
    finally:
        client.close()
