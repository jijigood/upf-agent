"""Controlled retrieval ablations; weak probes never become reviewed gold metrics."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from upf_agent.config import load_model_config
from upf_agent.p3_index import _config, dense_search, load_active_index, load_chunks
from upf_agent.p3_lexical import TechnicalTokenizer, search_bm25, search_exact
from upf_agent.p3_search import _candidate_budget, fuse_rrf
from upf_agent.retrieval import RetrievalWorker


def ranking_metrics(ranking: list[str], relevant: dict[str, int]) -> dict:
    positive = {cid for cid, grade in relevant.items() if grade > 0}
    if not positive:
        raise ValueError("retrieval metrics need at least one relevant item")
    ranked = list(dict.fromkeys(ranking))
    hit = positive.intersection(ranked[:20])
    dcg = sum(
        (2 ** relevant.get(cid, 0) - 1) / math.log2(i + 2) for i, cid in enumerate(ranked[:8])
    )
    ideal = sum(
        (2**grade - 1) / math.log2(i + 2)
        for i, grade in enumerate(sorted(relevant.values(), reverse=True)[:8])
    )
    return {
        "recall_at_20": len(hit) / len(positive),
        "hit_at_20": float(bool(hit)),
        "ndcg_at_8": dcg / ideal,
        "hit_at_3": float(bool(positive.intersection(ranked[:3]))),
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def run_ablation(
    root: Path,
    dataset: Path,
    output: Path,
    device: str = "cuda:1",
    bge_path: Path = Path("/home/huangxiaolin/models/BGE-M3"),
) -> dict:
    started = perf_counter()
    directory, manifest = load_active_index(root)
    config, fingerprint = _config(root)
    if any(manifest["identity"].get(k) != v for k, v in fingerprint.items()):
        raise ValueError("index config is stale")
    chunks = load_chunks(directory)
    cases = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("dataset is empty or has duplicate IDs")
    for case in cases:
        if case["index_generation"] != manifest["index_generation"]:
            raise ValueError("probe index generation mismatch")
        if not case["relevance"] or not set(case["relevance"]).issubset(chunks):
            raise ValueError("unknown or empty relevance IDs")
        if case["split"] == "heldout":
            raise ValueError("P3 development ablation must not consume heldout evaluation")
    # Fingerprint the actual local BGE files used by SentenceTransformers, not only its name.
    names = [
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "sentence_bert_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "1_Pooling/config.json",
        "model.safetensors" if (bge_path / "model.safetensors").exists() else "pytorch_model.bin",
    ]
    bge_hashes = {name: _file_hash(bge_path / name) for name in names}
    bge_identity = hashlib.sha256(json.dumps(bge_hashes, sort_keys=True).encode()).hexdigest()
    bge_dir = root / "data/p3/ablation" / f"{manifest['index_generation']}-{bge_identity[:12]}"
    bge_dir.mkdir(parents=True, exist_ok=True)
    collection = "bge_m3_1024"
    marker = bge_dir / "complete.json"
    timings = {}
    t = perf_counter()
    with RetrievalWorker(root, "embed", bge_path, device=device) as worker:
        timings["bge_load_ms"] = round((perf_counter() - t) * 1000)
        t = perf_counter()
        if not marker.exists():
            client = QdrantClient(path=str(bge_dir / "qdrant"))
            try:
                if client.collection_exists(collection):
                    client.delete_collection(collection)
                client.create_collection(
                    collection,
                    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
                )
                ordered = list(chunks.values())
                for start in range(0, len(ordered), 8):
                    batch = ordered[start : start + 8]
                    response = worker.request(
                        {"role": "document", "texts": [c.text for c in batch]}
                    )
                    if response["dimension"] != 1024 or not response["normalized"]:
                        raise ValueError("BGE dimension/normalization mismatch")
                    client.upsert(
                        collection,
                        points=[
                            models.PointStruct(
                                id=str(uuid5(NAMESPACE_URL, c.chunk_id)),
                                vector=v,
                                payload={
                                    "chunk_id": c.chunk_id,
                                    "product": c.product,
                                    "version": c.version,
                                    "role": c.role,
                                },
                            )
                            for c, v in zip(batch, response["vectors"], strict=True)
                        ],
                    )
                if client.count(collection, exact=True).count != len(chunks):
                    raise ValueError("incomplete BGE collection")
            finally:
                client.close()
            temporary = bge_dir / "complete.tmp"
            temporary.write_text(
                json.dumps(
                    {
                        "files": bge_hashes,
                        "chunks_sha256": manifest["chunks_sha256"],
                        "dimension": 1024,
                    }
                )
            )
            os.replace(temporary, marker)
        timings["bge_index_build_or_reuse_ms"] = round((perf_counter() - t) * 1000)
        bge_vectors = []
        t = perf_counter()
        for start in range(0, len(cases), 8):
            bge_vectors.extend(
                worker.request(
                    {"role": "query", "texts": [c["question"] for c in cases[start : start + 8]]}
                )["vectors"]
            )
        timings["bge_query_embedding_ms"] = round((perf_counter() - t) * 1000)
    mc = load_model_config(root / "config/models.yaml")
    qwen_vectors = []
    t = perf_counter()
    with RetrievalWorker(root, "embed", mc.embedding.path, device=device) as worker:
        timings["qwen_load_ms"] = round((perf_counter() - t) * 1000)
        t = perf_counter()
        for start in range(0, len(cases), 8):
            qwen_vectors.extend(
                worker.request(
                    {"role": "query", "texts": [c["question"] for c in cases[start : start + 8]]}
                )["vectors"]
            )
        timings["qwen_query_embedding_ms"] = round((perf_counter() - t) * 1000)
    tokenizer = TechnicalTokenizer(Path(config["dictionary_path"]))
    rows = []
    for i, case in enumerate(cases):
        scope = case["product"], case["version"]
        bge = dense_search(bge_dir, collection, bge_vectors[i], *scope, 20)
        dense = dense_search(directory, manifest["collection"], qwen_vectors[i], *scope, 20)
        bm25 = search_bm25(directory / "bm25", chunks, tokenizer, case["question"], *scope, 20)
        exact = search_exact(directory / "exact.json", case["question"], *scope, 20)
        hybrid = fuse_rrf({"dense": dense, "bm25": bm25}, int(config["rrf_k"]))
        all_routes = fuse_rrf({"dense": dense, "bm25": bm25, "exact": exact}, int(config["rrf_k"]))
        candidates = _candidate_budget(all_routes, int(config["rerank_candidates"]))
        candidates.sort(key=lambda c: (-c["rrf_score"], c["chunk_id"]))
        rows.append(
            {
                "case": case,
                "rankings": {
                    "bge_dense": [cid for cid, _ in bge],
                    "qwen_dense": [cid for cid, _ in dense],
                    "qwen_bm25": [c["chunk_id"] for c in hybrid][:32],
                    "qwen_bm25_exact": [c["chunk_id"] for c in candidates],
                },
                "route_counts": {
                    "bge": len(bge),
                    "dense": len(dense),
                    "bm25": len(bm25),
                    "exact": len(exact),
                },
                "candidates": candidates,
            }
        )
    t = perf_counter()
    with RetrievalWorker(root, "rerank", mc.reranker.path, device=device) as worker:
        timings["reranker_load_ms"] = round((perf_counter() - t) * 1000)
        t = perf_counter()
        for row in rows:
            candidates = row["candidates"]
            response = worker.request(
                {
                    "pairs": [
                        [row["case"]["question"], chunks[c["chunk_id"]].text] for c in candidates
                    ]
                }
            )
            ranked = sorted(
                zip(candidates, response["raw_logit_difference"], strict=True),
                key=lambda x: (-x[1], -x[0]["rrf_score"], x[0]["chunk_id"]),
            )
            # Match search-index's final-eight exact-command retention rule.
            exact_best = next((pair for pair in ranked if "exact" in pair[0]["routes"]), None)
            final_k = int(config["final_evidence"])
            if exact_best and exact_best not in ranked[:final_k]:
                ranked.remove(exact_best)
                ranked.insert(final_k - 1, exact_best)
            row["rankings"]["qwen_bm25_exact_rerank"] = [c["chunk_id"] for c, _ in ranked]
            row["rerank_scores"] = {c["chunk_id"]: score for c, score in ranked}
        timings["rerank_all_cases_ms"] = round((perf_counter() - t) * 1000)
    reviewed = all(c["label_status"] == "reviewed" for c in cases)
    metric_name = "reviewed_metrics" if reviewed else "unreviewed_target_probe_metrics"
    variants = list(rows[0]["rankings"])
    summary = {}
    for variant in variants:
        values = [
            ranking_metrics(row["rankings"][variant], row["case"]["relevance"]) for row in rows
        ]
        summary[variant] = {key: sum(v[key] for v in values) / len(values) for key in values[0]}
    for row in rows:
        row[metric_name] = {
            variant: ranking_metrics(ranking, row["case"]["relevance"])
            for variant, ranking in row["rankings"].items()
        }
    report = {
        "index_generation": manifest["index_generation"],
        "dataset_sha256": _file_hash(dataset),
        "dataset": str(dataset),
        "cases": len(cases),
        "label_status": "reviewed" if reviewed else "pending_human_review",
        "business_accuracy_claim": False,
        "metric_scope": metric_name,
        "warning": (
            "源章节派生的开发探针存在措辞与目标选择偏差；"
            "未人工复核，不是正式Recall/准确率或模型胜负结论。"
        )
        if not reviewed
        else "仅代表此开发集。",
        "models": {"bge_files": bge_hashes, "qwen": fingerprint},
        "budget": {"route_top_k": 20, "rerank_candidates": 32, "ndcg_k": 8},
        metric_name: summary,
        "timings_ms": timings,
        "rows": rows,
        "elapsed_ms": round((perf_counter() - started) * 1000),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {k: v for k, v in report.items() if k not in {"rows", "models"}}
