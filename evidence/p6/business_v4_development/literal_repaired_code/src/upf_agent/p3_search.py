"""Scoped retrieval, native chapter anchors, rank fusion, and bounded context."""

from __future__ import annotations

import re
from contextlib import nullcontext
from pathlib import Path

from tokenizers import Tokenizer

from upf_agent.config import load_model_config
from upf_agent.corpus_db import active_id, connect
from upf_agent.p3_chunks import RetrievalChunk
from upf_agent.p3_index import _config, dense_search, load_active_index, load_chunks
from upf_agent.p3_lexical import TechnicalTokenizer, scope_allows, search_bm25, search_exact
from upf_agent.p3_sections import related_section_routes, resolve_section, section_routes
from upf_agent.retrieval import RetrievalWorker

RETRIEVAL_PIPELINE_VERSION = "p3-retrieval-v3-section-rank-fusion"
MODEL_RANK_WEIGHT = 0.5


def combine_rerank(candidates: list[dict], k: int) -> list[dict]:
    """Add half a model-ranking route while retaining independent retrieval support."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    by_model = sorted(
        candidates,
        key=lambda item: (
            -item["rerank_raw_logit_difference"],
            -item["rrf_score"],
            item["chunk_id"],
        ),
    )
    for rank, item in enumerate(by_model, 1):
        item["model_rerank_rank"] = rank
        item["combined_score"] = item["rrf_score"] + MODEL_RANK_WEIGHT / (k + rank)
    return sorted(
        candidates,
        key=lambda item: (-item["combined_score"], -item["rrf_score"], item["chunk_id"]),
    )


def fuse_rrf(routes: dict[str, list[tuple[str, float]]], k: int = 60) -> list[dict]:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    ranked: dict[str, dict] = {}
    for route, hits in routes.items():
        for rank, (chunk_id, raw_score) in enumerate(hits, 1):
            item = ranked.setdefault(
                chunk_id, {"chunk_id": chunk_id, "rrf_score": 0.0, "routes": {}}
            )
            if route in item["routes"]:
                continue
            item["rrf_score"] += 1.0 / (k + rank)
            item["routes"][route] = {"rank": rank, "raw_score": raw_score}
    return sorted(ranked.values(), key=lambda item: (-item["rrf_score"], item["chunk_id"]))


def _candidate_budget(fused: list[dict], rerank_limit: int) -> list[dict]:
    exact = [item for item in fused if "exact" in item["routes"]]
    selected = exact[:rerank_limit]
    seen = {item["chunk_id"] for item in selected}
    for item in fused:
        if len(selected) >= rerank_limit:
            break
        if item["chunk_id"] not in seen:
            selected.append(item)
            seen.add(item["chunk_id"])
    return selected


def _trim_tokens(text: str, tokenizer: Tokenizer, limit: int) -> tuple[str, int, bool]:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    count = len(encoding.ids)
    if count <= limit:
        return text, count, False
    end = encoding.offsets[limit - 1][1]
    return text[:end], limit, True


def expand_context(
    selected: list[dict],
    chunks: dict[str, RetrievalChunk],
    tokenizer: Tokenizer,
    total_budget: int = 5000,
    neighbor_budget: int = 240,
) -> tuple[list[dict], int]:
    by_parent: dict[str, list[RetrievalChunk]] = {}
    for chunk in chunks.values():
        by_parent.setdefault(chunk.parent_id, []).append(chunk)
    for group in by_parent.values():
        group.sort(key=lambda c: (c.ordinal_start, c.chunk_id))
    used, output, selected_ids = 0, [], {item["chunk_id"] for item in selected}
    used_neighbors: set[str] = set()
    for item in selected:
        chunk = chunks[item["chunk_id"]]
        if used + chunk.token_count > total_budget:
            break
        used += chunk.token_count
        evidence = {
            **item,
            "source_id": chunk.source_id,
            "source_sha256": chunk.source_sha256,
            "relative_path": chunk.relative_path,
            "product": chunk.product,
            "version": chunk.version,
            "section_path": chunk.section_path,
            "parent_id": chunk.parent_id,
            "kind": chunk.kind,
            "text": chunk.text,
            "block_ids": chunk.block_ids,
            "locators": chunk.locators,
            "command_id": chunk.command_id,
            "command_family": chunk.command_family,
            "interface_tags": chunk.interface_tags,
            "table_id": chunk.table_id,
            "table_cells": chunk.table_cells,
            "quality_flags": chunk.quality_flags,
            "applicability_note": "版本未知，需人工核对适用性"
            if chunk.version in {None, "unknown"}
            else None,
            "neighbors": [],
        }
        output.append(evidence)
    for evidence in output:
        chunk = chunks[evidence["chunk_id"]]
        group = by_parent[chunk.parent_id]
        position = next(i for i, value in enumerate(group) if value.chunk_id == chunk.chunk_id)
        for neighbor in [
            *group[max(0, position - 1) : position],
            *group[position + 1 : position + 2],
        ]:
            if neighbor.chunk_id in selected_ids | used_neighbors or used >= total_budget:
                continue
            allowance = min(neighbor_budget, total_budget - used)
            if allowance <= 0:
                break
            excerpt, cost, truncated = _trim_tokens(neighbor.text, tokenizer, allowance)
            if not excerpt:
                continue
            used += cost
            used_neighbors.add(neighbor.chunk_id)
            evidence["neighbors"].append(
                {
                    "chunk_id": neighbor.chunk_id,
                    "text": excerpt,
                    "truncated": truncated,
                    "block_ids": neighbor.block_ids,
                    "locators": neighbor.locators,
                    "source_id": neighbor.source_id,
                }
            )
    return output, used


def _inferred_scope(
    query: str, product: str | None, version: str | None
) -> tuple[str | None, str | None]:
    if product is None:
        if re.search(r"网管|管理平台", query):
            product = "5GC网管"
        elif re.search(r"\bUPF\b|upfmgr|upfd|N[346]|PFCP|GTP-U", query, re.I):
            product = "UPF"
    if version is None:
        match = re.search(r"(?<![\d.])(?:UPF\s*)?(?:v|V)?(5\.0|1\.3)(?![\d.])", query)
        if match:
            version = "V1.3" if match.group(1) == "1.3" else "5.0"
    return product, version


def search(
    root: Path,
    query: str,
    product: str | None = None,
    version: str | None = None,
    device: str = "cuda:1",
    workers: tuple[RetrievalWorker, RetrievalWorker] | None = None,
) -> dict:
    if not query.strip() or len(query) > 2000:
        raise ValueError("query must contain 1–2000 characters")
    index_dir, manifest = load_active_index(root)
    config, fingerprint = _config(root)
    db = connect(root / "data/p2/corpus.sqlite3")
    try:
        current_p2 = active_id(db)
    finally:
        db.close()
    if manifest["p2_generation"] != current_p2 or any(
        manifest["identity"].get(key) != value for key, value in fingerprint.items()
    ):
        raise ValueError("P3 index is stale: source, model, dictionary, or tokenizer changed")
    chunks = load_chunks(index_dir)
    product, version = _inferred_scope(query, product, version)
    tokenizer = TechnicalTokenizer(Path(config["dictionary_path"]))
    exact = search_exact(
        index_dir / "exact.json", query, product, version, int(config["exact_top_k"])
    )
    bm25 = search_bm25(
        index_dir / "bm25", chunks, tokenizer, query, product, version, int(config["bm25_top_k"])
    )
    models_config = load_model_config(root / "config/models.yaml")
    embedding_context = (
        nullcontext(workers[0])
        if workers
        else RetrievalWorker(root, "embed", models_config.embedding.path, device=device)
    )
    with embedding_context as worker:
        embedding = worker.request({"role": "query", "texts": [query]})
    if embedding["dimension"] != manifest["stats"]["dimension"]:
        raise ValueError("query embedding dimension differs from index")
    dense = dense_search(
        index_dir,
        manifest["collection"],
        embedding["vectors"][0],
        product,
        version,
        int(config["dense_top_k"]),
    )
    routes = {"exact": exact, "bm25": bm25, "dense": dense}
    section_hint = resolve_section(query, chunks, product, version)
    routes = section_routes(
        routes, chunks, section_hint, int(config["rerank_candidates"]), product, version
    )
    related_sections = []
    if section_hint is None:
        routes, related_sections = related_section_routes(
            query, routes, chunks, product, version, int(config["rerank_candidates"])
        )
    for route, hits in routes.items():
        if any(not scope_allows(chunks[chunk_id], product, version) for chunk_id, _ in hits):
            raise ValueError(f"{route} returned a chunk outside requested scope")
    fused = fuse_rrf(routes, int(config["rrf_k"]))
    candidates = _candidate_budget(fused, int(config["rerank_candidates"]))
    if candidates:
        pairs = [[query, chunks[item["chunk_id"]].text] for item in candidates]
        scores = []
        reranking_context = (
            nullcontext(workers[1])
            if workers
            else RetrievalWorker(root, "rerank", models_config.reranker.path, device=device)
        )
        with reranking_context as worker:
            for start in range(0, len(pairs), 32):
                result = worker.request({"pairs": pairs[start : start + 32]}, timeout=300)
                scores.extend(zip(result["raw_logit_difference"], result["p_yes"], strict=True))
        for item, (raw, probability) in zip(candidates, scores, strict=True):
            item["rerank_raw_logit_difference"] = float(raw)
            item["rerank_p_yes"] = float(probability)
        candidates = combine_rerank(candidates, int(config["rrf_k"]))
    selected = candidates[: int(config["final_evidence"])]
    exact_best = next((item for item in candidates if "exact" in item["routes"]), None)
    if exact_best and exact_best not in selected:
        selected[-1:] = [exact_best]
    qwen_tokenizer = Tokenizer.from_file(str(config["tokenizer_path"]))
    evidence, context_tokens = expand_context(
        selected,
        chunks,
        qwen_tokenizer,
        int(config["context_token_budget"]),
        int(config["neighbor_token_budget"]),
    )
    return {
        "pipeline_version": RETRIEVAL_PIPELINE_VERSION,
        "rerank_policy": {
            "method": "retrieval_rrf_plus_model_rank",
            "model_rank_weight": MODEL_RANK_WEIGHT,
            "rrf_k": int(config["rrf_k"]),
        },
        "section_hint": (
            {
                "source_id": section_hint.source_id,
                "path": list(section_hint.path),
                "method": section_hint.method,
                "applied": "section" in routes,
            }
            if section_hint
            else None
        ),
        "related_sections": [
            {"source_id": hint.source_id, "path": list(hint.path), "method": hint.method}
            for hint in related_sections
        ],
        "query": query,
        "product": product,
        "version": version,
        "index_generation": manifest["index_generation"],
        "p2_generation": manifest["p2_generation"],
        "route_counts": {name: len(hits) for name, hits in routes.items()},
        "fused_candidates": len(fused),
        "reranked_candidates": len(candidates),
        "context_tokens": context_tokens,
        "context_token_budget": int(config["context_token_budget"]),
        "rankings": {
            **{name: [key for key, _ in hits] for name, hits in routes.items()},
            "rrf": [item["chunk_id"] for item in fused],
            "reranked": [item["chunk_id"] for item in candidates],
        },
        "evidence": evidence,
        "warnings": (["未找到检索证据"] if not evidence else []),
    }
