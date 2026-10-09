"""Official GraphRAG supplemental baseline, run only in .venv-graphrag."""

import json
import sys
from pathlib import Path

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import HybridCypherRetriever


def run(request_path: Path):
    request = json.loads(request_path.read_text())
    root = Path(request["root"])
    config = json.loads((root / "config/v8-neo4j.json").read_text())
    driver = GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"]))
    try:
        chunks = {
            v["chunk_id"]: v
            for v in (
                json.loads(line)
                for line in (root / "data/p3/indexes" / request["generation"] / "chunks.jsonl")
                .read_text()
                .splitlines()
            )
        }
        vectors = [
            json.loads(line)
            for line in (root / "data/v8/p3_vectors.jsonl").read_text().splitlines()
        ]
        for start in range(0, len(vectors), 32):
            rows = [
                {
                    "id": p["payload"]["chunk_id"],
                    "embedding": p["vector"],
                    "text": chunks[p["payload"]["chunk_id"]]["text"],
                    "source": p["payload"]["source_id"],
                    "section": chunks[p["payload"]["chunk_id"]]["parent_id"],
                }
                for p in vectors[start : start + 32]
            ]
            driver.execute_query(
                "UNWIND $rows AS row MERGE (n:V8BaselineChunk {id:row.id}) "
                "SET n.embedding=row.embedding,n.text=row.text,n.source=row.source,"
                "n.section=row.section",
                rows=rows,
            )
        driver.execute_query(
            "CREATE VECTOR INDEX v8_baseline_vector IF NOT EXISTS "
            "FOR (n:V8BaselineChunk) ON (n.embedding) OPTIONS {indexConfig: {"
            "`vector.dimensions`:2560,`vector.similarity_function`:'cosine'}}"
        )
        driver.execute_query(
            "CREATE FULLTEXT INDEX v8_baseline_fulltext IF NOT EXISTS "
            "FOR (n:V8BaselineChunk) ON EACH [n.text]"
        )
        driver.execute_query("CALL db.awaitIndexes(120)")
        driver.execute_query(
            "MATCH (a:V8BaselineChunk),(b:V8BaselineChunk) "
            "WHERE a.section=b.section AND a.source=b.source AND a.id<b.id "
            "MERGE (a)-[:V8BaselineRelation]->(b)"
        )
        retriever = HybridCypherRetriever(
            driver,
            "v8_baseline_vector",
            "v8_baseline_fulltext",
            "OPTIONAL MATCH (node)-[:V8BaselineRelation]-(related:V8BaselineChunk) "
            "RETURN node.id AS seed,collect(DISTINCT related.id)[..100] AS related,score",
        )
        results = []
        for query in request["queries"]:
            found = retriever.get_search_results(
                query_text=query["question"], query_vector=query["vector"], top_k=8
            )
            results.append(
                {"question": query["question"], "records": [r.data() for r in found.records]}
            )
        output = {
            "implementation": "neo4j-graphrag.HybridCypherRetriever",
            "version": "1.22.0",
            "vectors": len(vectors),
            "results": results,
            "substrate_difference": (
                "Neo4j fulltext/vector scoring differs from exact/BM25/RRF/Qwen reranker"
            ),
        }
        Path(request["output"]).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {"output": request["output"], "queries": len(results), "vectors": len(vectors)}
            )
        )
    finally:
        driver.close()


if __name__ == "__main__":
    run(Path(sys.argv[1]))
