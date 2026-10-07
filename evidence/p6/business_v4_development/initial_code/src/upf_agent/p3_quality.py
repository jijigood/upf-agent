"""Source-grounded machine regression; never substitutes for human business evaluation."""

import json
import re
from pathlib import Path
from time import perf_counter

from tokenizers import Tokenizer

from upf_agent.config import load_model_config
from upf_agent.corpus_db import active_id, connect
from upf_agent.models import LocalChatClient
from upf_agent.p3_answer import answer_from_retrieval, render_answer, source_lookup
from upf_agent.p3_index import _config, load_active_index, load_chunks
from upf_agent.p3_search import expand_context, search
from upf_agent.quality_repair import sha, write_json


def evaluate_answer(case: dict, answer: dict) -> dict:
    citations = {c["chunk_id"]: c for c in answer["citations"]}
    exact_quotes = all(
        cite["chunk_id"] in citations and cite["quote"] in citations[cite["chunk_id"]]["text"]
        for claim in answer["claims"]
        for cite in claim["citations"]
    )
    text = re.sub(r"\s+", "", "\n".join(c["text"] for c in answer["claims"])).casefold()
    required = case.get("critical_literals", [])
    literal_coverage = all(re.sub(r"\s+", "", value).casefold() in text for value in required)
    status_ok = answer["status"] in case["expected_statuses"]
    refusal_ok = case.get("expect_refusal") is not True or not answer["claims"]
    source_extract_integrity = None
    if (
        answer.get("publication_policy")
        == "exact_selected_source_units_model_paraphrase_audit_only"
    ):
        source_extract_integrity = all(
            claim["text"] == "\n".join(dict.fromkeys(cite["quote"] for cite in claim["citations"]))
            for claim in answer["claims"]
        )
    return {
        "status_expected": status_ok,
        "quote_integrity": exact_quotes,
        "refusal_contract": refusal_ok,
        "source_extract_integrity": source_extract_integrity,
        "critical_literal_coverage": literal_coverage if required else None,
        "machine_contract_pass": status_ok
        and exact_quotes
        and refusal_ok
        and literal_coverage
        and source_extract_integrity is not False,
        "business_semantics": "pending_human_review",
    }


def run_quality(root: Path, dataset: Path, output: Path, device: str = "cuda:1") -> dict:
    directory, manifest = load_active_index(root)
    config, fingerprint = _config(root)
    db = connect(root / "data/p2/corpus.sqlite3")
    try:
        if active_id(db) != manifest["p2_generation"]:
            raise ValueError("Quality index is stale against active P2 generation")
    finally:
        db.close()
    if any(manifest["identity"].get(key) != value for key, value in fingerprint.items()):
        raise ValueError("Quality evaluation index/config mismatch")
    cases = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Quality dataset empty or duplicate IDs")
    if any(
        c["index_generation"] != manifest["index_generation"]
        or c["split"] != "development_regression"
        for c in cases
    ):
        raise ValueError("Quality dataset generation/split mismatch")
    ablation_path = root / "evidence/p3/ablation.json"
    ablation = json.loads(ablation_path.read_text())
    if ablation["index_generation"] != manifest["index_generation"]:
        raise ValueError("Cached retrieval generation mismatch")
    cached = {row["case"]["id"]: row for row in ablation["rows"]}
    chunks = load_chunks(directory)
    tokenizer = Tokenizer.from_file(str(config["tokenizer_path"]))
    model = load_model_config(root / "config/models.yaml").generation["baseline"]
    report = {
        "index_generation": manifest["index_generation"],
        "dataset_sha256": sha(dataset),
        "ablation_sha256": sha(ablation_path),
        "model": model.id,
        "model_config": model.model_dump(mode="json"),
        "model_manifest_sha256": sha(root / "evidence/p1/model-hashes.json"),
        "pipeline_files_sha256": {
            name: sha(root / "src/upf_agent" / name)
            for name in ("p3_answer.py", "p3_search.py", "p3_quality.py")
        },
        "scope": "machine_contract_and_assistant_source_literal_regression",
        "human_signoff": False,
        "business_accuracy_claim": False,
        "completed": False,
        "rows": [],
    }
    write_json(output, report)
    with LocalChatClient(model) as client:
        for case in cases:
            started = perf_counter()
            if case.get("probe_id"):
                row = cached[case["probe_id"]]
                if any(case[k] != row["case"][k] for k in ("question", "product", "version")):
                    raise ValueError("Cached retrieval question/scope mismatch")
                candidates = {c["chunk_id"]: c for c in row["candidates"]}
                selected = [
                    candidates[cid]
                    for cid in row["rankings"]["qwen_bm25_exact_rerank"][
                        : int(config["final_evidence"])
                    ]
                ]
                evidence, tokens = expand_context(
                    selected,
                    chunks,
                    tokenizer,
                    int(config["context_token_budget"]),
                    int(config["neighbor_token_budget"]),
                )
                retrieval = {
                    "index_generation": manifest["index_generation"],
                    "p2_generation": manifest["p2_generation"],
                    "product": case["product"],
                    "version": case["version"],
                    "evidence": evidence,
                    "context_tokens": tokens,
                    "retrieval_origin": "same-budget-ablation-cache",
                }
            else:
                retrieval = search(root, case["question"], case["product"], case["version"], device)
            answer = answer_from_retrieval(root, case["question"], retrieval, client)
            for citation in answer["citations"]:
                source_lookup(root, answer["index_generation"], citation["chunk_id"])
            result = {
                "case": case,
                "answer": answer,
                "retrieval": retrieval,
                "checks": evaluate_answer(case, answer),
                "context_tokens": retrieval["context_tokens"],
                "elapsed_ms": round((perf_counter() - started) * 1000),
            }
            report["rows"].append(result)
            write_json(output, report)
            print(
                case["id"], answer["status"], result["checks"]["machine_contract_pass"], flush=True
            )
    if sha(dataset) != report["dataset_sha256"] or sha(ablation_path) != report["ablation_sha256"]:
        raise ValueError("Evaluation inputs changed during execution")
    report["completed"] = True
    report["summary"] = {
        "cases": len(cases),
        "machine_contract_pass": sum(x["checks"]["machine_contract_pass"] for x in report["rows"]),
        "quote_integrity_pass": sum(x["checks"]["quote_integrity"] for x in report["rows"]),
        "generation_failed": sum(x["answer"]["status"] == "failed" for x in report["rows"]),
        "human_semantic_completed": 0,
    }
    write_json(output, report)
    lines = [
        "# P3 本地问答回归",
        "",
        "只衡量状态、引用原文、固定关键字和拒答契约；不是人工业务准确率或正式测试集验收。",
        json.dumps(report["summary"], ensure_ascii=False),
        "",
    ]
    for row in report["rows"]:
        lines += [
            f"## {row['case']['id']}",
            render_answer(row["answer"]),
            "自动检查：" + json.dumps(row["checks"], ensure_ascii=False),
        ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return report["summary"]
