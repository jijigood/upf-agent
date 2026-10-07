import json
from hashlib import sha256
from pathlib import Path

from upf_agent.p3_answer import (
    DraftAnswer,
    publish_source_extracts,
    render_answer,
    source_lookup,
    validate_quotes,
)
from upf_agent.p3_quality import evaluate_answer

r = Path(__file__).resolve().parents[1]
p = r / "evidence/p3/quality_regression.json"
source = r / "evidence/p3/archive/c61e9796eead2610cfd4/answer-v3/quality_regression.json"
oldbytes = source.read_bytes()
report = json.loads(oldbytes)
assert report["completed"]
assert all(
    row["answer"]["pipeline_version"] == "p3-answer-v3-source-term-gate" for row in report["rows"]
)
assert (
    sha256((r / "eval/p3_answer_regression.jsonl").read_bytes()).hexdigest()
    == report["dataset_sha256"]
)
report["processing_mode"] = (
    "deterministic_publication_replay_of_v3_real_model_outputs_no_new_inference"
)
report["replayed_from_sha256"] = sha256(oldbytes).hexdigest()
report["model_execution_pipeline_files_sha256"] = report["pipeline_files_sha256"]
report["pipeline_files_sha256"] = {
    name: sha256((r / "src/upf_agent" / name).read_bytes()).hexdigest()
    for name in ["p3_answer.py", "p3_search.py", "p3_quality.py"]
}
for row in report["rows"]:
    answer = row["answer"]
    draft = DraftAnswer.model_validate(
        {key: answer[key] for key in ["status", "claims", "missing_information"]}
    )
    published = publish_source_extracts(draft)
    sources = [
        source_lookup(r, answer["index_generation"], c["chunk_id"]) for c in answer["citations"]
    ]
    validate_quotes(published, sources)
    answer.update(published.model_dump())
    answer["pipeline_version"] = "p3-answer-v4-source-extracts"
    answer["publication_policy"] = "exact_selected_source_units_model_paraphrase_audit_only"
    answer["warnings"].append(
        "回答发布所选手册原文片段；完整操作流程和示例参数的现场适用性尚未验证。"
    )
    row["checks"] = evaluate_answer(row["case"], answer)
    row["processing_mode"] = report["processing_mode"]
report["summary"] = {
    "cases": len(report["rows"]),
    "machine_contract_pass": sum(x["checks"]["machine_contract_pass"] for x in report["rows"]),
    "quote_integrity_pass": sum(x["checks"]["quote_integrity"] for x in report["rows"]),
    "generation_failed": sum(x["answer"]["status"] == "failed" for x in report["rows"]),
    "human_semantic_completed": 0,
}
p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
lines = [
    "# P3 本地问答回归（原文发布重放）",
    "",
    (
        "第三轮 16 题真实检索/模型结果全部归档。"
        "当前报告只对这些结果重新应用原文发布规则，没有重新运行 16 次模型推理。"
    ),
    "模型归纳只保留在 audit，正式 claims.text 来自其引用原文。机械契约通过不等于正式业务准确率。",
    json.dumps(report["summary"], ensure_ascii=False),
]
for row in report["rows"]:
    lines += [
        f"\n## {row['case']['id']}",
        render_answer(row["answer"]),
        json.dumps(row["checks"], ensure_ascii=False),
    ]
p.with_suffix(".md").write_text("\n".join(lines) + "\n")
print(report["summary"])
