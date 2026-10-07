"""Rebase source IDs after the native-command fix, retaining submitted decisions."""

import json
from collections import Counter
from pathlib import Path

from upf_agent.p3_index import load_active_index, load_chunks
from upf_agent.p6_contracts import EvalCase, validate_partition
from upf_agent.p6_truth_validation import validate_reviewed_sources
from upf_agent.quality_repair import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "eval/p6/business_v4"


def main() -> None:
    if DEST.exists():
        raise ValueError("Preserve existing revision; do not overwrite human reviews")
    submitted = ROOT / "eval/p6/truth_review.md"
    receipt_path = ROOT / "evidence/p6/review_followup" / sha(submitted)[:16] / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["submitted_sha256"] != sha(submitted):
        raise ValueError("Submitted review changed since receipt")
    previous = ROOT / "eval/p6/business_v3/candidates.jsonl"
    cases = [EvalCase.model_validate_json(line) for line in previous.read_text().splitlines()]
    old = load_chunks(ROOT / "data/p3/indexes" / cases[0].index_generation)
    directory, manifest = load_active_index(ROOT)
    current = load_chunks(directory)

    def source_key(chunk):
        return (
            chunk.source_id,
            tuple(chunk.block_ids),
            chunk.kind,
            tuple(chunk.section_path),
            chunk.command_id,
        )

    by_source = {source_key(chunk): chunk for chunk in current.values()}
    mapping = {cid: by_source[source_key(chunk)] for cid, chunk in old.items()}
    replacements = {chunk.text: mapping[cid].text for cid, chunk in old.items()}
    statuses, changed = {}, []
    for case in cases:
        before = case.model_copy(deep=True)
        case.index_generation = manifest["index_generation"]
        if case.category in {"command", "knowledge", "troubleshooting"}:
            case.evidence_ids = [mapping[eid].chunk_id for eid in case.evidence_ids]
            case.expected_extracts = [replacements[text] for text in case.expected_extracts]
        altered = before.expected_extracts != case.expected_extracts or any(
            old[eid].text != mapping[eid].text for eid in before.evidence_ids if eid in old
        )
        if altered:
            changed.append(case.id)
        if case.id in receipt["v3_changed_cases"]:
            statuses[case.id] = "new_business_question"
        elif receipt["decisions"][case.id]["结论"] != "通过" or altered:
            statuses[case.id] = "corrected_source_requires_recheck"
        else:
            statuses[case.id] = "same_truth_prior_pass_attribution_pending"
    validate_partition(cases)
    validation = validate_reviewed_sources(ROOT, cases)
    assert changed == [f"command-{number:03d}" for number in range(9, 13)]
    assert Counter(statuses.values()) == {
        "new_business_question": 44,
        "corrected_source_requires_recheck": 4,
        "same_truth_prior_pass_attribution_pending": 72,
    }
    DEST.mkdir()
    suite = DEST / "candidates.jsonl"
    suite.write_text("\n".join(case.model_dump_json() for case in cases) + "\n")
    references = json.loads((ROOT / "eval/p6/source_references.json").read_text())
    intro = [
        "# P6 业务真值复核单 v4：接收旧版复核后的来源修订",
        "",
        "120题，开发/保留各60；问题、来源分组及划分与v3相同。",
        "本轮重建原生命令索引，删除程序拼接的“相关说明”；示例独立保留原位置。",
        "旧版116通过、4不通过的提交保留；其中72题与本版真值一致，原结论列出供确认。",
        "旧版复核人仍为AI辅助待人工签名，因此尚未导入正式真值或代填实际姓名。",
        "本轮需查看48题：44道新版题及command-009至012的来源修正。",
        "填实际复核人/日期；新题与修正题通过后勾选。原件矛盾保留，禁止猜测现场事实。",
        "",
    ]
    review, pending = (
        list(intro),
        [*intro, "本文件仅列出48道需要本轮内容复核的题；完整版在truth_review.md。", ""],
    )
    source_rows = [
        "# 本轮48题原件定位",
        "",
        "DOCX内部p[...]是XML位置，不是Word页码；在Word/WPS搜索以下章节标题。",
        "",
        "| 题号 | 原件链接 | 章节路径 |",
        "|---|---|---|",
    ]
    for case in cases:
        state = statuses[case.id]
        prior = receipt["decisions"][case.id]
        retained = state == "same_truth_prior_pass_attribution_pending"
        section = [
            f"## {case.id} · {case.split} · {case.category}",
            "",
            "- [x] 问题及预期事实/相关证据已核对"
            if retained
            else "- [ ] 问题及预期事实/相关证据已核对",
            f"- 问题：{case.question}",
            "- 结论：通过" if retained else "- 结论：待复核",
            f"- 复核人：{prior['复核人']}" if retained else "- 复核人：",
            f"- 日期：{prior['日期']}" if retained else "- 日期：",
            "- 修正意见：无" if retained else "- 修正意见：",
            f"- 本轮状态：{state}",
            "- 分组：" + ", ".join(case.source_groups),
            f"- synthetic：{case.synthetic}",
            "- 禁止结论：" + "；".join(case.forbidden_claims),
        ]
        if state == "corrected_source_requires_recheck":
            section += [
                f"- 旧版结论：{prior['结论']}；原意见：{prior['修正意见']}",
                "- 本轮处理：命令格式与示例各自成块，各自引用p[177]、p[183]；请确认下方修正来源。",
            ]
        paths = {}
        for eid in case.evidence_ids:
            if eid in current:
                chunk = current[eid]
                path = ROOT / "data/raw" / chunk.relative_path
                section += [
                    f"- 来源：[{chunk.relative_path}](<{path}>) · `{eid}`",
                    "- 定位：" + ", ".join(chunk.locators),
                    "```text",
                    chunk.text,
                    "```",
                ]
                paths.setdefault(path, set()).add(" / ".join(chunk.section_path))
            else:
                section += [f"- 来源：{eid}"]
        if case.category == "log":
            for event in references[case.id]["data"]["events"]:
                section += [
                    f"- 原始日志：{event['event_id']}",
                    f"- 定位：行{event['line_start']} · 候选状态{event['event_state']}",
                    "```text",
                    event["raw_excerpt"].rstrip(),
                    "```",
                ]
        section += ["- 预期原文/缺口：", "```text", *case.expected_extracts, "```", ""]
        review.extend(section)
        if not retained:
            pending.extend(section)
            for path, headings in paths.items():
                label = f"[{path.name}](<{path}>)"
                heading = "<br>".join(sorted(headings)).replace("|", "\\|")
                source_rows.append(f"| {case.id} | {label} | {heading} |")
    (DEST / "truth_review.md").write_text("\n".join(review) + "\n")
    (DEST / "pending_48_review.md").write_text("\n".join(pending) + "\n")
    (DEST / "pending_48_sources.md").write_text("\n".join(source_rows) + "\n")
    record = {
        "suite_sha256": sha(suite),
        "previous_v3_sha256": sha(previous),
        "submitted_review_sha256": sha(submitted),
        "submitted_receipt": str(receipt_path.relative_to(ROOT)),
        "index_generation": manifest["index_generation"],
        "source_validation": validation,
        "cases": len(cases),
        "splits": dict(Counter(case.split for case in cases)),
        "status_counts": dict(Counter(statuses.values())),
        "case_status": statuses,
        "source_fix_affected_cases": changed,
        "questions_groups_and_splits_changed_from_v3": False,
        "human_truth_formally_imported": 0,
        "actual_human_attribution_pending": True,
        "heldout_executed": False,
        "M6_pass": False,
    }
    write_json(DEST / "manifest.json", record)
    print(json.dumps(record["status_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
