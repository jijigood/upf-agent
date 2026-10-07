"""Import explicit human truth decisions, retaining originals and binding their hashes."""

import json
import re
from pathlib import Path

from upf_agent.p6_contracts import EvalCase, validate_partition
from upf_agent.quality_repair import sha, write_json


def import_review(suite: Path, review: Path, output: Path, overrides: Path | None = None) -> dict:
    if output.exists():
        raise ValueError("Use a fresh reviewed suite path")
    cases = [EvalCase.model_validate_json(line) for line in suite.read_text().splitlines() if line]
    corrections = json.loads(overrides.read_text()) if overrides else {}
    if set(corrections) - {case.id for case in cases}:
        raise ValueError("Unknown case in overrides")
    sections = re.split(r"(?m)^## ", review.read_text())[1:]
    decisions = {}
    for section in sections:
        cid = section.split()[0]
        if cid in decisions:
            raise ValueError("Duplicate review section")
        decisions[cid] = section
    if set(decisions) != {case.id for case in cases}:
        raise ValueError("Review must contain exactly the suite case IDs")
    result = []
    for case in cases:
        # The source blocks are untrusted data, never a human signature.
        text = re.split(r"(?m)^\s*(?:```|~~~)", decisions[case.id], maxsplit=1)[0]
        pairs = re.findall(r"(?m)^- (结论|复核人|日期|修正意见|问题)：(.*)$", text)
        fields = dict(pairs)
        if len(fields) != len(pairs):
            raise ValueError(f"{case.id}: duplicate human review fields")
        if not re.search(r"(?mi)^- \[x\] 问题及预期事实/相关证据已核对", text):
            raise ValueError(f"{case.id}: human review checkbox missing")
        if fields.get("结论", "").strip() != "通过":
            raise ValueError(f"{case.id}: conclusion must explicitly be 通过")
        reviewer, date = fields.get("复核人", "").strip(), fields.get("日期", "").strip()
        if not reviewer or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise ValueError(f"{case.id}: named reviewer and YYYY-MM-DD required")
        opinion = fields.get("修正意见", "").strip()
        changes = corrections.get(case.id, {})
        allowed = {
            "question",
            "evidence_ids",
            "expected_extracts",
            "forbidden_claims",
            "expected_statuses",
        }
        if set(changes) - allowed:
            raise ValueError("Overrides cannot change source groups, splits or provenance")
        if opinion and opinion not in {"无", "无需修改"} and not changes:
            raise ValueError(f"{case.id}: translate corrections into explicit overrides JSON")
        question = fields.get("问题", "").strip()
        if question != changes.get("question", case.question):
            raise ValueError(f"{case.id}: changed question requires an explicit override")
        values = case.model_dump() | changes
        values.update(label_status="reviewed", reviewer=reviewer, review_date=date)
        result.append(EvalCase.model_validate(values))
    validate_partition(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(case.model_dump_json() for case in result) + "\n")
    receipt = {
        "reviewed": len(result),
        "input_sha256": sha(suite),
        "review_sha256": sha(review),
        "overrides_sha256": sha(overrides) if overrides else None,
        "output_sha256": sha(output),
        "scope": "business labels; not model answer approval",
    }
    write_json(output.with_suffix(".receipt.json"), receipt)
    return receipt


def adjudicate_answers(suite: Path, summary: Path, review: Path, output: Path) -> dict:
    """Calculate human answer metrics only from a complete, run-bound signed sheet."""
    if output.exists():
        raise ValueError("Preserve adjudication; use a fresh output path")
    cases = [EvalCase.model_validate_json(line) for line in suite.read_text().splitlines() if line]
    report, human = json.loads(summary.read_text()), json.loads(review.read_text())
    if report.get("suite_sha256") != sha(suite) or human.get("summary_sha256") != sha(summary):
        raise ValueError("Review/suite does not match the executed run")
    selected = {case.id: case for case in cases if case.split == report["split"]}
    execution = {row["case"]: row for row in report["rows"]}
    decisions = {row["case"]: row for row in human["rows"]}
    if (
        len(decisions) != len(human["rows"])
        or set(decisions) != set(selected)
        or set(execution) != set(selected)
    ):
        raise ValueError("Human review must cover exactly the executed partition")
    for cid, row in decisions.items():
        if not row.get("reviewer") or not row.get("date"):
            raise ValueError(f"{cid}: missing human attribution")
        required = ["supported", "complete"]
        if selected[cid].category == "command":
            required += ["command_exact", "unsupported_critical_parameter"]
        if selected[cid].category == "missing":
            required += ["missing_source_correct"]
        if any(type(row.get(key)) is not bool for key in required):
            raise ValueError(f"{cid}: fill every applicable judgement with true/false")

    def metric(field, category=None):
        rows = [
            row
            for cid, row in decisions.items()
            if category is None or selected[cid].category == category
        ]
        return {
            "n": len(rows),
            "pass": sum(row[field] is True for row in rows),
            "rate": sum(row[field] is True for row in rows) / len(rows) if rows else None,
        }

    support, complete = metric("supported"), metric("complete")
    command, missing = (
        metric("command_exact", "command"),
        metric("missing_source_correct", "missing"),
    )
    invented = sum(row.get("unsupported_critical_parameter") is True for row in decisions.values())
    gate = (
        support["rate"] >= 0.95
        and command["rate"] is not None
        and command["rate"] >= 0.95
        and missing["rate"] is not None
        and missing["rate"] >= 0.9
        and invented == 0
    )
    result = {
        "summary_sha256": sha(summary),
        "review_sha256": sha(review),
        "suite_sha256": sha(suite),
        "split": report["split"],
        "cases": len(decisions),
        "support": support,
        "completeness": complete,
        "command_exact": command,
        "missing_source_correct": missing,
        "invented_critical_parameters": invented,
        "semantic_gate_pass": gate,
        "human_review_complete": True,
        "M6_pass": False,
        "remaining": "retrieval/tools/performance/release gates evaluated separately",
    }
    write_json(output, result)
    return result
