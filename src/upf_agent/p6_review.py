"""Import explicit human truth decisions, retaining originals and binding their hashes."""

import json
import re
from pathlib import Path

from upf_agent.p6_contracts import EvalCase, validate_partition
from upf_agent.p6_truth_validation import validate_reviewed_sources
from upf_agent.quality_repair import sha, write_json


def import_review(
    suite: Path,
    review: Path,
    output: Path,
    overrides: Path | None = None,
    root: Path | None = None,
) -> dict:
    if output.exists():
        raise ValueError("Use a fresh reviewed suite path")
    cases = [EvalCase.model_validate_json(line) for line in suite.read_text().splitlines() if line]
    corrections = json.loads(overrides.read_text()) if overrides else {}
    if set(corrections) - {case.id for case in cases if case.label_status == "draft"}:
        raise ValueError("Unknown case in overrides")
    sections = re.split(r"(?m)^## ", review.read_text())[1:]
    decisions = {}
    for section in sections:
        cid = section.split()[0]
        if cid in decisions:
            raise ValueError("Duplicate review section")
        decisions[cid] = section
    draft_ids = {case.id for case in cases if case.label_status == "draft"}
    if set(decisions) != draft_ids:
        raise ValueError("Review must contain exactly the draft suite case IDs")
    result = []
    for case in cases:
        if case.label_status == "reviewed":
            result.append(case)
            continue
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
    source_validation = validate_reviewed_sources(root, result) if root is not None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(case.model_dump_json() for case in result) + "\n")
    receipt = {
        "reviewed": len(result),
        "newly_reviewed": len(draft_ids),
        "carried_reviewed": len(result) - len(draft_ids),
        "input_sha256": sha(suite),
        "review_sha256": sha(review),
        "overrides_sha256": sha(overrides) if overrides else None,
        "output_sha256": sha(output),
        "scope": "business labels; not model answer approval",
        "source_validation": source_validation,
    }
    write_json(output.with_suffix(".receipt.json"), receipt)
    return receipt


def adjudicate_answers(suite: Path, summary: Path, review: Path, output: Path) -> dict:
    """Calculate run-bound scores and keep AI reviews outside human acceptance."""
    if output.exists():
        raise ValueError("Preserve adjudication; use a fresh output path")
    cases = [EvalCase.model_validate_json(line) for line in suite.read_text().splitlines() if line]
    report, human = json.loads(summary.read_text()), json.loads(review.read_text())
    if report.get("execution_complete") is False:
        raise ValueError("Run inputs changed or execution is incomplete; cannot adjudicate")
    if report.get("suite_sha256") != sha(suite) or human.get("summary_sha256") != sha(summary):
        raise ValueError("Review/suite does not match the executed run")
    if report.get("split") == "heldout" and (
        report.get("machine_review_gate", {}).get("pass") is False
        or any(row.get("expected_status") is False or row.get("failure") for row in report["rows"])
    ):
        raise ValueError("Heldout machine gate failed; do not request human answer scoring")
    selected = {case.id: case for case in cases if case.split == report["split"]}
    execution = {row["case"]: row for row in report["rows"]}
    decisions = {row["case"]: row for row in human["rows"]}
    review_kind = human.get("review_kind", "human")
    if review_kind not in {"human", "ai_assisted"}:
        raise ValueError("review_kind must be human or ai_assisted")
    if any("AI辅助复核" in str(row.get("reviewer", "")) for row in decisions.values()):
        review_kind = "ai_assisted"
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
        "review_kind": review_kind,
        "review_complete": True,
        "human_review_complete": review_kind == "human",
        "M6_pass": False,
        "remaining": "retrieval/tools/performance/release gates evaluated separately",
    }
    write_json(output, result)
    return result
