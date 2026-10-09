"""Export empty expert forms or validate actual review records without altering a pilot."""

import argparse
import json
from pathlib import Path

from upf_agent.v8.models import digest
from upf_agent.v8.phase1 import write_json
from upf_agent.v8.plan_ir import CaseGold
from upf_agent.v8.plan_review import GoldReview, apply_reviews, calibration_report


def review_package(pilot: Path, output: Path, reviews: Path | None = None):
    if output.exists():
        raise ValueError("review output exists; use a new review ID")
    golds = [
        CaseGold.model_validate_json(p.read_text())
        for p in sorted((pilot / "cases").glob("*/case-reference.json"))
    ]
    if not golds:
        raise ValueError("pilot contains no frozen references")
    by_case = {g.case_id: g for g in golds}
    records = (
        []
        if reviews is None
        else [GoldReview.model_validate(r) for r in json.loads(reviews.read_text())]
    )
    keys = [(r.case_id, r.reviewer) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate case/reviewer review records")
    for record in records:
        if record.case_id not in by_case or record.reference_sha256 != digest(
            by_case[record.case_id]
        ):
            raise ValueError("review does not match frozen case/reference")
    approved, pending, adjudication = {}, [], []
    for gold in golds:
        pair = [r for r in records if r.case_id == gold.case_id]
        if len(pair) > 2:
            raise ValueError("more than two reviews; adjudication requires a new reference/version")
        if len(pair) != 2:
            pending.append(gold.case_id)
        elif all(r.decision == "approve" for r in pair):
            approved[gold.case_id] = apply_reviews(gold, pair)
        else:
            adjudication.append(gold.case_id)
    candidates = [g.case_id for g in golds if int(g.case_id.split("-")[1]) <= 10]
    status = {
        "schema_version": "v8-phase2-expert-intake-1",
        "pilot": str(pilot.resolve()),
        "cases": len(golds),
        "actual_review_records": len(records),
        "paired_review_calibration": calibration_report(records),
        "calibration_candidates": candidates,
        "calibration_pass": len(candidates) == 10 and set(candidates) <= set(approved),
        "approved_cases": sorted(approved),
        "pending_cases": pending,
        "adjudication_cases": adjudication,
        "all_references_reviewed": len(approved) == len(golds),
        "review_authenticity": (
            "intake validates supplied records; humans establish expertise/independence"
        ),
        "new_model_calls": 0,
        "original_pilot_modified": False,
    }
    output.mkdir(parents=True)
    for gold in golds:
        for seat in ["A", "B"]:
            write_json(
                output / "empty-forms" / gold.case_id / f"reviewer-{seat}.json",
                {
                    "case_id": gold.case_id,
                    "reference_sha256": digest(gold),
                    "reviewer": None,
                    "decision": None,
                    "notes": [],
                    "review_version": None,
                    "review_source": None,
                },
            )
    for case_id, reviewed in approved.items():
        write_json(
            output / "reviewed-references" / f"{case_id}.json", reviewed.model_dump(mode="json")
        )
    write_json(output / "status.json", status)
    write_json(output / "received-reviews.json", [r.model_dump(mode="json") for r in records])
    (output / "README.md").write_text(
        "# 独立专家复核入口\n\n"
        "空表不代表已评审。两名不同领域评审先读 cases 下的冻结事实材料及来源，再审参考分支；"
        "填写真实 reviewer、decision（approve/request_changes）、notes、"
        "review_version、review_source。"
        "reference_sha256 不得改写；有分歧时先仲裁并冻结新参考版本，"
        "再重新双评，不能自动多数表决。\n\n"
        "前十例用于校准，全部二十四例须有独立双评后才可作为正式 Gold。"
        "方案原文评审另见原批次 review-package；匿名化导出不等于盲评已经完成。\n\n"
        "将完成的记录组成 JSON 数组，使用脚本的 --reviews 参数，输出至新的目录。"
        "导入只生成新审核产物，不覆盖实验原文、指标或参考。\n"
    )
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    print(json.dumps(review_package(args.pilot, args.output, args.reviews), ensure_ascii=False))


if __name__ == "__main__":
    main()
