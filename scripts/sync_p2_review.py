"""Record the submitted review without changing its conclusions or source annotations."""

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from upf_agent.quality_repair import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def followup_records(text: str) -> dict[int, dict[str, str]]:
    sections = re.split(r"(?m)^## 截图 (\d+)\s*$", text)
    records = {}
    for offset in range(1, len(sections), 2):
        number, body = int(sections[offset]), sections[offset + 1]
        if number in records:
            raise ValueError("Duplicate follow-up screenshot")
        fields = dict(re.findall(r"(?m)^- ([^：\n]+)：([^\n]*)$", body))
        decision = re.fullmatch(r"(通过|不通过|无法确认)(?:[（(;；].*)?", fields.get("结论", ""))
        checked = re.search(r"(?m)^- \[([ xX])\] 修正后的可见内容与原图一致$", body)
        if decision is None or checked is None or "定位" not in fields:
            raise ValueError("Follow-up requires explicit conclusion, checkbox and locator")
        conclusion = decision.group(1)
        if (checked.group(1).lower() == "x") != (conclusion == "通过"):
            raise ValueError("Follow-up checkbox and conclusion disagree")
        records[number] = {**fields, "normalized_conclusion": conclusion, "submitted_section": body}
    if set(records) != {1, 7, 17, 20}:
        raise ValueError("Follow-up must cover exactly screenshots 01/07/17/20")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-confirmed", action="store_true", required=True)
    parser.add_argument("--followup", action="store_true")
    args = parser.parse_args()
    sheet = ROOT / "evidence/p2/review_sheet_c05fe621e37ebdb1c663_v3.md"
    packet = ROOT / "evidence/p2/structure_review_60.json"
    corrected = ROOT / "evidence/p2/ocr_corrected.json"
    prior = ROOT / "evidence/p2/archive/user_review_v3/ocr_corrected.json"
    receipt_path = ROOT / "evidence/p2/human_review_v3.json"
    existing = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    use_followup = args.followup or existing.get("followup_applied", False)
    followup = ROOT / "evidence/p2/review_followup_v3_4_images.md"
    records = followup_records(followup.read_text()) if use_followup else {}
    structure = json.loads(packet.read_text())
    images = json.loads(corrected.read_text())
    previous_images = json.loads(prior.read_text())["images"]
    sections = re.split(r"(?m)^(?=- \[[ xX]\] )", sheet.read_text())[1:]
    if len(sections) != 90:
        raise ValueError("Review sheet must contain the frozen 90 items")
    sources = structure["items"] + images["images"]
    items = []
    pending = []
    for index, (section, source) in enumerate(zip(sections, sources, strict=True)):
        heading = section.splitlines()[0]
        fields = {
            match.group(1): match.group(2).strip()
            for match in re.finditer(r"(?m)^  - ([^：\n]+)：(.*)$", section)
        }
        conclusion = fields.get("复核结论")
        if conclusion not in {"通过", "不通过", "无法确认"}:
            raise ValueError("Review conclusion is absent or invalid")
        if source["locator"] not in heading:
            raise ValueError("Review locator differs from frozen source")
        if heading.startswith("- [x]") != (conclusion == "通过"):
            raise ValueError("Review checkbox and conclusion disagree")
        record = {
            "number": index + 1,
            "category": source.get("category", "screenshot"),
            "source_id": source["source_id"],
            "block_id": source["block_id"],
            "locator": source["locator"],
            "conclusion": conclusion,
            "review_date_as_written": fields.get("日期"),
            "reviewer_as_written": fields.get("复核人"),
            "note": fields.get("问题与正确原文"),
            "full_submitted_section": section,
        }
        if index >= 60:
            number = index - 59
            record["screenshot"] = number
            record["image_sha256"] = source["image_sha256"]
            if conclusion == "通过" and source != previous_images[number - 1]:
                raise ValueError("Previously accepted screenshot content has changed")
            if conclusion != "通过":
                pending.append(number)
        items.append(record)
    for number, result in records.items():
        record = items[59 + number]
        if result["定位"].strip("`") != record["locator"]:
            raise ValueError("Follow-up screenshot locator differs from frozen source")
        record["initial_conclusion"] = record["conclusion"]
        record["conclusion"] = result["normalized_conclusion"]
        record["followup"] = result
    pending = [item["screenshot"] for item in items[60:] if item["conclusion"] != "通过"]
    receipt = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "generation": structure["generation"],
        "user_declared_review_completed": args.user_confirmed,
        "attestation_basis": "user_message: 我已经完成复核，并提交当前 v3 文件",
        "followup_applied": use_followup,
        "named_human_signature_present": False,
        "reviewer_note": "保留原复核人字段；以用户明确提交的完成声明记账，不代填姓名或签名",
        "counts": dict(Counter(item["conclusion"] for item in items)),
        "corrected_pending_recheck": pending,
        "m2_acceptance_pass": not pending and all(item["conclusion"] == "通过" for item in items),
        "bindings_sha256": {
            str(path.relative_to(ROOT)): sha(path)
            for path in (sheet, packet, corrected, prior, ROOT / "evidence/p2/ocr_corrections.json")
        },
        "items": items,
    }
    if use_followup:
        receipt["attestation_basis"] += "; user_message: review_followup_v3_4_images.md 已经复核"
        receipt["bindings_sha256"][str(followup.relative_to(ROOT))] = sha(followup)
    write_json(ROOT / "evidence/p2/human_review_v3.json", receipt)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in ("counts", "corrected_pending_recheck", "m2_acceptance_pass")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
