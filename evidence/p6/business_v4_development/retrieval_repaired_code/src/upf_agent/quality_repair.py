"""Traceable OCR corrections and asset rebinding; human sign-off is never inferred."""

import copy
import hashlib
import json
import os
from pathlib import Path

from upf_agent.auxiliary import SCHEMA
from upf_agent.corpus_db import active_id, connect
from upf_agent.image_queue import record_ocr_results


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def active_ocr_path(root: Path) -> Path:
    db = connect(root / "data/p2/corpus.sqlite3")
    try:
        generation = active_id(db)
        for relative, expected_sha in db.execute(
            "SELECT relative_path,sha256 FROM sources WHERE generation=?", (generation,)
        ):
            if sha(root / "data/raw" / relative) != expected_sha:
                raise ValueError("OCR cannot be rebound: original manual source SHA changed")
    finally:
        db.close()
    rebased = root / "data/p2/assets" / str(generation) / "ocr_original.json"
    path = rebased if rebased.exists() else root / "evidence/p2/ocr_30.json"
    if json.loads(path.read_text())["generation"] != generation:
        raise ValueError("OCR artifact is stale; run attach-p2-assets after ingest")
    return path


def rebase_ocr(root: Path) -> dict:
    original = root / "evidence/p2/ocr_30.json"
    value = json.loads(original.read_text())
    db = connect(root / "data/p2/corpus.sqlite3")
    try:
        generation = active_id(db)
        for relative, expected_sha in db.execute(
            "SELECT relative_path,sha256 FROM sources WHERE generation=?", (generation,)
        ):
            if sha(root / "data/raw" / relative) != expected_sha:
                raise ValueError("OCR cannot be rebound: original manual source SHA changed")
        blocks = {
            bid: json.loads(raw)
            for bid, raw in db.execute(
                "SELECT block_id,payload FROM blocks WHERE generation=?", (generation,)
            )
        }
        for image in value["images"]:
            block = blocks.get(image["block_id"])
            if (
                not block
                or block["source_id"] != image["source_id"]
                or block["metadata"]["sha256"] != image["image_sha256"]
            ):
                raise ValueError("OCR cannot be rebound: source image changed or missing")
            if sha(Path(block["metadata"]["media_path"])) != image["image_sha256"]:
                raise ValueError("OCR cannot be rebound: image file SHA changed")
    finally:
        db.close()
    if generation != value["generation"]:
        value["rebased_from_generation"] = value["generation"]
        value["original_artifact_sha256"] = sha(original)
        value["generation"] = generation
        path = root / "data/p2/assets" / str(generation) / "ocr_original.json"
        write_json(path, value)
    return {
        "generation": generation,
        "original_ocr_sha256": sha(original),
        "images_revalidated": len(value["images"]),
    }


def apply_corrections(original: dict, ledger: dict) -> dict:
    if original["generation"] != ledger["generation"]:
        raise ValueError("Correction ledger and original OCR generations differ")
    corrected = copy.deepcopy(original)
    seen = set()
    for patch in ledger["patches"]:
        key = patch["screenshot"], patch["line"]
        if key in seen:
            raise ValueError("Duplicate correction target")
        seen.add(key)
        if not isinstance(patch["usable"], bool) or patch["disposition"] not in {
            "corrected",
            "report_false_positive",
            "quarantined",
        }:
            raise ValueError("Correction disposition or usability is invalid")
        if not 1 <= key[0] <= len(corrected["images"]) or not 1 <= key[1] <= len(
            corrected["images"][key[0] - 1]["lines"]
        ):
            raise ValueError("Correction target is outside OCR bounds")
        image = corrected["images"][key[0] - 1]
        line = image["lines"][key[1] - 1]
        if line["text"] != patch["before"]:
            raise ValueError("Correction before text differs from original OCR")
        if not isinstance(patch["after"], str) or not patch["after"].strip():
            raise ValueError("Correction must have nonblank exact replacement")
        line["text"] = patch["after"]
        line["correction_disposition"] = patch["disposition"]
        line["usable"] = patch["usable"]
        line["correction_reason"] = patch.get("reason", "原图与用户复核建议逐行核对")
    for item in ledger["occlusions"]:
        if not 1 <= item["screenshot"] <= len(corrected["images"]) or not 1 <= item["line"] <= len(
            corrected["images"][item["screenshot"] - 1]["lines"]
        ):
            raise ValueError("Occlusion target is outside OCR bounds")
        line = corrected["images"][item["screenshot"] - 1]["lines"][item["line"] - 1]
        line["usable"] = False
        line["correction_disposition"] = "quarantined"
        line["correction_reason"] = item["reason"]
    for index, image in enumerate(corrected["images"]):
        image["status"] = "ocr_assistant_corrected_unverified"
        image["human_review_status"] = "pending"
        image["assistant_visual_review_status"] = "completed"
        image["coverage"] = (
            "partially_occluded"
            if index + 1 in ledger["incomplete_images"]
            else "visible_screenshot_only"
        )
        for before, line in zip(original["images"][index]["lines"], image["lines"], strict=True):
            line["original_text"] = before["text"]
            line.setdefault("usable", True)
    for group in ledger.get("reconstructed_rows", []):
        number = group["screenshot"]
        if not isinstance(number, int) or not 1 <= number <= len(corrected["images"]):
            raise ValueError("Reconstructed row image is outside OCR bounds")
        image = corrected["images"][number - 1]
        indices = group["lines"]
        if (
            not indices
            or any(type(i) is not int or not 1 <= i <= len(image["lines"]) for i in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError("Reconstructed row requires ordered unique source lines")
        fragments = [image["lines"][i - 1] for i in indices]
        if any(not line["usable"] for line in fragments):
            raise ValueError("Quarantined text cannot enter a reconstructed row")
        if "".join(line["text"] for line in fragments) != group["text"]:
            raise ValueError("Reconstructed row differs from corrected source fragments")
        image.setdefault("reconstructed_rows", []).append(
            {**group, "boxes": [line["box"] for line in fragments]}
        )
    corrected["correction_version"] = ledger["version"]
    corrected["human_signoff"] = False
    return corrected


def repair_ocr(root: Path) -> dict:
    original_path = root / "evidence/p2/ocr_30.json"
    ledger_path = root / "evidence/p2/ocr_corrections.json"
    ledger = json.loads(ledger_path.read_text())
    if sha(original_path) != ledger["original_sha256"]:
        raise ValueError("Original OCR SHA differs from correction ledger")
    report = root / "evidence/p2/截图OCR复核报告_bf8fb923cd90036e501f.md"
    if sha(report) != ledger["review_report_sha256"]:
        raise ValueError("User review report changed since correction inspection")
    for binding in ledger.get("additional_review_reports", []):
        if sha(root / binding["path"]) != binding["sha256"]:
            raise ValueError("Additional review sheet changed since correction inspection")
    original = json.loads(original_path.read_text())
    corrected = apply_corrections(original, ledger)
    rebase = rebase_ocr(root)
    corrected["generation"] = rebase["generation"]
    corrected["original_artifact_sha256"] = ledger["original_sha256"]
    corrected["correction_ledger_sha256"] = sha(ledger_path)
    corrected["rebased_from_generation"] = original["generation"]
    path = root / "evidence/p2/ocr_corrected.json"
    write_json(path, corrected)
    db = connect(root / "data/p2/corpus.sqlite3")
    db.executescript(SCHEMA)
    try:
        with db:
            db.execute(
                "DELETE FROM auxiliary_items WHERE generation=? AND kind='ocr_corrected'",
                (corrected["generation"],),
            )
            db.execute(
                "DELETE FROM auxiliary_assets WHERE generation=? AND kind='ocr_corrected'",
                (corrected["generation"],),
            )
            for image in corrected["images"]:
                for index, line in enumerate(image["lines"]):
                    metadata = {
                        "image_block_id": image["block_id"],
                        "image_sha256": image["image_sha256"],
                        "locator": image["locator"],
                        "box": line["box"],
                        "score": line["score"],
                        "status": image["status"],
                        "human_review_status": "pending",
                        "original_text": line["original_text"],
                        "usable": line["usable"],
                        "coverage": image["coverage"],
                        "correction_ledger_sha256": corrected["correction_ledger_sha256"],
                    }
                    db.execute(
                        "INSERT INTO auxiliary_items VALUES(?,?,?,?,?,?,?)",
                        (
                            corrected["generation"],
                            "ocr_corrected",
                            image["source_id"],
                            f"{image['block_id']}:{index:04d}",
                            "manual_ocr_corrected_unverified"
                            if line["usable"]
                            else "manual_ocr_quarantined",
                            line["text"],
                            json.dumps(metadata, ensure_ascii=False),
                        ),
                    )
            db.execute(
                "INSERT INTO auxiliary_assets VALUES(?,?,?,?,?,?)",
                (
                    corrected["generation"],
                    "ocr_corrected",
                    "screenshot_set",
                    sha(path),
                    str(path),
                    sum(len(i["lines"]) for i in corrected["images"]),
                ),
            )
    finally:
        db.close()
    lines = [
        "# OCR 纠错副本逐行对照",
        "机器原始 OCR 保留不变；以下为助手图像核对后的副本，尚非人工签署的权威文本。",
    ]
    for index, image in enumerate(corrected["images"], 1):
        lines += [
            f"\n## 截图 {index:02d}",
            f"位置：`{image['locator']}`；覆盖：{image['coverage']}",
            "```text",
        ]
        lines += [
            f"{j:03d}{' [隔离]' if not line['usable'] else ''}: {line['text']}"
            for j, line in enumerate(image["lines"], 1)
        ]
        lines += ["```"]
        if image.get("reconstructed_rows"):
            lines += [
                "",
                "按原图恢复的同一配置行（OCR 段号可回查；不含编辑器行号）：",
                "```text",
                *[
                    f"{','.join(str(i) for i in row['lines'])}: {row['text']}"
                    for row in image["reconstructed_rows"]
                ],
                "```",
            ]
    (root / "evidence/p2/ocr_corrected_text.md").write_text("\n".join(lines) + "\n")
    summary = {
        **rebase,
        "review_findings": len(ledger["patches"]),
        "corrected_findings": sum(p["disposition"] == "corrected" for p in ledger["patches"]),
        "report_false_positives": sum(
            p["disposition"] == "report_false_positive" for p in ledger["patches"]
        ),
        "quarantined_lines": sum(
            not line["usable"] for i in corrected["images"] for line in i["lines"]
        ),
        "original_preserved": sha(original_path) == ledger["original_sha256"],
        "human_signoff": False,
        "corrected_artifact_sha256": sha(path),
    }
    write_json(root / "evidence/p2/ocr_repair_validation.json", summary)
    queue_path = root / "evidence/p2/image_review_queue.json"
    if queue_path.exists():
        record_ocr_results(queue_path, active_ocr_path(root))
        queue = json.loads(queue_path.read_text())
        by_id = {item["block_id"]: item for item in corrected["images"]}
        for item in queue["selected"]:
            image = by_id[item["block_id"]]
            item["assistant_visual_review_status"] = "completed"
            item["corrected_ocr_output"] = str(path)
            item["quarantined_lines"] = sum(not line["usable"] for line in image["lines"])
            item["coverage"] = image["coverage"]
        write_json(queue_path, queue)
    return summary
