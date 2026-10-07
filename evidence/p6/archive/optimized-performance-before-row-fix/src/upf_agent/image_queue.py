"""Prioritize screenshot evidence without promoting OCR text to manual authority."""

import json
import re
from pathlib import Path

from upf_agent.contracts import DocumentBlock
from upf_agent.corpus_db import active_id, connect

KEYWORDS = re.compile(
    r"命令|参数|步骤|配置|查询|查看|登录|故障|报错|告警|界面|截图|UPF|N4|PFCP", re.I
)


def build_queue(db_path: Path, output: Path, limit: int = 30) -> list[dict]:
    db = connect(db_path)
    try:
        generation = active_id(db)
        if generation is None:
            raise ValueError("No active corpus generation")
        candidates = []
        seen = set()
        for (payload,) in db.execute(
            "SELECT payload FROM blocks WHERE generation=? AND kind='image'", (generation,)
        ):
            block = DocumentBlock.model_validate_json(payload)
            meta = block.metadata
            sha = str(meta["sha256"])
            if sha in seen:
                continue
            seen.add(sha)
            width, height = meta.get("width"), meta.get("height")
            if not isinstance(width, int) or not isinstance(height, int):
                continue
            if width < 450 or height < 160:
                continue
            context = " ".join([*block.section_path, str(meta.get("context", ""))])
            score = min(width * height / 250000, 5) + 2 * len(KEYWORDS.findall(context))
            candidates.append(
                {
                    "source_id": block.source_id,
                    "block_id": block.block_id,
                    "locator": block.locator,
                    "section_path": block.section_path,
                    "context": meta.get("context", ""),
                    "sha256": sha,
                    "path": meta["media_path"],
                    "width": width,
                    "height": height,
                    "priority": round(score, 2),
                    "ocr_status": "pending",
                    "human_review_status": "pending",
                    "review_fields": [
                        "大小写",
                        "冒号/等号",
                        "连字符/下划线",
                        "I/l/1",
                        "O/0",
                        "IP地址",
                        "端口",
                    ],
                }
            )
        candidates.sort(key=lambda item: (-item["priority"], item["source_id"], item["locator"]))
        selected = candidates[:limit]
        if len(selected) < limit:
            raise ValueError(f"Only {len(selected)} screenshot candidates for requested {limit}")
        if output.is_file():
            previous = json.loads(output.read_text(encoding="utf-8"))
            if previous.get("generation") == generation:
                old_by_id = {item["block_id"]: item for item in previous.get("selected", [])}
                for item in selected:
                    old = old_by_id.get(item["block_id"], {})
                    for key in (
                        "ocr_status",
                        "human_review_status",
                        "ocr_line_count",
                        "ocr_output",
                        "assistant_visual_review_status",
                        "corrected_ocr_output",
                        "quarantined_lines",
                        "coverage",
                    ):
                        if key in old:
                            item[key] = old[key]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "generation": generation,
                    "selected": selected,
                    "eligible_unique": len(candidates),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return selected
    finally:
        db.close()


def record_ocr_results(queue_path: Path, ocr_path: Path) -> dict:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    result = json.loads(ocr_path.read_text(encoding="utf-8"))
    if queue["generation"] != result["generation"]:
        raise ValueError("OCR result belongs to a different corpus generation")
    by_id = {item["block_id"]: item for item in result["images"]}
    if len(by_id) != len(queue["selected"]):
        raise ValueError("OCR result count does not match review queue")
    for item in queue["selected"]:
        ocr = by_id[item["block_id"]]
        if ocr["image_sha256"] != item["sha256"]:
            raise ValueError(f"OCR image SHA mismatch: {item['block_id']}")
        item["ocr_status"] = "completed_unverified"
        item["ocr_line_count"] = len(ocr["lines"])
        item["ocr_output"] = str(ocr_path)
    queue_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "images": len(by_id),
        "ocr_lines": sum(len(x["lines"]) for x in result["images"]),
        "human_reviewed": 0,
    }
