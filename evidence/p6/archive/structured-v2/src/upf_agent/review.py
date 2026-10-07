"""Deterministic P2 review packet; human sign-off remains explicit."""

import hashlib
import json
import re
import shutil
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from upf_agent.contracts import DocumentBlock
from upf_agent.corpus_db import active_id, connect
from upf_agent.docx_parse import DocxParser, local

REVIEW_SELECTION_VERSION = "review-selection-v3"


def is_instruction(block: DocumentBlock) -> bool:
    text = block.raw_text.strip()
    if block.kind != "paragraph" or block.metadata.get("heading_level"):
        return False
    if text in {"功能描述", "解决方法", "信息收集", "操作步骤", "功能配置"}:
        return False
    if re.match(r"^[\d.]+(?:\s|$)", text) or "No such file or directory" in text:
        return False
    if block.section_path and text == block.section_path[-1]:
        return False
    if re.search(r"HYPERLINK|PAGEREF|^图\s*\d|^检查结果|<<<|^\s*\w+\s*:", text):
        return False
    return len(text) > 8 and bool(
        re.search(
            r"^步骤[：:]|^(?:首先|然后|最后|请)?"
            r"(?:登录|点击|选择|执行|输入|打开|检查|上传|下载|手动修改|复制并输入)"
            r"|点击[“\"]|-->点击",
            text,
        )
    )


def _source_text(raw_path: Path, locator: str) -> str:
    """Independently resolve a persisted body locator against the original OOXML."""
    with ZipFile(raw_path) as zf:
        root = etree.fromstring(zf.read("word/document.xml"))
    parser = DocxParser.__new__(DocxParser)
    parser.compatibility = {"alternate_content": 0, "choice": 0, "fallback": 0}
    parser._compat(root)
    node = next(x for x in root if isinstance(x.tag, str) and local(x) == "body")
    path = locator.removeprefix("word/document.xml/body/")
    for part in path.split("/"):
        match = re.fullmatch(r"([A-Za-z]+)\[(\d+)\]", part)
        if not match:
            raise ValueError(f"Invalid locator segment: {part}")
        name, index = match.group(1), int(match.group(2))
        if name == "txbxContent":
            choices = [
                x for x in node.iterdescendants() if isinstance(x.tag, str) and local(x) == name
            ]
        else:
            choices = [x for x in node if isinstance(x.tag, str) and local(x) == name]
        node = choices[index - 1]
    if local(node) == "tr":
        values = []
        for cell in [x for x in node if isinstance(x.tag, str) and local(x) == "tc"]:
            values.append(
                "\n".join(
                    parser._text(p) for p in cell if isinstance(p.tag, str) and local(p) == "p"
                )
            )
        return "\t".join(values)
    return parser._text(node)


def make_review_packet(db_path: Path, output: Path, source_root: Path | None = None) -> dict:
    db = connect(db_path)
    try:
        generation = active_id(db)
        if generation is None:
            raise ValueError("No active generation")
        if output.exists():
            previous = json.loads(output.read_text(encoding="utf-8"))
            if (
                previous["generation"] == generation
                and previous.get("selection_version") == REVIEW_SELECTION_VERSION
            ):
                return {
                    "generation": generation,
                    "items": len(previous["items"]),
                    "human_reviewed": sum(
                        i["human_status"] == "completed" for i in previous["items"]
                    ),
                }
            archive = output.parent / "archive" / previous["generation"]
            archive.mkdir(parents=True, exist_ok=True)
            target = archive / output.name
            if target.exists() and target.read_bytes() != output.read_bytes():
                target = (
                    archive
                    / f"{output.stem}-{hashlib.sha256(output.read_bytes()).hexdigest()[:12]}.json"
                )
            shutil.copy2(output, target)
        blocks = {
            b.block_id: b
            for (raw,) in db.execute("SELECT payload FROM blocks WHERE generation=?", (generation,))
            if (b := DocumentBlock.model_validate_json(raw))
        }
        commands = [
            json.loads(raw)
            for (raw,) in db.execute(
                "SELECT payload FROM commands WHERE generation=? ORDER BY source_id, command_id",
                (generation,),
            )
        ]
        paths = {
            source_id: relative_path
            for source_id, relative_path in db.execute(
                "SELECT source_id,relative_path FROM sources WHERE generation=?", (generation,)
            )
        }
        command_items = []
        for command in commands:
            block = blocks[command["syntax_block_id"]]
            if block.raw_text.strip() != command["syntax"]:
                raise ValueError(f"Command differs from source block: {command['command_id']}")
            command_items.append(
                {
                    "category": "command",
                    "source_id": block.source_id,
                    "block_id": block.block_id,
                    "locator": block.locator,
                    "section_path": block.section_path,
                    "excerpt": block.raw_text[:300],
                    "source_xml_exact": True,
                    "human_status": "pending",
                }
            )
        tables = [b for b in blocks.values() if b.kind == "table_row" and b.raw_text.strip()]
        table_items = [
            {
                "category": "table",
                "source_id": b.source_id,
                "block_id": b.block_id,
                "locator": b.locator,
                "section_path": b.section_path,
                "excerpt": b.raw_text[:300],
                "cells": b.metadata.get("cells", []),
                "human_status": "pending",
            }
            for b in tables
        ]
        steps = [b for b in blocks.values() if is_instruction(b)]
        step_items = [
            {
                "category": "step",
                "source_id": b.source_id,
                "block_id": b.block_id,
                "locator": b.locator,
                "section_path": b.section_path,
                "excerpt": b.raw_text[:300],
                "human_status": "pending",
            }
            for b in steps
        ]

        def spaced(items: list[dict], count: int) -> list[dict]:
            if len(items) < count:
                raise ValueError(f"Only {len(items)} items for review sample of {count}")
            return [items[i * len(items) // count] for i in range(count)]

        selected = spaced(command_items, 20) + spaced(table_items, 20) + spaced(step_items, 20)
        if source_root is not None:
            for item in selected:
                block = blocks[item["block_id"]]
                original = _source_text(source_root / paths[block.source_id], block.locator)
                item["source_xml_exact"] = original == block.raw_text
                if not item["source_xml_exact"]:
                    raise ValueError(f"Source XML mismatch: {block.block_id} {block.locator}")
        packet = {
            "generation": generation,
            "selection_version": REVIEW_SELECTION_VERSION,
            "counts": {"command": 20, "table": 20, "step": 20},
            "human_reviewed": 0,
            "items": selected,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"generation": generation, "items": len(selected), "human_reviewed": 0}
    finally:
        db.close()


def write_review_sheet(
    packet_path: Path, image_queue_path: Path, sources_path: Path, output: Path
) -> Path:
    """Make a human-friendly checklist without overwriting prior annotations."""
    if output.exists():
        return output
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    queue = json.loads(image_queue_path.read_text(encoding="utf-8"))
    paths = {
        item["source_id"]: item["relative_path"]
        for item in json.loads(sources_path.read_text(encoding="utf-8"))
    }
    lines = [
        f"# P2 人工复核单 · {packet['generation']}",
        "",
        "请对照本地 `data/raw/` 原件和截图检查；机器 XML 回查已通过，不代表人工确认。",
        "在每项下填写通过/不通过/无法确认、纠错原文、复核人和日期；通过才勾选。",
        "只需填写本文件，JSON 状态由程序维护者同步；本步骤不执行手册中的命令。",
        "",
    ]
    for category, label in (("command", "命令"), ("table", "表行"), ("step", "操作步骤")):
        lines.extend([f"## {label}（20 条）", ""])
        for index, item in enumerate((x for x in packet["items"] if x["category"] == category), 1):
            source_label = paths[item["source_id"]]
            lines.extend(
                [
                    f"- [ ] {label} {index:02d} · `{source_label}` · `{item['locator']}`",
                    f"  - 章节：{' / '.join(item['section_path'])}",
                    f"  - 原文：{item['excerpt'].replace(chr(10), ' / ')}",
                    "  - 复核记录：",
                    "",
                ]
            )
    lines.extend(["## 截图 OCR（30 张）", ""])
    for index, item in enumerate(queue["selected"], 1):
        review_fields = "、".join(item["review_fields"])
        lines.extend(
            [
                f"- [ ] 截图 {index:02d} · `{paths[item['source_id']]}` · `{item['locator']}`",
                f"  - [查看原图]({item['path']})；章节：{' / '.join(item['section_path'])}",
                f"  - OCR 行数：{item.get('ocr_line_count', '待处理')}；重点核对：{review_fields}",
                "  - 复核记录：",
                "",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
