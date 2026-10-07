import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from upf_agent.contracts import DocumentBlock
from upf_agent.corpus_db import active_id, connect
from upf_agent.review import _source_text, is_instruction

r = Path(__file__).resolve().parents[1]
c = connect(r / "data/p2/corpus.sqlite3")
g = active_id(c)
p = json.loads((r / "evidence/p2/structure_review_60.json").read_text())
assert g == p["generation"]
assert hashlib.sha256((r / "evidence/p2/structure_review_60.json").read_bytes()).hexdigest() == (
    "5ab8ba724456d1fa84ff5f0847cbc4149d87272d33f0cbf8d3895e23f7e1c0bb"
), "Frozen audit packet changed; re-review semantic annotations before running"
b = {
    x.block_id: x
    for row in c.execute("select payload from blocks where generation=?", (g,))
    if (x := DocumentBlock.model_validate_json(row[0]))
}
paths = {
    sid: relative
    for sid, relative in c.execute(
        "select source_id,relative_path from sources where generation=?", (g,)
    )
}
ordered = sorted(b.values(), key=lambda x: (x.source_id, x.ordinal))
issues = {
    1: "此行是 heredoc 起始命令，不能把单行当成完整脚本；须连同同节 EOF 与脚本主体查看。",
    6: "core_pattern 原文以单独 % 结尾；提取忠实保留，不能自动推断丢失的格式字符。",
    12: "原文末尾含中文括号说明“业务运行前配置”；当前属于候选语法，不可直接作为可执行命令。",
    36: (
        "原文将 NfDiscoveryReq Send 解释为 NF发现成功；"
        "请求发送与成功含义存在疑点，需业务复核原手册。"
    ),
    37: (
        "Initial ContextSetup Request 的中文列为下行RAN状态传递；"
        "疑似原手册指标映射问题，不改写原始文档。"
    ),
    49: "Unlock 操作原文写“需要锁定的User Name”；动作是解锁，文字有原手册表述疑点。",
}
rows = []
for i, item in enumerate(p["items"], 1):
    x = b[item["block_id"]]
    raw = r / "data/raw" / paths[x.source_id]
    assert _source_text(raw, x.locator) == x.raw_text
    if item["category"] == "step":
        assert is_instruction(x)
    neighbors = [
        z
        for z in ordered
        if z.source_id == x.source_id
        and abs(z.ordinal - x.ordinal) <= 2
        and z.block_id != x.block_id
    ]
    if item["category"] == "table":
        assert x.kind == "table_row"
    row = {
        **item,
        "sample_number": i,
        "full_text": x.raw_text,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "assistant_status": "source_requires_clarification"
        if i in issues
        else "faithful_extraction_and_category_checked",
        "assistant_note": issues.get(
            i, "已核对原件定位、字符保留、类别与邻近上下文；不代表现场可执行或人工业务验收。"
        ),
        "context": [{"locator": z.locator, "text": z.raw_text[:1200]} for z in neighbors],
        "human_status": "pending",
    }
    rows.append(row)
report = {
    "generation": g,
    "packet_sha256": hashlib.sha256(
        (r / "evidence/p2/structure_review_60.json").read_bytes()
    ).hexdigest(),
    "reviewer_type": "assistant_source_semantic_audit",
    "human_reviewed": 0,
    "assistant_checked": 60,
    "source_requires_clarification": len(issues),
    "items": rows,
}
(r / "evidence/p2/structure_audit.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
)
lines = [
    "# 60 条结构抽样核对",
    "",
    (
        f"当前 generation：`{g}`。20 命令、20 表行、20 操作步骤已进行助手原件/语义核对；"
        "人工签署仍为 0/60。"
    ),
    "原件疑点单独保留，不把忠实抽取解释成原手册事实无误。",
]
for x in rows:
    lines += [
        f"\n## {x['sample_number']:02d} · {x['category']}",
        f"原件：`{paths[x['source_id']]}`；定位：`{x['locator']}`",
        "```text",
        x["full_text"],
        "```",
        x["assistant_note"],
        "人工复核：待签署。",
    ]
(r / "evidence/p2/structure_audit.md").write_text("\n".join(lines) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "items"}, ensure_ascii=False))

# Independently check every merged cell against original OOXML properties and origin text.

namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
roots = {}
for sid, relative in paths.items():
    with ZipFile(r / "data/raw" / relative) as archive:
        roots[sid] = etree.fromstring(archive.read("word/document.xml"))


def xml_node(sid, locator):
    segments = locator.removeprefix("word/document.xml/body/").split("/")
    xpath = "/w:document/w:body/" + "/".join("w:" + segment for segment in segments)
    nodes = roots[sid].xpath(xpath, namespaces=namespace)
    assert len(nodes) == 1, locator
    return nodes[0]


def xml_cell_text(node):
    paragraphs = []
    for paragraph in node.findall("w:p", namespace):
        parts = []
        for item in paragraph.iter():
            tag = etree.QName(item).localname
            if tag in {"t", "instrText", "delText"}:
                parts.append(item.text or "")
            elif tag == "tab":
                parts.append("\t")
            elif tag in {"br", "cr"}:
                parts.append("\n")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


merged = []
for row in b.values():
    if row.kind != "table_row":
        continue
    column = 1
    for cell in row.metadata["cells"]:
        if cell["span"] > 1 or cell["vmerge"]:
            node = xml_node(row.source_id, cell["locator"])
            span_value = node.xpath("./w:tcPr/w:gridSpan/@w:val", namespaces=namespace)
            span = int(span_value[0]) if span_value else 1
            merge_nodes = node.xpath("./w:tcPr/w:vMerge", namespaces=namespace)
            merge = (
                (merge_nodes[0].get("{" + namespace["w"] + "}val") or "continue")
                if merge_nodes
                else None
            )
            assert cell["span"] == span and cell["column"] == column
            assert cell["vmerge"] == merge
            assert cell["raw_text"] == xml_cell_text(node)
            origin = xml_node(row.source_id, cell["origin"])
            assert cell["effective_text"] == xml_cell_text(origin)
            if merge == "continue":
                assert cell["origin"] != cell["locator"]
                assert origin.xpath("./w:tcPr/w:vMerge/@w:val", namespaces=namespace) == ["restart"]
            merged.append(
                {
                    "source_id": row.source_id,
                    "row_block_id": row.block_id,
                    **cell,
                    "original_xml_properties_and_origin_text_exact": True,
                }
            )
        column += cell["span"]
merge_report = {
    "generation": g,
    "checked_cells": len(merged),
    "horizontal_span_cells": sum(x["span"] > 1 for x in merged),
    "vertical_starts": sum(x["vmerge"] == "restart" for x in merged),
    "vertical_continuations": sum(x["vmerge"] == "continue" for x in merged),
    "human_signoff": False,
    "scope": "all_native_merged_cells_original_xml_machine_check",
    "cells": merged,
}
(r / "evidence/p2/merged_cells_audit.json").write_text(
    json.dumps(merge_report, ensure_ascii=False, indent=2) + "\n"
)
print(
    json.dumps(
        {key: value for key, value in merge_report.items() if key != "cells"}, ensure_ascii=False
    )
)
