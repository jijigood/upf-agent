"""Bounded original-text answers for unique, scoped native manual sections."""

from __future__ import annotations

import re
from dataclasses import dataclass

from upf_agent.p3_chunks import RetrievalChunk
from upf_agent.p3_lexical import scope_allows
from upf_agent.p3_sections import SectionHint, belongs, related_section_routes, resolve_section


@dataclass(frozen=True)
class NativeSectionExtracts:
    source_id: str
    path: tuple[str, ...]
    method: str
    extracts: list[tuple[RetrievalChunk, str]]


def interface_conflicts(native: NativeSectionExtracts) -> list[dict[str, str]]:
    """Flag explicit interface mismatches in a case's solution prose without correcting it."""
    title = native.path[-1]
    title_interfaces = set(re.findall(r"(?<![A-Za-z0-9])N[3469](?![A-Za-z0-9])", title, re.I))
    if len(title_interfaces) != 1:
        return []
    expected = next(iter(title_interfaces)).upper()
    conflicts = []
    seen = set()
    pattern = re.compile(r"排查交换机接口有无收到\s*(N[3469])\s*接口的?数据报文", re.I)
    for chunk, _ in native.extracts:
        if chunk.chunk_id in seen or chunk.section_path[-1] != "解决方法":
            continue
        seen.add(chunk.chunk_id)
        for match in pattern.finditer(chunk.text):
            observed = match.group(1).upper()
            if observed != expected:
                conflicts.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "title_interface": expected,
                        "solution_interface": observed,
                        "source_quote": match.group(0),
                    }
                )
    return conflicts


def empty_collection_sections(native: NativeSectionExtracts) -> list[dict[str, str]]:
    """Identify information-collection headings with no source body or child blocks."""
    by_path: dict[tuple[str, ...], list[RetrievalChunk]] = {}
    for chunk, _ in native.extracts:
        group = by_path.setdefault(tuple(chunk.section_path), [])
        if all(existing.chunk_id != chunk.chunk_id for existing in group):
            group.append(chunk)
    result = []
    for path, group in by_path.items():
        if "信息收集" not in path[-1]:
            continue
        if any(len(other) > len(path) and other[: len(path)] == path for other in by_path):
            continue
        bodies = [" ".join(chunk.text.partition("\n")[2].split()) for chunk in group]
        if all(not body or re.fullmatch(r"(?:\d+\.)*\s*信息收集", body) for body in bodies):
            result.append({"chunk_id": group[0].chunk_id, "section_title": path[-1]})
    return result


def native_source_gap_notes(native: NativeSectionExtracts) -> list[str]:
    """Describe omissions explicit in a complete native section without inventing values."""
    body = "\n".join(quote for _, quote in native.extracts)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    notes: list[str] = []
    if any(
        line == "功能描述" and next_line in {"功能配置", "操作步骤：", "参数配置"}
        for line, next_line in zip(lines, lines[1:], strict=False)
    ):
        notes.append("原件的“功能描述”只有标题、没有说明正文，不能补造功能用途。")
    headers: list[str] = []
    for line in lines:
        if line.startswith("表头："):
            headers = [field.strip() for field in line.removeprefix("表头：").split("\t")]
        elif headers and line.startswith("参数名称："):
            values = {
                field.split("：", 1)[0].strip()
                for field in line.split(" | ")
                if "：" in field
            }
            missing = [field for field in headers if field not in values]
            if missing:
                parameter = line.split(" | ", 1)[0].removeprefix("参数名称：")
                notes.append(
                    f"原件参数“{parameter}”的{'、'.join(missing)}栏为空，不能补造。"
                )
    if "如下图" in body:
        notes.append("原文引用图示；当前原生文字不能确认图中未写出的界面字段或布局。")
    if any("故障案例" in part for part in native.path):
        notes.append("这是手册中的历史故障案例，不代表当前设备正在发生同一故障。")
    return list(dict.fromkeys(notes))


def adjacent_case_overlap(
    native: NativeSectionExtracts, chunks: dict[str, RetrievalChunk]
) -> list[dict[str, str]]:
    """Expose a nearby case's different error without equating the two incidents."""
    if len(native.path) < 2 or "故障案例" not in native.path[0]:
        return []
    own_text = "\n".join(quote for _, quote in native.extracts)
    config = re.search(r"/root/[^\s'\"]+\.conf", own_text)
    if not (config and "No such file or directory" in own_text and "upfmgr hc" in own_text):
        return []
    first_ordinal = min(chunk.ordinal_start for chunk, _ in native.extracts)
    candidates = sorted(
        (
            chunk
            for chunk in chunks.values()
            if chunk.source_id == native.source_id
            and len(chunk.section_path) >= 2
            and chunk.section_path[0] == native.path[0]
            and chunk.section_path[1] != native.path[1]
            and "upfmgr hc" in chunk.section_path[1]
            and "Get GTPU Node IP Failed!" in chunk.text
            and abs(chunk.ordinal_start - first_ordinal) <= 100
        ),
        key=lambda chunk: (abs(chunk.ordinal_start - first_ordinal), chunk.chunk_id),
    )
    if not candidates:
        return []
    neighbor = candidates[0]
    return [
        {
            "neighbor_chunk_id": neighbor.chunk_id,
            "neighbor_title": neighbor.section_path[1],
            "note": (
                f"邻近手册案例“{neighbor.section_path[1]}”也使用 upfmgr hc，"
                f"但本节记录配置文件“{config.group(0)}”不存在，"
                "邻节记录“Get GTPU Node IP Failed!”和N3/N6地址异常；"
                "原件未确认两节属于同一次故障，配置路径、报错与时间须分别核对。"
            ),
        }
    ]


def _units(raw: str, limit: int = 1200) -> list[str]:
    result: list[str] = []
    current = ""
    for line in raw.splitlines(keepends=True):
        while line:
            space = limit - len(current)
            if not space:
                if current.strip():
                    result.append(current)
                current, space = "", limit
            if len(line) <= space:
                current += line
                line = ""
            elif current:
                if current.strip():
                    result.append(current)
                current = ""
            else:
                result.append(line[:limit])
                line = line[limit:]
    if current.strip():
        result.append(current)
    return result


def native_section_extracts(
    question: str,
    chunks: dict[str, RetrievalChunk],
    product: str | None,
    version: str | None,
    total_budget: int,
) -> NativeSectionExtracts | None:
    """Return the whole small chapter, never an arbitrary prefix or guessed section."""
    historical_case = bool(re.search(r"症状|信息收集|处理建议|排查依据|处理前提", question))
    function_review = bool(re.search(r"说明|功能|配置项|参数|描述|适用范围", question))
    if not (historical_case or function_review):
        return None
    if "只摘录原文" in question or re.search(r"执行该命令|现场实时|当前状态是什么", question):
        return None
    hint = resolve_section(question, chunks, product, version)
    if "手册" not in question and not (
        historical_case and hint is not None and any("故障案例" in part for part in hint.path)
    ):
        return None
    if hint is None and function_review and not historical_case:
        _, candidates = related_section_routes(question, {}, chunks, product, version, 32)
        with_tables = [
            candidate
            for candidate in candidates
            if any(chunk.kind == "table" and belongs(chunk, candidate) for chunk in chunks.values())
        ]
        if len(with_tables) == 1:
            chosen = with_tables[0]
            hint = SectionHint(
                chosen.source_id, chosen.path, "same_name_native_table_disambiguation"
            )
    if hint is None:
        return None
    selected = sorted(
        (
            chunk
            for chunk in chunks.values()
            if belongs(chunk, hint) and scope_allows(chunk, product, version)
        ),
        key=lambda chunk: (chunk.ordinal_start, chunk.chunk_id),
    )
    if not selected or sum(chunk.token_count for chunk in selected) > total_budget:
        return None
    extracts = [
        (chunk, unit) for chunk in selected for unit in _units(chunk.text.split("\n", 1)[-1])
    ]
    if not extracts or len(extracts) > 32:
        return None
    return NativeSectionExtracts(hint.source_id, hint.path, hint.method, extracts)
