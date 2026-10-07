"""Bounded native extracts for explicit chapter/command quotation requests."""

from __future__ import annotations

import re
from dataclasses import dataclass

from upf_agent.p3_chunks import RetrievalChunk
from upf_agent.p3_lexical import scope_allows
from upf_agent.p3_sections import belongs, resolve_section

PROMPT = re.compile(
    r"^\s*(?:\[[^\]\n]{1,80}\][#$]|[\w.-]+@[\w.-]+:[^\n#$]{0,160}[#$]|[\w.-]+[>#]|[#$])\s*"
)
TOKEN = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_.-]*(?![A-Za-z0-9_-])")


@dataclass(frozen=True)
class LiteralExtracts:
    intent: str
    command_family: str
    extracts: list[tuple[RetrievalChunk, str]]


def literal_extracts(
    question: str,
    chunks: dict[str, RetrievalChunk],
    product: str | None,
    version: str | None,
    total_budget: int,
) -> LiteralExtracts | None:
    if not re.search(r"原文写法|请引用|原始语法|只摘录原文|给出原文|核对.{0,80}语法", question):
        return None
    if re.search(r"执行|实时|当前|根因|原因|是否", question):
        return None
    hint = resolve_section(question, chunks, product, version)
    if hint is not None and hint.method.startswith("explicit_section"):
        section = [
            chunk
            for chunk in chunks.values()
            if belongs(chunk, hint) and scope_allows(chunk, product, version)
        ]
    else:
        quoted = re.search(r"[‘“]([^’”]+)[’”]", question)
        parts = [part.strip() for part in quoted.group(1).split(" / ")] if quoted else []
        if len(parts) < 2 or any(not part for part in parts):
            return None
        section = [
            chunk
            for chunk in chunks.values()
            if tuple(chunk.section_path[-len(parts) :]) == tuple(parts)
            and scope_allows(chunk, product, version)
        ]
        if len({chunk.source_id for chunk in section}) != 1:
            return None
    section.sort(key=lambda chunk: (chunk.ordinal_start, chunk.chunk_id))
    remaining = re.sub(r"‘[^’]+’|“[^”]+”", "", question)
    asked = {match.group().casefold() for match in TOKEN.finditer(remaining)}
    matched = []
    families = set()
    for chunk in section:
        if chunk.kind != "command" or "\n" not in chunk.text:
            continue
        raw = chunk.text.split("\n", 1)[1]
        body = PROMPT.sub("", raw)
        first = re.match(r"[A-Za-z][A-Za-z0-9_.-]*", body)
        if first and first.group().casefold() in asked:
            matched.append(chunk)
            families.add(first.group().casefold())
    if not matched or len(families) != 1:
        return None
    intent = (
        "section_context"
        if re.search(r"只摘录原文|适用范围|示例值|现场值", question)
        else "syntax"
    )
    selected = matched
    if intent == "section_context":
        matched_ids = {chunk.chunk_id for chunk in matched}
        command_parameters = {
            token.group().casefold()
            for chunk in matched
            for token in TOKEN.finditer(chunk.text.split("\n", 1)[-1])
            if "-" in token.group() or "_" in token.group()
        }
        selected = []
        for chunk in section:
            if chunk.chunk_id in matched_ids:
                selected.append(chunk)
                continue
            if chunk.kind != "prose":
                continue
            raw = chunk.text.split("\n", 1)[-1]
            if not raw.strip():
                continue
            # Only a short, directly applicable prerequisite or parameter
            # explanation can accompany the named command family.
            prerequisite = bool(re.match(r"^(?:前提|条件|要求)[：:]", raw.strip()))
            parameters = {token.group().casefold() for token in TOKEN.finditer(raw)}
            explains = bool(command_parameters & parameters) and bool(
                re.match(r"^(?:说明|参数|取值)[：:]", raw.strip())
            )
            if (prerequisite or explains) and not re.search(r"(?m)^\s*[#$]\s*\w+", raw):
                selected.append(chunk)
    if sum(chunk.token_count for chunk in selected) > total_budget:
        return None
    extracts = []
    for chunk in selected:
        raw = chunk.text.split("\n", 1)[1] if "\n" in chunk.text else ""
        if not raw.strip():
            continue
        for offset in range(0, len(raw), 1200):
            if raw[offset : offset + 1200].strip():
                extracts.append((chunk, raw[offset : offset + 1200]))
    if not extracts or len(extracts) > 32:
        return None
    return LiteralExtracts(intent, next(iter(families)), extracts)
