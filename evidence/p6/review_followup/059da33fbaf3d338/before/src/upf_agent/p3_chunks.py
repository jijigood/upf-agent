"""Source-preserving structural chunks for the three P3 indexes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import Field
from tokenizers import Tokenizer

from upf_agent.contracts import CommandRecord, DocumentBlock, StrictModel
from upf_agent.corpus_db import active_id, connect

CHUNKER_VERSION = "structural-v1"
JWT = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{20,}(?:\.[A-Za-z0-9_-]{15,})?"
)
INTERFACE = re.compile(r"(?<![A-Za-z0-9])(?:N3|N4|N6|PFCP|GTP-U|GRE|SMF|AMF)(?![A-Za-z0-9])", re.I)


class RetrievalChunk(StrictModel):
    chunk_id: str
    source_id: str
    source_sha256: str
    relative_path: str
    role: Literal["manual"] = "manual"
    product: str | None = None
    version: str | None = None
    section_path: list[str] = Field(default_factory=list)
    parent_id: str
    kind: Literal["prose", "code", "table", "command"]
    text: str
    token_count: int
    block_ids: list[str]
    locators: list[str]
    ordinal_start: int
    ordinal_end: int
    extraction_methods: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    interface_tags: list[str] = Field(default_factory=list)
    command_id: str | None = None
    command_family: str | None = None
    table_id: str | None = None
    table_header: str | None = None
    table_cells: list[dict] = Field(default_factory=list)


@dataclass
class Unit:
    kind: str
    block: DocumentBlock
    text: str
    table_id: str | None = None
    table_header: str | None = None
    table_cells: list[dict] = field(default_factory=list)
    command: CommandRecord | None = None
    flags: list[str] = field(default_factory=list)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _redact(value: str) -> tuple[str, list[str]]:
    replaced, count = JWT.subn("[REDACTED_JWT]", value)
    return replaced, ["sensitive_token_redacted"] if count else []


def _section_parent(source_id: str, section: list[str]) -> str:
    return "section-" + _sha(json.dumps([source_id, section], ensure_ascii=False))[:20]


def _table_row_text(row: DocumentBlock, header: DocumentBlock | None) -> tuple[str, list[dict]]:
    header_cells = header.metadata["cells"] if header else []
    header_for_col = {}
    for cell in header_cells:
        for col in range(cell["column"], cell["column"] + cell["span"]):
            header_for_col[col] = str(cell["effective_text"]).strip()
    parts, coordinates = [], []
    for cell in row.metadata["cells"]:
        value = str(cell["effective_text"]).strip()
        if not value:
            continue
        label = header_for_col.get(cell["column"], "")
        label = label[:60] if label else f"列{cell['column']}"
        parts.append(f"{label}：{value}")
        coordinates.append(
            {
                "row": row.metadata["row_index"],
                "column": cell["column"],
                "span": cell["span"],
                "origin": cell["origin"],
                "locator": cell["locator"],
                "vmerge": cell["vmerge"],
            }
        )
    return " | ".join(parts), coordinates


def _split_preserving_text(text: str, tokenizer: Tokenizer, cap: int) -> list[str]:
    if len(tokenizer.encode(text, add_special_tokens=False).ids) <= cap:
        return [text]
    parts = re.split(r"(?<=[。！？；.!?;\n])", text)
    output, current = [], ""
    for part in parts:
        if not part:
            continue
        proposed = current + part
        if current and len(tokenizer.encode(proposed, add_special_tokens=False).ids) > cap:
            output.append(current)
            current = ""
        if len(tokenizer.encode(part, add_special_tokens=False).ids) <= cap:
            current += part
            continue
        encoding = tokenizer.encode(part, add_special_tokens=False)
        start = 0
        while start < len(encoding.ids):
            end = min(start + cap, len(encoding.ids))
            char_start = encoding.offsets[start][0]
            char_end = encoding.offsets[end - 1][1]
            piece = part[char_start:char_end]
            if piece:
                output.append(piece)
            start = end
    if current:
        output.append(current)
    return output


def _load_p2(root: Path) -> tuple[str, list[DocumentBlock], list[CommandRecord], dict]:
    db = connect(root / "data/p2/corpus.sqlite3")
    try:
        generation = active_id(db)
        if not generation:
            raise ValueError("P2 corpus has no active generation")
        blocks = [
            DocumentBlock.model_validate_json(raw)
            for (raw,) in db.execute(
                "SELECT payload FROM blocks WHERE generation=? ORDER BY source_id,ordinal",
                (generation,),
            )
        ]
        commands = [
            CommandRecord.model_validate_json(raw)
            for (raw,) in db.execute(
                "SELECT payload FROM commands WHERE generation=?", (generation,)
            )
        ]
        sources = {
            source_id: {
                "sha256": sha,
                "relative_path": relative,
                "product": product,
                "version": version,
            }
            for source_id, sha, relative, product, version in db.execute(
                "SELECT source_id,sha256,relative_path,product,version FROM sources "
                "WHERE generation=? AND role='manual'",
                (generation,),
            )
        }
        return generation, blocks, commands, sources
    finally:
        db.close()


def build_chunks(
    root: Path, tokenizer_path: Path, target_tokens: int = 600, max_tokens: int = 800
) -> tuple[str, list[RetrievalChunk]]:
    if not 100 <= target_tokens <= max_tokens:
        raise ValueError("invalid chunk token budget")
    generation, blocks, commands, sources = _load_p2(root)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    by_id = {block.block_id: block for block in blocks}
    command_by_block = {c.syntax_block_id: c for c in commands if c.syntax_block_id}
    rows_by_table: dict[str, list[DocumentBlock]] = {}
    for block in blocks:
        if block.kind == "table_row" and block.parent_id:
            rows_by_table.setdefault(block.parent_id, []).append(block)
    units: list[Unit] = []
    for block in blocks:
        if block.kind == "paragraph":
            if not block.raw_text.strip() or "table_of_contents" in block.quality_flags:
                continue
            if block.parent_id and by_id[block.parent_id].kind == "table":
                continue
            command = command_by_block.get(block.block_id)
            kind = (
                "command"
                if command
                else (
                    "code"
                    if str(block.metadata.get("style_name", "")).lower()
                    in {"cmd", "melo-codeblock"}
                    or "\n" in block.raw_text
                    else "prose"
                )
            )
            text = block.raw_text
            if command:
                related = []
                for related_id in (
                    command.precondition_block_ids
                    + command.parameter_block_ids
                    + command.example_block_ids
                ):
                    if related_id in by_id:
                        related.append(by_id[related_id].raw_text)
                if related:
                    text += "\n相关说明：" + "\n".join(related)
            text, flags = _redact(text)
            units.append(Unit(kind, block, text, command=command, flags=flags))
        elif block.kind == "table_row" and block.raw_text.strip():
            rows = rows_by_table.get(block.parent_id or "", [])
            header = rows[0] if rows else None
            # A one-row table still needs its own retrievable chunk.
            if header is block and len(rows) > 1:
                continue
            row_text, coordinates = _table_row_text(block, header if header is not block else None)
            title = " / ".join(block.section_path[-2:])
            text, flags = _redact(
                f"表格：{title}\n表头：{header.raw_text if header else ''}\n{row_text}"
            )
            units.append(
                Unit(
                    "table",
                    block,
                    text,
                    block.parent_id,
                    header.raw_text if header else None,
                    coordinates,
                    flags=flags,
                )
            )
    units.sort(key=lambda u: (u.block.source_id, u.block.ordinal))
    chunks: list[RetrievalChunk] = []
    batch: list[Unit] = []

    def flush() -> None:
        if not batch:
            return
        first, last = batch[0].block, batch[-1].block
        section = first.section_path
        heading = " / ".join(section[-3:])
        body = "\n".join(unit.text for unit in batch)
        text = f"{heading}\n{body}" if heading else body
        token_count = len(tokenizer.encode(text, add_special_tokens=False).ids)
        if token_count > max_tokens:
            raise ValueError(f"chunk over token budget: {first.block_id} {token_count}")
        block_ids = [unit.block.block_id for unit in batch]
        command = batch[0].command if len(batch) == 1 else None
        source = sources[first.source_id]
        key = json.dumps([generation, first.source_id, block_ids, text], ensure_ascii=False)
        chunk = RetrievalChunk(
            chunk_id="chunk-" + _sha(key)[:24],
            source_id=first.source_id,
            source_sha256=source["sha256"],
            relative_path=source["relative_path"],
            product=source["product"],
            version=source["version"],
            section_path=section,
            parent_id=_section_parent(first.source_id, section),
            kind=batch[0].kind,
            text=text,
            token_count=token_count,
            block_ids=block_ids,
            locators=[unit.block.locator for unit in batch],
            ordinal_start=first.ordinal,
            ordinal_end=last.ordinal,
            extraction_methods=sorted({unit.block.extraction_method for unit in batch}),
            quality_flags=sorted(
                {
                    flag
                    for unit in batch
                    for flag in (
                        unit.block.quality_flags
                        + unit.flags
                        + (unit.command.quality_flags if unit.command else [])
                    )
                }
            ),
            interface_tags=sorted(set(INTERFACE.findall(text)), key=str.upper),
            command_id=command.command_id if command else None,
            command_family=command.family if command else None,
            table_id=batch[0].table_id,
            table_header=batch[0].table_header,
            table_cells=[cell for unit in batch for cell in unit.table_cells],
        )
        chunks.append(chunk)
        batch.clear()

    for unit in units:
        heading = " / ".join(unit.block.section_path[-3:])
        cap = max_tokens - len(tokenizer.encode(heading, add_special_tokens=False).ids) - 20
        fragments = _split_preserving_text(unit.text, tokenizer, max(50, cap))
        for fragment in fragments:
            current = Unit(
                unit.kind,
                unit.block,
                fragment,
                unit.table_id,
                unit.table_header,
                unit.table_cells,
                unit.command,
                unit.flags,
            )
            if batch:
                same_group = (
                    batch[0].block.source_id == current.block.source_id
                    and batch[0].block.section_path == current.block.section_path
                    and batch[0].kind == current.kind
                    and batch[0].table_id == current.table_id
                    and current.kind != "command"
                )
                proposed = (
                    " / ".join(current.block.section_path[-3:])
                    + "\n"
                    + "\n".join(x.text for x in [*batch, current])
                )
                if (
                    not same_group
                    or len(tokenizer.encode(proposed, add_special_tokens=False).ids) > target_tokens
                ):
                    flush()
            batch.append(current)
            if current.kind == "command":
                flush()
    flush()
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("duplicate chunk ids")
    return generation, chunks
