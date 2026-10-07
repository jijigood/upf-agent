"""OOXML extraction with stable XML locators and one selected compatibility branch."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from lxml import etree
from PIL import Image

from upf_agent.contracts import DocumentBlock, SourceRef

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
V = "urn:schemas-microsoft-com:vml"
NS = {"w": W, "r": R, "a": A, "mc": MC, "v": V}
PARSER_VERSION = "docx-xml-v4"


def local(node: etree._Element) -> str:
    return etree.QName(node).localname


def attr(node: etree._Element, namespace: str, name: str) -> str | None:
    return node.get(f"{{{namespace}}}{name}")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


@dataclass
class ParsedDocx:
    blocks: list[DocumentBlock] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class DocxParser:
    def __init__(self, source: SourceRef, path: Path, media_dir: Path):
        self.source, self.path, self.media_dir = source, path, media_dir
        self.result = ParsedDocx()
        self.section: list[str] = []
        self.styles: dict[str, tuple[str, int | None]] = {}
        self.rels: dict[str, str] = {}
        self.zip: ZipFile | None = None
        self.sequence = 0
        self.selected_textboxes = 0
        self.compatibility = {"alternate_content": 0, "choice": 0, "fallback": 0}

    def _xml(self, name: str) -> etree._Element:
        assert self.zip is not None
        return etree.fromstring(self.zip.read(name))

    def _prepare(self) -> None:
        assert self.zip is not None
        if "word/styles.xml" in self.zip.namelist():
            root = self._xml("word/styles.xml")
            for style in root.findall("w:style", NS):
                sid = attr(style, W, "styleId")
                name_node = style.find("w:name", NS)
                outline_node = style.find("w:pPr/w:outlineLvl", NS)
                if sid:
                    name = attr(name_node, W, "val") if name_node is not None else sid
                    raw_level = attr(outline_node, W, "val") if outline_node is not None else None
                    self.styles[sid] = (name or sid, int(raw_level) if raw_level else None)
        if "word/_rels/document.xml.rels" in self.zip.namelist():
            root = self._xml("word/_rels/document.xml.rels")
            for rel in root:
                rid, target = rel.get("Id"), rel.get("Target")
                if rid and target and not target.startswith(("http:", "https:", "/")):
                    self.rels[rid] = "word/" + target.removeprefix("./")

    def _compat(self, node: etree._Element) -> None:
        for alt in list(node.xpath(".//mc:AlternateContent", namespaces=NS)):
            self.compatibility["alternate_content"] += 1
            choices = alt.findall("mc:Choice", NS)
            fallback = alt.find("mc:Fallback", NS)
            chosen = choices[0] if choices else fallback
            if chosen is None:
                alt.getparent().remove(alt)
                continue
            self.compatibility["choice" if choices else "fallback"] += 1
            self.compatibility["fallback" if choices else "choice"] += int(
                fallback is not None if choices else bool(choices)
            )
            parent, idx = alt.getparent(), alt.getparent().index(alt)
            for child in list(chosen):
                parent.insert(idx, child)
                idx += 1
            parent.remove(alt)

    def _add(
        self,
        kind: str,
        raw: str,
        locator: str,
        parent: str | None = None,
        metadata: dict | None = None,
        flags: list[str] | None = None,
    ) -> DocumentBlock:
        self.sequence += 1
        block = DocumentBlock(
            source_id=self.source.source_id,
            block_id=f"{self.source.source_id}:b{self.sequence:06d}",
            parent_id=parent,
            section_path=self.section.copy(),
            kind=kind,
            raw_text=raw,
            normalized_text=normalize(raw),
            locator=locator,
            extraction_method=PARSER_VERSION,
            quality_flags=flags or [],
            ordinal=self.sequence,
            metadata=metadata or {},
        )
        self.result.blocks.append(block)
        return block

    def _text(self, node: etree._Element) -> str:
        parts = []
        for element in node.iter():
            if not isinstance(element.tag, str):
                continue
            if element is not node and local(element) == "txbxContent":
                continue
            # Descendants of a textbox belong only to its own block.
            inside_nested_box = False
            for ancestor in element.iterancestors():
                if ancestor is node:
                    break
                if local(ancestor) == "txbxContent":
                    inside_nested_box = True
                    break
            if inside_nested_box:
                continue
            name = local(element)
            if name in {"t", "instrText", "delText"} and element.text:
                parts.append(element.text)
            elif name == "tab":
                parts.append("\t")
            elif name in {"br", "cr"}:
                parts.append("\n")
        return "".join(parts)

    def _style(self, p: etree._Element) -> dict:
        ppr = p.find("w:pPr", NS)
        style_node = ppr.find("w:pStyle", NS) if ppr is not None else None
        sid = attr(style_node, W, "val") if style_node is not None else None
        name, level = self.styles.get(sid or "", (sid or "", None))
        num = ppr.find("w:numPr", NS) if ppr is not None else None
        data = {
            "style_id": sid,
            "style_name": name,
            "outline_level": level,
            "list": num is not None,
        }
        if num is not None:
            for key in ("ilvl", "numId"):
                child = num.find(f"w:{key}", NS)
                data[key] = attr(child, W, "val") if child is not None else None
        return data

    def _locator(self, parent: str, child: etree._Element) -> str:
        tag = local(child)
        siblings = [x for x in child.getparent() if isinstance(x.tag, str) and local(x) == tag]
        return f"{parent}/{tag}[{siblings.index(child) + 1}]"

    def _paragraph(self, p: etree._Element, locator: str, parent: str | None = None) -> None:
        raw = self._text(p)
        style = self._style(p)
        level = style["outline_level"]
        if level is not None and level < 9 and normalize(raw):
            self.section = self.section[:level] + [normalize(raw)]
        flags = []
        if style["style_name"].lower().startswith(("toc", "目录")):
            flags.append("table_of_contents")
        if not raw.strip():
            flags.append("empty_text")
        block = self._add("paragraph", raw, locator, parent, style, flags)
        for box_index, box in enumerate(p.xpath(".//w:txbxContent", namespaces=NS), 1):
            self.selected_textboxes += 1
            box_locator = f"{locator}/txbxContent[{box_index}]"
            box_block = self._add("textbox", self._text(box), box_locator, block.block_id)
            for child in box:
                if local(child) == "p":
                    self._paragraph(child, self._locator(box_locator, child), box_block.block_id)
                elif local(child) == "tbl":
                    self._table(child, self._locator(box_locator, child), box_block.block_id)
        for idx, image in enumerate(p.xpath(".//a:blip | .//v:imagedata", namespaces=NS), 1):
            rid = attr(image, R, "embed") or attr(image, R, "id")
            target = self.rels.get(rid or "")
            if target and self.zip is not None and target in self.zip.namelist():
                data = self.zip.read(target)
                sha = digest(data)
                suffix = Path(target).suffix.lower() or ".bin"
                self.media_dir.mkdir(parents=True, exist_ok=True)
                output = self.media_dir / f"{sha}{suffix}"
                if not output.exists():
                    output.write_bytes(data)
                try:
                    with Image.open(BytesIO(data)) as im:
                        width, height = im.size
                except Exception:
                    width, height = None, None
                image_block = self._add(
                    "image",
                    "",
                    f"{locator}/image[{idx}]",
                    block.block_id,
                    {
                        "sha256": sha,
                        "media_path": str(output),
                        "width": width,
                        "height": height,
                        "context": normalize(raw),
                        "relationship_id": rid,
                    },
                    ["image_text_unverified"],
                )
                self.result.images.append(
                    {
                        "block_id": image_block.block_id,
                        "source_id": self.source.source_id,
                        "sha256": sha,
                        "path": str(output),
                        "width": width,
                        "height": height,
                        "locator": image_block.locator,
                        "context": normalize(raw),
                        "section_path": self.section.copy(),
                    }
                )

    def _table(self, table: etree._Element, locator: str, parent: str | None = None) -> None:
        rows = table.findall("w:tr", NS)
        table_block = self._add("table", "", locator, parent, {"row_count": len(rows)})
        carry: dict[int, dict] = {}
        row_texts = []
        for row_number, row in enumerate(rows, 1):
            col = 0
            cells = []
            for cell_number, cell in enumerate(row.findall("w:tc", NS), 1):
                tcpr = cell.find("w:tcPr", NS)
                span_node = tcpr.find("w:gridSpan", NS) if tcpr is not None else None
                span = int(attr(span_node, W, "val") or 1) if span_node is not None else 1
                merge_node = tcpr.find("w:vMerge", NS) if tcpr is not None else None
                merge = attr(merge_node, W, "val") or "continue" if merge_node is not None else None
                cell_locator = f"{locator}/tr[{row_number}]/tc[{cell_number}]"
                texts = []
                for child in cell:
                    child_locator = self._locator(cell_locator, child)
                    if local(child) == "p":
                        self._paragraph(child, child_locator, table_block.block_id)
                        texts.append(self._text(child))
                    elif local(child) == "tbl":
                        self._table(child, child_locator, table_block.block_id)
                raw = "\n".join(texts)
                inherited = carry.get(col) if merge == "continue" else None
                entry = {
                    "column": col + 1,
                    "span": span,
                    "raw_text": raw,
                    "effective_text": inherited["effective_text"] if inherited else raw,
                    "origin": inherited["origin"] if inherited else cell_locator,
                    "locator": cell_locator,
                    "vmerge": merge,
                }
                cells.append(entry)
                for offset in range(span):
                    if merge in {"restart", "continue"}:
                        carry[col + offset] = entry
                    else:
                        carry.pop(col + offset, None)
                col += span
            row_raw = "\t".join(c["raw_text"] for c in cells)
            row_texts.append(row_raw)
            self._add(
                "table_row",
                row_raw,
                f"{locator}/tr[{row_number}]",
                table_block.block_id,
                {"cells": cells, "row_index": row_number},
            )
        table_block.raw_text = "\n".join(row_texts)
        table_block.normalized_text = normalize(table_block.raw_text)

    def parse(self) -> ParsedDocx:
        with ZipFile(self.path) as zf:
            self.zip = zf
            self._prepare()
            root = self._xml("word/document.xml")
            raw_boxes = len(root.xpath(".//w:txbxContent", namespaces=NS))
            self._compat(root)
            body = root.find("w:body", NS)
            if body is None:
                raise ValueError(f"DOCX has no body: {self.path}")
            for child in body:
                if not isinstance(child.tag, str):
                    continue
                locator = self._locator("word/document.xml/body", child)
                if local(child) == "p":
                    self._paragraph(child, locator)
                elif local(child) == "tbl":
                    self._table(child, locator)
            all_media = [
                x for x in zf.namelist() if x.startswith("word/media/") and not x.endswith("/")
            ]
            referenced = {image["sha256"] for image in self.result.images}
            unreferenced = 0
            for media_name in all_media:
                data = zf.read(media_name)
                sha = digest(data)
                if sha in referenced:
                    continue
                unreferenced += 1
                self.media_dir.mkdir(parents=True, exist_ok=True)
                output = self.media_dir / f"{sha}{Path(media_name).suffix.lower()}"
                if not output.exists():
                    output.write_bytes(data)
                self._add(
                    "image",
                    "",
                    media_name,
                    metadata={"sha256": sha, "media_path": str(output)},
                    flags=["unreferenced_media", "image_text_unverified"],
                )
            self.result.stats = {
                "source_id": self.source.source_id,
                "blocks": len(self.result.blocks),
                "tables": sum(b.kind == "table" for b in self.result.blocks),
                "textboxes_raw": raw_boxes,
                "textboxes_selected": self.selected_textboxes,
                "compatibility": self.compatibility,
                "images_referenced": len(self.result.images),
                "media_files": len(all_media),
                "media_unreferenced": unreferenced,
            }
            self.zip = None
        return self.result
