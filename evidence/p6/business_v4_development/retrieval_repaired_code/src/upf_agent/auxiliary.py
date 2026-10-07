"""Versioned OCR/PDF sidecars, isolated from authoritative manual blocks."""

import hashlib
import json
from pathlib import Path

from upf_agent.corpus_db import active_id, connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS auxiliary_assets (
  generation TEXT NOT NULL, kind TEXT NOT NULL, source_id TEXT NOT NULL,
  sha256 TEXT NOT NULL, path TEXT NOT NULL, count INTEGER NOT NULL,
  PRIMARY KEY(generation, kind, source_id)
);
CREATE TABLE IF NOT EXISTS auxiliary_items (
  generation TEXT NOT NULL, kind TEXT NOT NULL, source_id TEXT NOT NULL,
  item_id TEXT NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL, payload TEXT NOT NULL,
  PRIMARY KEY(generation, kind, source_id, item_id)
);
CREATE INDEX IF NOT EXISTS auxiliary_search ON auxiliary_items(generation, kind, role);
"""


def attach_assets(db_path: Path, ocr_path: Path, pdf_path: Path, expected_paper_sha: str) -> dict:
    db = connect(db_path)
    db.executescript(SCHEMA)
    try:
        generation = active_id(db)
        if generation is None:
            raise ValueError("No active DOCX generation")
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        pdf = json.loads(pdf_path.read_text(encoding="utf-8"))
        if ocr["generation"] != generation:
            raise ValueError("OCR generation differs from active DOCX generation")
        if pdf["role"] != "paper" or pdf["source_sha256"] != expected_paper_sha:
            raise ValueError("Paper source role or SHA mismatch")
        records = []
        for image in ocr["images"]:
            if image["status"] != "ocr_unverified":
                raise ValueError("OCR text may not be marked as verified")
            for index, line in enumerate(image["lines"]):
                item_id = f"{image['block_id']}:{index:04d}"
                payload = {
                    "image_block_id": image["block_id"],
                    "image_sha256": image["image_sha256"],
                    "locator": image["locator"],
                    "box": line["box"],
                    "score": line["score"],
                    "status": "ocr_unverified",
                }
                records.append(
                    (
                        generation,
                        "ocr",
                        image["source_id"],
                        item_id,
                        "manual_ocr_unverified",
                        line["text"],
                        json.dumps(payload, ensure_ascii=False),
                    )
                )
        for index, item in enumerate(pdf["items"]):
            payload = {
                "provenance": item["provenance"],
                "label": item["label"],
                "table_cells": item.get("table_cells"),
                "source_sha256": expected_paper_sha,
            }
            records.append(
                (
                    generation,
                    "pdf",
                    pdf["source_id"],
                    f"paper:{index:05d}",
                    "paper",
                    item["text"],
                    json.dumps(payload, ensure_ascii=False),
                )
            )
        info = []
        for kind, path, source_id, count in (
            ("ocr", ocr_path, "screenshot_set", sum(len(x["lines"]) for x in ocr["images"])),
            ("pdf", pdf_path, pdf["source_id"], len(pdf["items"])),
        ):
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            previous = db.execute(
                "SELECT sha256 FROM auxiliary_assets WHERE generation=? AND kind=? AND source_id=?",
                (generation, kind, source_id),
            ).fetchone()
            info.append(
                {
                    "kind": kind,
                    "count": count,
                    "sha256": sha,
                    "reused": bool(previous and previous[0] == sha),
                }
            )
        with db:
            for entry in info:
                if entry["reused"]:
                    continue
                kind = entry["kind"]
                db.execute(
                    "DELETE FROM auxiliary_items WHERE generation=? AND kind=?", (generation, kind)
                )
                db.execute(
                    "DELETE FROM auxiliary_assets WHERE generation=? AND kind=?", (generation, kind)
                )
                path = ocr_path if kind == "ocr" else pdf_path
                source_id = "screenshot_set" if kind == "ocr" else pdf["source_id"]
                db.execute(
                    "INSERT INTO auxiliary_assets VALUES(?,?,?,?,?,?)",
                    (generation, kind, source_id, entry["sha256"], str(path), entry["count"]),
                )
                db.executemany(
                    "INSERT INTO auxiliary_items VALUES(?,?,?,?,?,?,?)",
                    (record for record in records if record[1] == kind),
                )
        return {"generation": generation, "assets": info}
    finally:
        db.close()


def search_auxiliary(db_path: Path, query: str, kind: str, limit: int = 20) -> list[dict]:
    if kind not in {"ocr", "pdf"}:
        raise ValueError("kind must be ocr or pdf")
    db = connect(db_path)
    db.executescript(SCHEMA)
    try:
        generation = active_id(db)
        if generation is None:
            return []
        if (
            kind == "ocr"
            and db.execute(
                "SELECT 1 FROM auxiliary_assets WHERE generation=? AND kind='ocr_corrected'",
                (generation,),
            ).fetchone()
        ):
            kind = "ocr_corrected"
        # Literal substring search; P3 will supply the ranked index.
        return [
            {
                "item_id": item_id,
                "source_id": source_id,
                "role": role,
                "text": text,
                "metadata": json.loads(payload),
            }
            for item_id, source_id, role, text, payload in db.execute(
                "SELECT item_id,source_id,role,text,payload FROM auxiliary_items "
                "WHERE generation=? AND kind=? AND role!='manual_ocr_quarantined' "
                "AND instr(lower(text),lower(?))>0 LIMIT ?",
                (generation, kind, query, limit),
            )
        ]
    finally:
        db.close()
