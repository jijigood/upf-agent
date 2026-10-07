"""SQLite-backed, atomically published document generations."""

import hashlib
import json
import sqlite3
from pathlib import Path

import yaml

from upf_agent.commands import COMMAND_PARSER_VERSION, extract_commands
from upf_agent.contracts import DocumentBlock, SourceRef
from upf_agent.docx_parse import PARSER_VERSION, DocxParser

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS generations (
  id TEXT PRIMARY KEY, parser_version TEXT NOT NULL, state TEXT NOT NULL,
  source_manifest TEXT NOT NULL, stats TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS active_generation (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1), id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
  generation TEXT NOT NULL, source_id TEXT NOT NULL, sha256 TEXT NOT NULL,
  relative_path TEXT NOT NULL, role TEXT NOT NULL, product TEXT, version TEXT,
  PRIMARY KEY(generation, source_id)
);
CREATE TABLE IF NOT EXISTS blocks (
  generation TEXT NOT NULL, block_id TEXT NOT NULL, source_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL, kind TEXT NOT NULL, section_path TEXT NOT NULL,
  normalized_text TEXT NOT NULL, payload TEXT NOT NULL,
  PRIMARY KEY(generation, block_id)
);
CREATE INDEX IF NOT EXISTS blocks_scope ON blocks(generation, source_id, kind);
CREATE TABLE IF NOT EXISTS commands (
  generation TEXT NOT NULL, command_id TEXT NOT NULL, source_id TEXT NOT NULL,
  syntax TEXT NOT NULL, payload TEXT NOT NULL,
  PRIMARY KEY(generation, command_id)
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    return db


def _source(item: dict) -> SourceRef:
    return SourceRef.model_validate({k: item[k] for k in SourceRef.model_fields})


def active_id(db: sqlite3.Connection) -> str | None:
    row = db.execute("SELECT id FROM active_generation WHERE singleton=1").fetchone()
    return row[0] if row else None


def ingest(project_root: Path, parser_version: str = PARSER_VERSION) -> dict:
    corpus_config = yaml.safe_load(
        (project_root / "config/corpus.yaml").read_text(encoding="utf-8")
    )
    source_root = Path(corpus_config["source_root"])
    entries = json.loads((project_root / "evidence/p0/sources.json").read_text(encoding="utf-8"))
    sources = [_source(item) for item in entries if item["role"] == "manual"]
    for source in sources:
        raw_path = source_root / source.relative_path
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != source.sha256:
            raise ValueError(f"Source SHA mismatch: {source.relative_path}")
    manifest = [
        {"source_id": s.source_id, "sha256": s.sha256, "product": s.product, "version": s.version}
        for s in sources
    ]
    pipeline_version = f"{parser_version}+{COMMAND_PARSER_VERSION}"
    generation = hashlib.sha256(
        json.dumps(
            {"parser_version": pipeline_version, "sources": manifest}, sort_keys=True
        ).encode()
    ).hexdigest()[:20]
    db = connect(project_root / "data/p2/corpus.sqlite3")
    try:
        previous = active_id(db)
        row = db.execute(
            "SELECT state, stats FROM generations WHERE id=?", (generation,)
        ).fetchone()
        if row and row[0] == "ready":
            if previous != generation:
                with db:
                    db.execute(
                        "INSERT INTO active_generation VALUES(1,?) "
                        "ON CONFLICT(singleton) DO UPDATE SET id=excluded.id",
                        (generation,),
                    )
            stats = json.loads(row[1])
            return {"generation": generation, "reused": True, "stats": stats}
        totals = {
            "sources": 0,
            "blocks": 0,
            "commands": 0,
            "images": 0,
            "textboxes_raw": 0,
            "textboxes_selected": 0,
            "fallback_skipped": 0,
            "tables": 0,
            "media_unreferenced": 0,
            "per_source": [],
        }
        # One transaction: a failed parse leaves the formerly active generation intact.
        with db:
            db.execute("DELETE FROM blocks WHERE generation=?", (generation,))
            db.execute("DELETE FROM commands WHERE generation=?", (generation,))
            db.execute("DELETE FROM sources WHERE generation=?", (generation,))
            db.execute("DELETE FROM generations WHERE id=?", (generation,))
            for source in sources:
                raw_path = source_root / source.relative_path
                parsed = DocxParser(source, raw_path, project_root / "data/p2/media").parse()
                commands = extract_commands(source, parsed.blocks)
                db.execute(
                    "INSERT INTO sources VALUES(?,?,?,?,?,?,?)",
                    (
                        generation,
                        source.source_id,
                        source.sha256,
                        source.relative_path,
                        source.role,
                        source.product,
                        source.version,
                    ),
                )
                db.executemany(
                    "INSERT INTO blocks VALUES(?,?,?,?,?,?,?,?)",
                    [
                        (
                            generation,
                            b.block_id,
                            b.source_id,
                            b.ordinal,
                            b.kind,
                            json.dumps(b.section_path, ensure_ascii=False),
                            b.normalized_text,
                            b.model_dump_json(),
                        )
                        for b in parsed.blocks
                    ],
                )
                db.executemany(
                    "INSERT INTO commands VALUES(?,?,?,?,?)",
                    [
                        (generation, c.command_id, c.source_id, c.syntax, c.model_dump_json())
                        for c in commands
                    ],
                )
                totals["sources"] += 1
                totals["blocks"] += len(parsed.blocks)
                totals["commands"] += len(commands)
                totals["images"] += len(parsed.images)
                totals["tables"] += parsed.stats["tables"]
                totals["media_unreferenced"] += parsed.stats["media_unreferenced"]
                totals["textboxes_raw"] += parsed.stats["textboxes_raw"]
                totals["textboxes_selected"] += parsed.stats["textboxes_selected"]
                totals["fallback_skipped"] += parsed.stats["compatibility"]["fallback"]
                totals["per_source"].append({**parsed.stats, "commands": len(commands)})
            db.execute(
                "INSERT INTO generations VALUES(?,?,?,?,?)",
                (
                    generation,
                    pipeline_version,
                    "ready",
                    json.dumps(manifest, ensure_ascii=False),
                    json.dumps(totals, ensure_ascii=False),
                ),
            )
            db.execute(
                "INSERT INTO active_generation VALUES(1,?) "
                "ON CONFLICT(singleton) DO UPDATE SET id=excluded.id",
                (generation,),
            )
        return {"generation": generation, "reused": False, "stats": totals}
    finally:
        db.close()


def query_blocks(
    path: Path,
    *,
    product: str | None = None,
    version: str | None = None,
    kind: str | None = None,
    limit: int = 100,
) -> list[DocumentBlock]:
    db = connect(path)
    try:
        generation = active_id(db)
        if generation is None:
            return []
        sql = (
            "SELECT b.payload FROM blocks b JOIN sources s "
            "ON s.generation=b.generation AND s.source_id=b.source_id "
            "WHERE b.generation=?"
        )
        params: list[str | int] = [generation]
        for field, value in (("product", product), ("version", version)):
            if value is not None:
                sql += f" AND s.{field}=?"
                params.append(value)
        if kind is not None:
            sql += " AND b.kind=?"
            params.append(kind)
        sql += " ORDER BY s.relative_path, b.ordinal LIMIT ?"
        params.append(limit)
        return [DocumentBlock.model_validate_json(row[0]) for row in db.execute(sql, params)]
    finally:
        db.close()
