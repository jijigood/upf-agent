"""Cross-check the P2 corpus and sidecars against original local sources."""

import hashlib
import json
from pathlib import Path

from upf_agent.contracts import DocumentBlock
from upf_agent.corpus_db import active_id, connect
from upf_agent.quality_repair import active_ocr_path, apply_corrections
from upf_agent.quality_repair import sha as file_sha
from upf_agent.review import _source_text


def verify(root: Path, output: Path) -> dict:
    db = connect(root / "data/p2/corpus.sqlite3")
    try:
        generation = active_id(db)
        if generation is None:
            raise ValueError("No active generation")
        sources = {
            sid: (relative, sha)
            for sid, relative, sha in db.execute(
                "SELECT source_id,relative_path,sha256 FROM sources WHERE generation=?",
                (generation,),
            )
        }
        source_root = root / "data/raw"
        if len(sources) != 4:
            raise ValueError(f"Expected four manuals, got {len(sources)}")
        for sid, (relative, sha) in sources.items():
            if hashlib.sha256((source_root / relative).read_bytes()).hexdigest() != sha:
                raise ValueError(f"Manual source SHA mismatch: {sid}")
        blocks = {
            b.block_id: b
            for (raw,) in db.execute("SELECT payload FROM blocks WHERE generation=?", (generation,))
            if (b := DocumentBlock.model_validate_json(raw))
        }
        commands = [
            json.loads(raw)
            for (raw,) in db.execute(
                "SELECT payload FROM commands WHERE generation=?", (generation,)
            )
        ]
        for block in blocks.values():
            if block.parent_id and block.parent_id not in blocks:
                raise ValueError(f"Missing parent: {block.block_id}")
        for command in commands:
            source_block = blocks[command["syntax_block_id"]]
            if command["syntax"] != source_block.raw_text.strip():
                raise ValueError(f"Command changed original syntax: {command['command_id']}")
        review = json.loads(
            (root / "evidence/p2/structure_review_60.json").read_text(encoding="utf-8")
        )
        if review["generation"] != generation or len(review["items"]) != 60:
            raise ValueError("Review packet differs from active generation")
        for item in review["items"]:
            block = blocks[item["block_id"]]
            original = _source_text(source_root / sources[block.source_id][0], block.locator)
            if original != block.raw_text:
                raise ValueError(f"Review source mismatch: {block.block_id}")
        ocr = json.loads(active_ocr_path(root).read_text(encoding="utf-8"))
        pdf = json.loads((root / "evidence/p2/paper_docling.json").read_text(encoding="utf-8"))
        if ocr["generation"] != generation or len(ocr["images"]) != 30:
            raise ValueError("OCR artifact differs from active generation")
        for image in ocr["images"]:
            block = blocks[image["block_id"]]
            if (
                block.metadata["sha256"] != image["image_sha256"]
                or image["status"] != "ocr_unverified"
            ):
                raise ValueError(f"OCR image mismatch or promoted authority: {image['block_id']}")
        if pdf["role"] != "paper" or len(pdf["pages"]) != 12:
            raise ValueError("Paper PDF role or page count changed")
        page_sizes = {page["page_no"]: page["size"] for page in pdf["pages"]}
        pdf_tables = sum(item.get("table_cells") is not None for item in pdf["items"])
        for item in pdf["items"]:
            if not item["provenance"]:
                raise ValueError("PDF item lacks page provenance")
            for ref in item["provenance"]:
                box = ref["bbox"]
                width, height = page_sizes[ref["page_no"]]
                if box is None or not (
                    0 <= box["l"] <= box["r"] <= width and 0 <= box["b"] <= box["t"] <= height
                ):
                    raise ValueError("PDF boundary box is outside its page")
        auxiliary = dict(
            db.execute(
                "SELECT kind,count(*) FROM auxiliary_items WHERE generation=? GROUP BY kind",
                (generation,),
            )
        )
        if auxiliary.get("ocr") != sum(len(x["lines"]) for x in ocr["images"]):
            raise ValueError("OCR sidecar count mismatch")
        if auxiliary.get("pdf") != len(pdf["items"]):
            raise ValueError("PDF sidecar count mismatch")
        corrected_path = root / "evidence/p2/ocr_corrected.json"
        correction_summary = {}
        human_summary = {}
        if corrected_path.exists():
            corrected = json.loads(corrected_path.read_text())
            original_path = root / "evidence/p2/ocr_30.json"
            ledger_path = root / "evidence/p2/ocr_corrections.json"
            ledger = json.loads(ledger_path.read_text())
            if (
                file_sha(original_path) != ledger["original_sha256"]
                or file_sha(ledger_path) != corrected["correction_ledger_sha256"]
            ):
                raise ValueError("OCR correction lineage mismatch")
            expected = apply_corrections(json.loads(original_path.read_text()), ledger)
            if (
                corrected["generation"] != generation
                or corrected["images"] != expected["images"]
                or corrected["human_signoff"]
            ):
                raise ValueError("OCR corrected text or authority mismatch")
            if auxiliary.get("ocr_corrected") != auxiliary["ocr"]:
                raise ValueError("Corrected OCR sidecar count mismatch")
            correction_summary = {
                "ocr_assistant_checked": len(corrected["images"]),
                "ocr_quarantined_lines": sum(
                    not line["usable"] for image in corrected["images"] for line in image["lines"]
                ),
                "ocr_corrected_sha256": file_sha(corrected_path),
            }
        human_path = root / "evidence/p2/human_review_v3.json"
        if human_path.exists():
            human = json.loads(human_path.read_text())
            if human["generation"] != generation or not human["user_declared_review_completed"]:
                raise ValueError("Human review lacks matching generation or user attestation")
            for relative, expected_sha in human["bindings_sha256"].items():
                if file_sha(root / relative) != expected_sha:
                    raise ValueError(f"Human review source changed: {relative}")
            if len(human["items"]) != 90:
                raise ValueError("Human review must cover the frozen 90 entries")
            structured = human["items"][:60]
            images = human["items"][60:]
            for item, source in zip(structured, review["items"], strict=True):
                if item["locator"] != source["locator"] or item["source_id"] != source["source_id"]:
                    raise ValueError("Human review structural source does not match")
            for item, image in zip(images, corrected["images"], strict=True):
                if (
                    item["locator"] != image["locator"]
                    or item["image_sha256"] != image["image_sha256"]
                ):
                    raise ValueError("Human review screenshot source does not match")
            if any(
                item["conclusion"] not in {"通过", "不通过", "无法确认"} for item in human["items"]
            ):
                raise ValueError("Human review has an unknown conclusion")
            human_summary = {
                "review_human_completed": len(structured),
                "review_human_passed": sum(item["conclusion"] == "通过" for item in structured),
                "ocr_human_completed": len(images),
                "ocr_human_passed": sum(item["conclusion"] == "通过" for item in images),
                "ocr_corrected_pending_recheck": human["corrected_pending_recheck"],
                "human_review_receipt_sha256": file_sha(human_path),
                "m2_human_acceptance_pass": not human["corrected_pending_recheck"]
                and all(item["conclusion"] == "通过" for item in human["items"]),
            }
        report = {
            "generation": generation,
            "source_count": len(sources),
            "blocks": len(blocks),
            "commands": len(commands),
            "tables": sum(b.kind == "table" for b in blocks.values()),
            "textboxes": sum(b.kind == "textbox" for b in blocks.values()),
            "review_source_exact": len(review["items"]),
            "review_human_completed": sum(
                x["human_status"] == "completed" for x in review["items"]
            ),
            "ocr_images": len(ocr["images"]),
            "ocr_lines": auxiliary["ocr"],
            "ocr_human_completed": sum(
                x["human_review_status"] == "completed" for x in ocr["images"]
            ),
            "pdf_pages": len(pdf["pages"]),
            "pdf_items": auxiliary["pdf"],
            "pdf_tables": pdf_tables,
            "paper_role_isolated": True,
            "pass": True,
            **correction_summary,
            **human_summary,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        db.close()
