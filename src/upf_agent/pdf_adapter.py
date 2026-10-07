"""Standalone Docling PDF worker. Run with the pinned .venv-docling interpreter."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def convert_pdf(source: Path, output: Path, artifacts: Path, source_id: str) -> dict:
    if not artifacts.is_dir() or not any(artifacts.iterdir()):
        raise FileNotFoundError(f"Docling local artifacts missing: {artifacts}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    options.enable_remote_services = False
    options.artifacts_path = artifacts
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    result = converter.convert(source)
    doc = result.document
    pages = []
    for number, page in doc.pages.items():
        pages.append({"page_no": int(number), "size": [page.size.width, page.size.height]})
    items = []
    for item, level in doc.iterate_items():
        text = getattr(item, "text", None)
        table_cells = None
        if text is None and hasattr(item, "export_to_dataframe"):
            frame = item.export_to_dataframe(doc=doc).fillna("")
            table_cells = [[str(value) for value in row] for row in frame.values.tolist()]
            text = frame.to_csv(index=False)
        if not text:
            continue
        prov = []
        for ref in getattr(item, "prov", []):
            box = getattr(ref, "bbox", None)
            prov.append(
                {
                    "page_no": ref.page_no,
                    "bbox": {
                        "l": box.l,
                        "t": box.t,
                        "r": box.r,
                        "b": box.b,
                        "coord_origin": str(box.coord_origin),
                    }
                    if box
                    else None,
                }
            )
        items.append(
            {
                "text": text,
                "label": str(getattr(item, "label", "")),
                "table_cells": table_cells,
                "level": level,
                "provenance": prov,
            }
        )
    if not pages or not items:
        raise ValueError("Docling produced no pages or text items")
    data = {
        "source_id": source_id,
        "role": "paper",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parser": "docling-standard-no-ocr-v2",
        "offline": True,
        "pages": pages,
        "items": items,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"page_count": len(pages), "item_count": len(items), "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("source_id")
    args = parser.parse_args()
    print(json.dumps(convert_pdf(args.source, args.output, args.artifacts, args.source_id)))
