"""Text-layer checkpoint for PDFs, kept outside the product manual corpus."""

import hashlib
import json
import logging
from pathlib import Path

from pypdf import PdfReader


def extract_text_layer(source: Path, source_id: str, output: Path) -> dict:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    pages = []
    reader = PdfReader(source)
    for index, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(
            {
                "page_no": index,
                "text": text,
                "width_pt": float(page.mediabox.width),
                "height_pt": float(page.mediabox.height),
                "quality_flags": ["no_bbox", "text_layer_only"],
            }
        )
    data = {
        "source_id": source_id,
        "role": "paper",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parser": "pypdf-text-layer-fallback",
        "pages": pages,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "pages": len(pages),
        "nonempty_pages": sum(bool(p["text"].strip()) for p in pages),
        "output": str(output),
    }
