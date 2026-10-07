"""Standalone offline PP-OCRv5 worker; never labels recognized text as verified syntax."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def _hash_tree(path: Path) -> str:
    sha = hashlib.sha256()
    for file in sorted(x for x in path.rglob("*") if x.is_file()):
        sha.update(str(file.relative_to(path)).encode())
        sha.update(hashlib.sha256(file.read_bytes()).digest())
    return sha.hexdigest()


def run_ocr(queue_path: Path, output: Path, detection_dir: Path, recognition_dir: Path) -> dict:
    for directory in (detection_dir, recognition_dir):
        if (
            not directory.is_dir()
            or not any(directory.rglob("*.pdmodel"))
            and not any(directory.rglob("*.json"))
        ):
            raise FileNotFoundError(f"Incomplete local PaddleOCR model: {directory}")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from paddleocr import PaddleOCR

    engine = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_dir=str(detection_dir),
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir=str(recognition_dir),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
    )
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    results = []
    for item in queue["selected"]:
        image = Path(item["path"])
        if hashlib.sha256(image.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Image SHA mismatch: {image}")
        predictions = engine.predict(str(image))
        lines = []
        for prediction in predictions:
            data = prediction.json
            if callable(data):
                data = data()
            data = data.get("res", data)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            boxes = data.get("rec_boxes", [])
            for index, text in enumerate(texts):
                box = boxes[index]
                if hasattr(box, "tolist"):
                    box = box.tolist()
                lines.append({"text": text, "score": float(scores[index]), "box": box})
        results.append(
            {
                "source_id": item["source_id"],
                "block_id": item["block_id"],
                "image_sha256": item["sha256"],
                "locator": item["locator"],
                "section_path": item["section_path"],
                "context": item["context"],
                "lines": lines,
                "status": "ocr_unverified",
                "human_review_status": "pending",
            }
        )
    manifest = {
        "engine": "PaddleOCR",
        "models": {
            "detection": {
                "name": "PP-OCRv5_server_det",
                "sha256_tree": _hash_tree(detection_dir),
                "path": str(detection_dir),
            },
            "recognition": {
                "name": "PP-OCRv5_server_rec",
                "sha256_tree": _hash_tree(recognition_dir),
                "path": str(recognition_dir),
            },
        },
        "generation": queue["generation"],
        "images": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "images": len(results),
        "lines": sum(len(x["lines"]) for x in results),
        "output": str(output),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("detection_dir", type=Path)
    parser.add_argument("recognition_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_ocr(args.queue, args.output, args.detection_dir, args.recognition_dir)))
