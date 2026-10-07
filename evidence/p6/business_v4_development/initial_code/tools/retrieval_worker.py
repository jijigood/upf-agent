"""Local Sentence Transformers worker, launched by the application as a subprocess.

Input and output are JSONL. Only the configured local model directory is opened.
The process handles one model type so embedding and reranking can be scheduled apart.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["embed", "rerank"])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        raise SystemExit("local model assets missing")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["OMP_NUM_THREADS"] = "4"
    import torch

    if args.device == "cpu":
        torch.set_num_threads(4)
    model_kwargs = {"torch_dtype": torch.float32} if args.device == "cpu" else {}
    if args.task == "embed":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            str(args.model),
            device=args.device,
            local_files_only=True,
            model_kwargs=model_kwargs,
        )
    else:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            str(args.model),
            device=args.device,
            local_files_only=True,
            model_kwargs=model_kwargs,
        )
    print(json.dumps({"ready": True, "task": args.task}), flush=True)
    for line in sys.stdin:
        request: dict[str, object] = {}
        try:
            request = json.loads(line)
            request_id = request["request_id"]
            if args.task == "embed":
                texts = request["texts"]
                if not isinstance(texts, list) or not 1 <= len(texts) <= 16:
                    raise ValueError("embed expects 1–16 texts")
                if any(not isinstance(text, str) or len(text) > 8000 for text in texts):
                    raise ValueError("each input must be text of at most 8000 characters")
                role = request.get("role", "document")
                if role not in {"query", "document"}:
                    raise ValueError("embedding role must be query or document")
                kwargs = (
                    {"prompt_name": "query"}
                    if role == "query" and "query" in model.prompts
                    else {"prompt": ""}
                )
                vectors = model.encode(texts, normalize_embeddings=True, **kwargs)
                output = {
                    "request_id": request_id,
                    "vectors": vectors.tolist(),
                    "dimension": int(vectors.shape[-1]),
                    "normalized": True,
                }
            else:
                pairs = request["pairs"]
                if not isinstance(pairs, list) or not 1 <= len(pairs) <= 32:
                    raise ValueError("rerank expects 1–32 query/document pairs")
                if any(
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or any(not isinstance(item, str) or len(item) > 8000 for item in pair)
                    for pair in pairs
                ):
                    raise ValueError("each pair must contain two bounded strings")
                raw = model.predict(pairs)
                logits = [float(value) for value in raw]
                output = {
                    "request_id": request_id,
                    "raw_logit_difference": logits,
                    "p_yes": [sigmoid(value) for value in logits],
                    "score_convention": "ST5.4 LogitScore; p_yes=sigmoid(raw)",
                }
            print(json.dumps(output, ensure_ascii=False), flush=True)
        except (KeyError, TypeError, ValueError) as error:
            print(
                json.dumps({"request_id": request.get("request_id"), "error": str(error)}),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
