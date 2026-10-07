"""Send an actual >16K-token local prompt to a 32K model service."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx
from transformers import AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18001/v1")
    parser.add_argument("--served-name", default="Qwen3.8-27B")
    parser.add_argument("--target-tokens", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.target_tokens <= 16384 or args.target_tokens >= 30000:
        parser.error("target-tokens must be between 16,385 and 29,999")
    if args.endpoint != "http://127.0.0.1:18001/v1":
        parser.error("only the dedicated loopback target endpoint is accepted")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    unit = "背景记录：设备运行正常。\n"
    unit_tokens = len(tokenizer(unit, add_special_tokens=False)["input_ids"])
    filler = unit * ((args.target_tokens // unit_tokens) + 1)
    code = "UPF-CTX-7319"
    content = filler + f"\n最后的校验码是 {code}。请只回答最后的校验码。"
    content_tokens = len(tokenizer(content, add_special_tokens=False)["input_ids"])
    payload = {
        "model": args.served_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    started = perf_counter()
    with httpx.Client(timeout=600, trust_env=False) as client:
        response = client.post(f"{args.endpoint}/chat/completions", json=payload)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]
    elapsed_ms = round((perf_counter() - started) * 1000)
    result = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 -- Python 3.10
        "model": args.served_name,
        "content_tokens": content_tokens,
        "elapsed_ms": elapsed_ms,
        "pass": content_tokens > 16384 and code in (answer.get("content") or ""),
        "answer": answer.get("content"),
        "expected_code": code,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
