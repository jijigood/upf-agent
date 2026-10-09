"""E4 four-mode correctness first, then gated performance expansion; no model access."""

import argparse
from pathlib import Path

from upf_agent.v8.e4 import run_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["small", "stress", "full"], default="small")
    parser.add_argument("--workers", type=int, choices=[1, 4], default=1)
    parser.add_argument("--correctness", type=Path)
    args = parser.parse_args()
    result = run_matrix(args.project, args.output, args.profile, args.workers, args.correctness)
    print({"pass": result["pass"], "method_batches": result["method_batches"]}, flush=True)
    raise SystemExit(0 if result["pass"] else 1)
