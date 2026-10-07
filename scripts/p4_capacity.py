"""Measure ingestion/query wall time and process RSS; amplification is explicitly synthetic."""

import argparse
import json
import math
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from upf_agent.log_parser import file_sha
from upf_agent.log_tools import LogStore, ingest_log, write_json

ROOT = Path(__file__).resolve().parents[1]


def worker(source: Path, synthetic: bool, output: Path) -> None:
    started = time.monotonic()
    manifest = ingest_log(source, ROOT / "data/p4/capacity/logs", synthetic=synthetic)
    ingestion_time = time.monotonic() - started
    query_started = time.monotonic()
    store = LogStore(ROOT / "data/p4/capacity/logs", manifest["generation"])
    analysis = store.analyze()
    query_seconds = time.monotonic() - query_started
    assert analysis.data["source_bytes"] == source.stat().st_size
    assert analysis.data["physical_lines"] == manifest["physical_lines"]
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux KiB
    report = {
        "source": str(source),
        "source_sha256": manifest["source_sha256"],
        "source_bytes": source.stat().st_size,
        "synthetic": synthetic,
        "generation": manifest["generation"],
        "physical_lines": manifest["physical_lines"],
        "events": manifest["events"],
        "ingestion_seconds": round(ingestion_time, 4),
        "query_seconds": round(query_seconds, 4),
        "peak_process_rss_bytes": rss_bytes,
        "rss_limit_bytes": 4 * 1024**3,
        "rss_pass": rss_bytes <= 4 * 1024**3,
        "duckdb_memory_limit": "256MB",
        "arrow_batch_rows": 4096,
        "accounting_pass": True,
        "event_counts": manifest["event_counts"],
        "parquet_bytes": sum(path.stat().st_size for path in store.directory.rglob("*.parquet")),
        "measurement": (
            "fresh subprocess resource.getrusage(RUSAGE_SELF).ru_maxrss; full import and queries"
        ),
        "cached_generation": ingestion_time < manifest["elapsed_seconds"],
    }
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.synthetic, args.output)
        return
    raw = ROOT / "data/raw/upfd_masked.log"
    evidence = ROOT / "evidence/p4"
    amplification = ROOT / "data/p4/capacity/synthetic_amplified_upfd.log"
    copies = math.ceil(1024**3 / raw.stat().st_size)
    amplification.parent.mkdir(parents=True, exist_ok=True)
    if not amplification.exists():
        stage = amplification.with_suffix(".building")
        with stage.open("wb") as target:
            for _ in range(copies):
                with raw.open("rb") as source:
                    shutil.copyfileobj(source, target, 1024 * 1024)
        stage.rename(amplification)
    assert amplification.stat().st_size == copies * raw.stat().st_size
    write_json(
        evidence / "amplification.json",
        {
            "synthetic": True,
            "method": "repeat complete masked source byte-for-byte; no new incidents",
            "copies": copies,
            "bytes": amplification.stat().st_size,
            "sha256": file_sha(amplification),
            "original_sha256": file_sha(raw),
            "warning": (
                "Repeated timestamps/entities are artificial; "
                "use only for capacity, never business accuracy."
            ),
        },
    )
    reports = []
    for source, synthetic, name in [
        (raw, False, "capacity_real.json"),
        (amplification, True, "capacity_synthetic.json"),
    ]:
        output = evidence / name
        command = [sys.executable, __file__, "--worker", str(source), "--output", str(output)]
        if synthetic:
            command.append("--synthetic")
        subprocess.run(command, check=True, cwd=ROOT)
        reports.append(json.loads(output.read_text()))
    assert reports[1]["physical_lines"] == reports[0]["physical_lines"] * copies
    assert reports[1]["event_counts"] == {
        key: count * copies for key, count in reports[0]["event_counts"].items()
    }
    write_json(
        evidence / "capacity.json",
        {
            "runs": reports,
            "synthetic_copies": copies,
            "scaled_counts_pass": True,
            "rss_pass": all(row["rss_pass"] for row in reports),
        },
    )


if __name__ == "__main__":
    main()
