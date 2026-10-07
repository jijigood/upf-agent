"""Versioned Parquet log storage and bounded, parameterized deterministic tools."""

import hashlib
import json
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import JsonValue

from upf_agent.contracts import LogEvent, ResultStatus, TimeWindow, ToolResult
from upf_agent.log_parser import PARSER_VERSION, RULE_VERSION, file_sha, stream_events

Row = dict[str, JsonValue]
INTEGER_FIELDS = {"line_start", "line_end", "byte_offset", "byte_length"}
LIST_FIELDS = {"masked_fields", "quality_flags"}
SCHEMA = pa.schema(
    [
        (
            key,
            pa.int64()
            if key in INTEGER_FIELDS
            else pa.list_(pa.string())
            if key in LIST_FIELDS
            else pa.string(),
        )
        for key in LogEvent.model_fields
    ]
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ingest_log(
    source: Path,
    root: Path,
    timezone_name: str | None = None,
    device_id: str | None = None,
    synthetic: bool = False,
    template_snapshot: Path | None = None,
) -> Row:
    source = source.resolve(strict=True)
    sha = file_sha(source)
    source_id = f"log-{sha[:16]}"
    parser_sha = file_sha(Path(__file__).with_name("log_parser.py"))
    store_sha = file_sha(Path(__file__))
    template_sha = file_sha(template_snapshot / "manifest.json") if template_snapshot else None
    identity = {
        "source_sha256": sha,
        "timezone": timezone_name,
        "device_id": device_id,
        "parser": PARSER_VERSION,
        "rules": RULE_VERSION,
        "parser_sha256": parser_sha,
        "store_sha256": store_sha,
        "template_sha256": template_sha,
        "synthetic": synthetic,
        "source_path": str(source),
        "contracts_sha256": file_sha(Path(__file__).with_name("contracts.py")),
    }
    generation = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    root.mkdir(parents=True, exist_ok=True)
    target = root / generation
    if target.exists():
        existing = LogStore(root, generation)
        if file_sha(Path(str(existing.manifest["source_path"]))) != sha:
            raise ValueError("registered source hash changed")
        return existing.manifest
    stage = Path(tempfile.mkdtemp(prefix=".building-", dir=root))
    started = time.monotonic()
    counts: Counter[str] = Counter()
    states: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    total_bytes = lines = events = unresolved = disorder = 0
    previous: str | None = None
    batch: list[Row] = []
    part = 0
    matcher = None
    if template_snapshot:
        from upf_agent.log_templates import FrozenTemplates

        matcher = FrozenTemplates(template_snapshot)

    def flush() -> None:
        nonlocal part
        months: dict[str, list[Row]] = {}
        for row in batch:
            original = row["timestamp_original"]
            month = str(original)[:7] if row["parse_status"] == "parsed" else "unknown"
            months.setdefault(month, []).append(row)
        for month, rows in months.items():
            directory = stage / f"month={month}"
            directory.mkdir(exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(rows, schema=SCHEMA),
                directory / f"part-{part:06d}.parquet",
                compression="zstd",
            )
            part += 1
        batch.clear()

    try:
        for event in stream_events(source, source_id, sha, timezone_name, device_id):
            if matcher and event.parse_status == "parsed":
                event.template_id = matcher.match(event.raw_excerpt)
            events += 1
            lines += event.line_end - event.line_start + 1
            total_bytes += event.byte_length
            counts[str(event.event_type)] += 1
            states[f"{event.event_type}:{event.event_state}"] += 1
            statuses[event.parse_status] += event.line_end - event.line_start + 1
            if event.parse_status == "parsed":
                unresolved += event.timestamp_utc is None
                original = event.timestamp_original
                if original is not None:
                    disorder += previous is not None and original < previous
                    previous = original
            batch.append(event.model_dump(mode="json"))
            if len(batch) >= 4096:
                flush()
        if batch:
            flush()
        if not events:
            pq.write_table(pa.Table.from_pylist([], schema=SCHEMA), stage / "empty.parquet")
        if total_bytes != source.stat().st_size or file_sha(source) != sha:
            raise ValueError("source hash/size changed during import")
        manifest: Row = {
            **identity,
            "generation": generation,
            "source_id": source_id,
            "source_path": str(source),
            "source_bytes": total_bytes,
            "physical_lines": lines,
            "events": events,
            "line_status_counts": dict(statuses),
            "event_counts": dict(counts),
            "state_counts": dict(states),
            "unresolved_timezone_events": unresolved,
            "adjacent_wall_time_disorder": disorder,
            "timezone_source": "user_provided" if timezone_name else "unknown",
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "files": {str(p.relative_to(stage)): file_sha(p) for p in stage.rglob("*.parquet")},
            "count_unit": "log_messages_not_independent_faults",
            "template_snapshot": str(template_snapshot.resolve()) if template_snapshot else None,
        }
        write_json(stage / "manifest.json", manifest)
        stage.rename(target)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


class LogStore:
    def __init__(self, root: Path, generation: str) -> None:
        if not generation.isalnum() or len(generation) != 20:
            raise ValueError("invalid registered log generation")
        self.directory = root / generation
        self.manifest: Row = json.loads((self.directory / "manifest.json").read_text())
        files = cast(dict[str, str], self.manifest["files"])
        for name, sha in files.items():
            path = (self.directory / name).resolve()
            if not path.is_relative_to(self.directory.resolve()) or file_sha(path) != sha:
                raise ValueError("Parquet archive hash mismatch")
        self.paths = [str(self.directory / name) for name in files]

    def _query(self, sql: str, parameters: list[object] | None = None) -> list[Row]:
        with duckdb.connect(config={"memory_limit": "256MB", "threads": "2"}) as connection:
            connection.read_parquet(self.paths, hive_partitioning=False).create_view("events")
            cursor = connection.execute(sql, parameters or [])
            keys = [item[0] for item in cursor.description]
            return [dict(zip(keys, row, strict=True)) for row in cursor.fetchall()]

    def _where(
        self, window: TimeWindow | None, filters: dict[str, str | None]
    ) -> tuple[str, list[object]]:
        clauses: list[str] = []
        parameters: list[object] = []
        if window:
            clauses += [
                "CAST(timestamp_utc AS TIMESTAMPTZ) >= ?",
                "CAST(timestamp_utc AS TIMESTAMPTZ) < ?",
            ]
            parameters += [window.start, window.end]
        allowed = {
            "event_type",
            "event_state",
            "protocol",
            "device_id",
            "peer_ip",
            "seid",
            "teid",
            "parse_status",
        }
        for key, value in filters.items():
            if key not in allowed:
                raise ValueError("unsupported log filter")
            if value is not None:
                clauses.append(f'"{key}" = ?')
                parameters.append(value)
        return " AND ".join(clauses) or "TRUE", parameters

    def _timezone_gap(self, window: TimeWindow | None) -> ToolResult | None:
        if window and self.manifest["unresolved_timezone_events"]:
            return ToolResult(
                status=ResultStatus.NEED_INPUT,
                data={"generation": self.manifest["generation"]},
                warnings=["Source timezone unresolved; absolute time filtering is blocked."],
                elapsed_ms=0,
            )
        return None

    def find(
        self,
        window: TimeWindow | None = None,
        limit: int = 20,
        event_type: str | None = None,
        event_state: str | None = None,
        protocol: str | None = None,
        device_id: str | None = None,
        peer_ip: str | None = None,
        seid: str | None = None,
        teid: str | None = None,
        parse_status: str | None = None,
    ) -> ToolResult:
        if not 1 <= limit <= 10000:
            raise ValueError("limit must be between 1 and 10000")
        if gap := self._timezone_gap(window):
            return gap
        started = time.monotonic()
        filters = {
            "event_type": event_type,
            "event_state": event_state,
            "protocol": protocol,
            "device_id": device_id,
            "peer_ip": peer_ip,
            "seid": seid,
            "teid": teid,
            "parse_status": parse_status,
        }
        where, parameters = self._where(window, filters)
        rows = self._query(
            f"SELECT * EXCLUDE(raw_excerpt), left(raw_excerpt, 512) AS raw_excerpt "
            f"FROM events WHERE {where} ORDER BY byte_offset LIMIT ?",
            [*parameters, limit + 1],
        )
        truncated = len(rows) > limit
        rows = rows[:limit]
        return ToolResult(
            status=ResultStatus.PARTIAL if truncated else ResultStatus.COMPLETED,
            data={
                "events": rows,
                "truncated": truncated,
                "limit": limit,
                "generation": self.manifest["generation"],
                "order": "source_byte_offset",
            },
            evidence_ids=[str(row["event_id"]) for row in rows],
            warnings=["Result capped; counts must use analyze, not sample length."]
            if truncated
            else [],
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    def analyze(
        self,
        window: TimeWindow | None = None,
        event_type: str | None = None,
        device_id: str | None = None,
        bucket_minutes: int = 60,
    ) -> ToolResult:
        if not 1 <= bucket_minutes <= 1440:
            raise ValueError("bucket_minutes must be in 1..1440")
        if gap := self._timezone_gap(window):
            return gap
        started = time.monotonic()
        where, parameters = self._where(window, {"event_type": event_type, "device_id": device_id})
        summary = self._query(
            "SELECT count(*) total_events, "
            "coalesce(sum(line_end-line_start+1),0)::BIGINT physical_lines, "
            f"coalesce(sum(byte_length),0)::BIGINT source_bytes FROM events WHERE {where}",
            parameters,
        )[0]
        counts = self._query(
            f"SELECT event_type, event_state, count(*) AS messages FROM events WHERE {where} "
            "GROUP BY event_type, event_state ORDER BY messages DESC, event_type, event_state",
            parameters,
        )
        # Both samples and timeline are bounded; summary counts cover the entire selected scope.
        representatives = self._query(
            f"SELECT event_type, min_by(event_id, byte_offset) first_event_id, "
            f"max_by(event_id, byte_offset) last_event_id FROM events WHERE {where} "
            "GROUP BY event_type ORDER BY event_type",
            parameters,
        )
        buckets = self._query(
            f"SELECT floor(epoch(CAST(timestamp_utc AS TIMESTAMPTZ)) / ?)::BIGINT bucket_index, "
            f"count(*) messages FROM events WHERE ({where}) AND timestamp_utc IS NOT NULL "
            "GROUP BY bucket_index ORDER BY bucket_index LIMIT 10001",
            [bucket_minutes * 60, *parameters],
        )
        peak = self._query(
            "SELECT bucket_index, messages, example_event_id FROM ("
            "SELECT floor(epoch(CAST(timestamp_utc AS TIMESTAMPTZ)) / ?)::BIGINT bucket_index, "
            "count(*) messages, min_by(event_id, byte_offset) example_event_id "
            f"FROM events WHERE ({where}) AND timestamp_utc IS NOT NULL "
            "GROUP BY bucket_index) ORDER BY messages DESC, bucket_index LIMIT 1",
            [bucket_minutes * 60, *parameters],
        )
        entities = self._query(
            "SELECT device_id, peer_ip, count(*) messages, "
            "min_by(event_id, byte_offset) example_event_id "
            f"FROM events WHERE ({where}) AND (device_id IS NOT NULL OR peer_ip IS NOT NULL) "
            "GROUP BY device_id, peer_ip ORDER BY messages DESC, device_id, peer_ip LIMIT 101",
            parameters,
        )
        warnings = [
            "Counts represent log messages, not independent faults; FATAL remains raw severity."
        ]
        if self.manifest["unresolved_timezone_events"]:
            warnings.append(
                "Timezone unknown: UTC timeline unavailable for unresolved events; "
                "no cross-source time correlation."
            )
        truncated = len(buckets) > 10000
        return ToolResult(
            status=ResultStatus.PARTIAL if truncated else ResultStatus.COMPLETED,
            data={
                **summary,
                "generation": self.manifest["generation"],
                "counts": counts,
                "representatives": representatives,
                "peak_bucket": peak,
                "entity_representatives": entities[:100],
                "entity_representatives_truncated": len(entities) > 100,
                "buckets": buckets[:10000],
                "bucket_seconds": bucket_minutes * 60,
                "buckets_truncated": truncated,
                "scope": {
                    "event_type": event_type,
                    "device_id": device_id,
                    "window": window.model_dump(mode="json") if window else None,
                },
                "accounting": {
                    key: self.manifest[key]
                    for key in ("source_bytes", "physical_lines", "line_status_counts")
                },
            },
            warnings=warnings,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    def context(self, event_id: str, before: int = 2, after: int = 2) -> ToolResult:
        if not 0 <= before <= 20 or not 0 <= after <= 20:
            raise ValueError("context lines must be within 0..20")
        rows = self._query("SELECT * FROM events WHERE event_id = ?", [event_id])
        if len(rows) != 1:
            return ToolResult(status=ResultStatus.MISSING_SOURCE, data={}, elapsed_ms=0)
        event = rows[0]
        source = Path(str(self.manifest["source_path"]))
        if file_sha(source) != self.manifest["source_sha256"]:
            raise ValueError("registered source hash changed")
        offset, length = int(event["byte_offset"]), int(event["byte_length"])
        with source.open("rb") as handle:
            start = max(0, offset - 65536)
            handle.seek(start)
            prefix = handle.read(offset - start).splitlines(keepends=True)
            if start and prefix:
                prefix = prefix[1:]
            prefix_bytes = b"".join(prefix[-before:]) if before else b""
            handle.seek(offset)
            event_bytes = handle.read(min(length, 65536))
            suffix = b""
            if length <= 65536:
                for _ in range(after):
                    line = handle.readline(8193)
                    suffix += line
                    if len(line) > 8192:
                        break
        truncated = length > 65536 or len(prefix) < before or len(suffix) > 8192 * max(1, after)
        return ToolResult(
            status=ResultStatus.PARTIAL if truncated else ResultStatus.COMPLETED,
            data={
                "event": event,
                "raw_text": (prefix_bytes + event_bytes + suffix).decode("utf-8", errors="replace"),
                "source_sha256": self.manifest["source_sha256"],
                "byte_offset": offset,
                "byte_length": length,
                "line_start": event["line_start"],
                "line_end": event["line_end"],
                "truncated": truncated,
            },
            evidence_ids=[event_id],
            warnings=["Context bounded; source bytes remain authoritative."] if truncated else [],
            elapsed_ms=0,
        )

    def template_inventory(self) -> ToolResult:
        if self.manifest["template_snapshot"] is None:
            return ToolResult(
                status=ResultStatus.NEED_INPUT,
                warnings=["Train and import with a frozen snapshot first."],
                elapsed_ms=0,
            )
        rows = self._query(
            "SELECT template_id, count(*) messages, min_by(event_id, byte_offset) example_event_id "
            "FROM events WHERE parse_status='parsed' AND event_type='unclassified' "
            "GROUP BY template_id ORDER BY messages DESC, template_id LIMIT 101"
        )
        unmatched = self._query(
            "SELECT count(*) messages FROM events "
            "WHERE parse_status='parsed' AND template_id IS NULL"
        )[0]
        return ToolResult(
            status=ResultStatus.PARTIAL if len(rows) > 100 else ResultStatus.COMPLETED,
            data={
                "unclassified_message_templates": rows[:100],
                "snapshot_unmatched_messages": unmatched["messages"],
                "templates_truncated": len(rows) > 100,
                "unknown_is_anomaly": False,
                "snapshot": self.manifest["template_snapshot"],
            },
            warnings=[
                "Unclassified business rules and unseen Drain templates are separate; "
                "neither proves an anomaly."
            ],
            elapsed_ms=0,
        )
