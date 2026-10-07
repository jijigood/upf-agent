"""Bounded binary log parsing; physical lines and bytes are never silently dropped."""

import hashlib
import ipaddress
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from upf_agent.contracts import LogEvent

PARSER_VERSION = "upfd-stream-v1"
RULE_VERSION = "upfd-rules-v1"
MAX_LINE = 65536
MAX_EVENT = 65536
EXCERPT_LIMIT = 8192
HEADER = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?"
    r"(?:Z|[+-]\d{2}:\d{2})?)\s+(?P<severity>[A-Z]+)\s+(?P<body>.*)$"
)
State = Literal["raised", "cleared", "observed", "unknown"]
RULES: list[tuple[str, State, str | None, re.Pattern[str]]] = [
    (
        "pfcp_node_unreachable",
        "cleared",
        "PFCP",
        re.compile(r"#Alarm# clear PFCP node unreachable\."),
    ),
    (
        "pfcp_node_unreachable",
        "raised",
        "PFCP",
        re.compile(r"#Alarm# report PFCP node unreachable!"),
    ),
    (
        "license_not_installed",
        "cleared",
        None,
        re.compile(r"Clear license not installed alarm", re.I),
    ),
    (
        "license_not_installed",
        "raised",
        None,
        re.compile(r"Report license not installed alarm", re.I),
    ),
    ("pfcp_heartbeat_request", "observed", "PFCP", re.compile(r"Recv #MsgHeartbeat# Req from cp")),
    ("pfcp_heartbeat_response", "observed", "PFCP", re.compile(r"Send #MsgHeartbeat# Rsp to cp")),
    (
        "far_handover",
        "observed",
        "GTP-U",
        re.compile(r"#GtpuNode# handover old ip: .+ to new ip: .+\. CAUSE farId:\S+ update"),
    ),
    (
        "gtpu_unreachable_report",
        "raised",
        "GTP-U",
        re.compile(r"report GTPU node unreachable alarm"),
    ),
    (
        "gtpu_unreachable_detected",
        "raised",
        "GTP-U",
        re.compile(r"alarm gtpu unreachable ipv[46]!"),
    ),
    ("gtpu_echo_retry", "observed", "GTP-U", re.compile(r"resend gtpu echo req, peer ip:")),
    (
        "pfcp_session_deleted",
        "observed",
        "PFCP",
        re.compile(r"free reason: 'success pfcp delete req' stats:"),
    ),
    (
        "timer_init_failure",
        "observed",
        None,
        re.compile(r"upf return error: Invalid upfd public start timer"),
    ),
    (
        "interface_index_failure",
        "observed",
        None,
        re.compile(r"upf return error: iface index get fail!"),
    ),
    ("n4_name_failure", "observed", "PFCP", re.compile(r"get N4 comm name by cmd .+ failed")),
    ("upfd_start", "observed", None, re.compile(r"#init# upfd pod start up!")),
]
FIELD_PATTERNS = {
    "peer_ip": re.compile(r"(?:peer ip:\s*|cp node:\s*)([^\s,}]+)"),
    "session": re.compile(r"(?<!#)\bsession:\s*([^\s,}]+)"),
    "seid": re.compile(r"local seid:\s*([^\s,}]+)"),
    "teid": re.compile(r"local teid:\s*([^\s,}]+)"),
    "imsi": re.compile(r"\bimsi:\s*([^\s,}]+)"),
}
MASKS = {
    "IP_ADDR": "peer_ip",
    "CPNODE_MASKED": "cp_node",
    "SESSION_MASKED": "session",
    "DNN_MASKED": "dnn",
    "IMSI_MASKED": "imsi",
    "SEID_MASKED": "seid",
    "TEID_MASKED": "teid",
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(text: str) -> tuple[str, State, str | None, str]:
    for index, (kind, state, protocol, pattern) in enumerate(RULES):
        if pattern.search(text):
            return kind, state, protocol, f"{RULE_VERSION}:{index + 1}"
    return "unclassified", "unknown", None, f"{RULE_VERSION}:unmatched"


def message_body(raw: str) -> str:
    match = HEADER.match(raw.splitlines()[0]) if raw else None
    text = match.group("body") if match else raw
    # FID and C source locations are execution metadata, not device identity.
    return re.sub(r"^FID\.\d+\s+\S+\):\s*", "", text)


def _timestamp(text: str, zone: ZoneInfo | None) -> tuple[datetime | None, str]:
    value = datetime.fromisoformat(text.replace(",", ".").replace("Z", "+00:00"))
    if value.tzinfo is not None:
        return value.astimezone(UTC), "explicit"
    if zone is None:
        return None, "unknown"
    first, second = value.replace(tzinfo=zone, fold=0), value.replace(tzinfo=zone, fold=1)
    if first.utcoffset() != second.utcoffset():
        # Ambiguous/nonexistent DST wall times require an explicit source offset.
        return None, "unknown"
    return first.astimezone(UTC), "assumed"


def stream_events(
    path: Path,
    source_id: str,
    source_sha: str,
    timezone_name: str | None = None,
    device_id: str | None = None,
) -> Iterator[LogEvent]:
    zone = ZoneInfo(timezone_name) if timezone_name else None
    pending: LogEvent | None = None
    offset = 0
    number = 0
    with path.open("rb") as handle:
        while chunk := handle.readline(MAX_LINE + 1):
            number += 1
            length = len(chunk)
            oversized = length > MAX_LINE
            if oversized and not chunk.endswith(b"\n"):
                while rest := handle.readline(MAX_LINE + 1):
                    length += len(rest)
                    if rest.endswith(b"\n"):
                        break
            invalid_utf8 = False
            try:
                text = chunk.decode("utf-8")
            except UnicodeDecodeError:
                text = chunk.decode("utf-8", errors="replace")
                invalid_utf8 = True
            match = HEADER.match(text.rstrip("\r\n"))
            status: Literal["parsed", "blank", "quarantined"] = "parsed"
            flags: list[str] = []
            utc = None
            tz_status = "unknown"
            if oversized or invalid_utf8:
                status = "quarantined"
                flags.append("oversized_line" if oversized else "invalid_utf8")
            elif not text.strip():
                status = "blank"
            elif match:
                try:
                    utc, tz_status = _timestamp(match.group("ts"), zone)
                except ValueError:
                    status = "quarantined"
                    flags.append("invalid_timestamp")
                if utc is None and status == "parsed":
                    flags.append("timezone_unresolved")
            elif (
                text.startswith((" ", "\t", "Traceback ("))
                and pending is not None
                and pending.parse_status == "parsed"
                and pending.byte_length + length <= MAX_EVENT
            ):
                pending.line_end = number
                pending.byte_length += length
                if len(pending.raw_excerpt) < EXCERPT_LIMIT:
                    pending.raw_excerpt = (pending.raw_excerpt + text)[:EXCERPT_LIMIT]
                if (
                    pending.byte_length > EXCERPT_LIMIT
                    and "excerpt_truncated" not in pending.quality_flags
                ):
                    pending.quality_flags.append("excerpt_truncated")
                offset += length
                continue
            else:
                status = "quarantined"
                flags.append("unrecognized_line_or_event_limit")
            if pending is not None:
                yield pending
            kind, state, protocol, rule = (
                classify(text)
                if status == "parsed"
                else ("quarantine" if status == "quarantined" else "blank", "unknown", None, "none")
            )
            fields: dict[str, str | None] = {key: None for key in FIELD_PATTERNS}
            masked = sorted({field for marker, field in MASKS.items() if marker in text})
            for key, pattern in FIELD_PATTERNS.items():
                field_match = pattern.search(text)
                if field_match and key not in masked and status == "parsed":
                    value = field_match.group(1)
                    if key == "peer_ip":
                        try:
                            value = str(ipaddress.ip_address(value))
                        except ValueError:
                            flags.append("invalid_peer_ip")
                            continue
                    fields[key] = value
            if length > EXCERPT_LIMIT:
                flags.append("excerpt_truncated")
            pending = LogEvent(
                event_id=f"log-{source_sha[:16]}-{offset}",
                source_id=source_id,
                source_sha256=source_sha,
                line_start=number,
                line_end=number,
                byte_offset=offset,
                byte_length=length,
                timestamp_original=match.group("ts") if match else None,
                timestamp_utc=utc,
                timezone_status=tz_status,
                device_id=device_id,
                protocol=protocol,
                event_type=kind,
                event_state=state,
                severity_raw=match.group("severity") if match else None,
                masked_fields=masked,
                raw_excerpt=text[:EXCERPT_LIMIT],
                parser_version=PARSER_VERSION,
                rule_id=rule,
                classification_basis="message_regex" if rule != "none" else "parser",
                parse_status=status,
                quality_flags=flags,
                **fields,
            )
            offset += length
        if pending is not None:
            yield pending
