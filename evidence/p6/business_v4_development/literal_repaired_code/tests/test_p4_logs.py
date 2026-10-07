from datetime import datetime
from pathlib import Path

import pytest

from upf_agent.contracts import TimeWindow
from upf_agent.log_parser import classify, stream_events
from upf_agent.log_tools import LogStore, ingest_log


def test_semantic_rules_do_not_confuse_severity_or_clear_with_raise() -> None:
    assert classify("Recv #MsgHeartbeat# Req from cp.")[0:2] == (
        "pfcp_heartbeat_request",
        "observed",
    )
    assert classify("Clear license not installed alarm")[0:2] == (
        "license_not_installed",
        "cleared",
    )
    assert classify("Report license not installed alarm")[0:2] == (
        "license_not_installed",
        "raised",
    )
    assert classify("#Alarm# report PFCP node unreachable!")[0:2] == (
        "pfcp_node_unreachable",
        "raised",
    )
    assert classify("#Alarm# clear PFCP node unreachable.")[0:2] == (
        "pfcp_node_unreachable",
        "cleared",
    )
    assert (
        classify("#GtpuNode# handover old ip: IP_ADDR to new ip: IP_ADDR. CAUSE farId:0x2 update")[
            0
        ]
        == "far_handover"
    )
    assert classify("FATAL initializing a normal service")[0] == "unclassified"


def test_every_line_byte_accounted_including_invalid_and_multiline(tmp_path: Path) -> None:
    path = tmp_path / "sample.log"
    raw = (
        b"2025-05-14 00:00:00.000001 ERROR issue\r\n"
        b"  traceback detail\r\n\r\nmalformed\n"
        b"2025-02-30 00:00:00 ERROR invalid\n\xff\n"
        b"2025-05-14 00:00:00.999999 INFO ok"
    )
    path.write_bytes(raw)
    events = list(stream_events(path, "source", "a" * 64))
    assert sum(e.line_end - e.line_start + 1 for e in events) == 7
    assert sum(e.byte_length for e in events) == len(raw)
    assert events[0].line_end == 2
    assert [e.parse_status for e in events] == [
        "parsed",
        "blank",
        "quarantined",
        "quarantined",
        "quarantined",
        "parsed",
    ]
    assert all(e.timestamp_utc is None for e in events)
    assert events[-1].byte_offset + events[-1].byte_length == len(raw)


def test_oversized_line_and_masked_entities_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "huge.log"
    path.write_bytes(
        b"x" * 200000 + b"\n2025-05-14 00:00:00 ERROR peer ip: IP_ADDR, local seid: SEID_MASKED\n"
    )
    events = list(stream_events(path, "s", "a" * 64))
    assert len(events) == 2
    assert events[0].parse_status == "quarantined"
    assert len(events[0].raw_excerpt) <= 8192
    assert sum(e.byte_length for e in events) == path.stat().st_size
    assert events[1].peer_ip is None
    assert set(events[1].masked_fields) == {"peer_ip", "seid"}


def test_half_open_window_and_exact_raw_context(tmp_path: Path) -> None:
    source = tmp_path / "a.log"
    source.write_bytes(
        b"2025-05-14 00:00:00.999999 ERROR #Alarm# report PFCP node unreachable!\r\n"
        b"2025-05-14 00:00:01.000000 ERROR #Alarm# clear PFCP node unreachable.\r\n"
    )
    manifest = ingest_log(source, tmp_path / "store", timezone_name="Asia/Shanghai")
    store = LogStore(tmp_path / "store", manifest["generation"])
    window = TimeWindow(
        start=datetime.fromisoformat("2025-05-14T00:00:00+08:00"),
        end=datetime.fromisoformat("2025-05-14T00:00:01+08:00"),
        timezone_source="test",
    )
    result = store.analyze(window=window)
    assert result.data["total_events"] == 1
    found = store.find(window=window).data["events"]
    assert len(found) == 1
    event_id = found[0]["event_id"]
    assert store.context(event_id, before=0, after=0).data["raw_text"].endswith("unreachable!\r\n")
    source.write_text("changed")
    with pytest.raises(ValueError, match="source.*hash"):
        store.context(event_id)


def test_unknown_timezone_refuses_absolute_window_and_filters_are_safe(tmp_path: Path) -> None:
    source = tmp_path / "a.log"
    source.write_text("2025-05-14 00:00:00 ERROR example\n")
    manifest = ingest_log(source, tmp_path / "store")
    store = LogStore(tmp_path / "store", manifest["generation"])
    window = TimeWindow(
        start=datetime.fromisoformat("2025-05-14T00:00:00+00:00"),
        end=datetime.fromisoformat("2025-05-15T00:00:00+00:00"),
        timezone_source="test",
    )
    assert store.analyze(window=window).status.value == "need_input"
    assert store.find(event_type="'; DROP TABLE events; --").data["events"] == []
    with pytest.raises(ValueError):
        store.find(limit=10001)
    assert store.analyze().data["physical_lines"] == 1


@pytest.mark.parametrize(
    ("text", "expected_kind", "expected_state"),
    [
        ("upf return error: Invalid upfd public start timer", "timer_init_failure", "observed"),
        ("get N4 comm name by cmd ifconfig svc2 failed", "n4_name_failure", "observed"),
        ("upf return error: iface index get fail!", "interface_index_failure", "observed"),
        ("resend gtpu echo req, peer ip: IP_ADDR", "gtpu_echo_retry", "observed"),
        ("report GTPU node unreachable alarm", "gtpu_unreachable_report", "raised"),
        ("alarm gtpu unreachable ipv6!", "gtpu_unreachable_detected", "raised"),
        ("free reason: 'success pfcp delete req' stats: 1", "pfcp_session_deleted", "observed"),
        ("#init# upfd pod start up!", "upfd_start", "observed"),
        ("Send #MsgHeartbeat# Rsp to cp node:192.0.2.1", "pfcp_heartbeat_response", "observed"),
    ],
)
def test_remaining_rule_families(text: str, expected_kind: str, expected_state: str) -> None:
    assert classify(text)[:2] == (expected_kind, expected_state)


def test_explicit_offset_and_ambiguous_dst_do_not_use_host_timezone(tmp_path: Path) -> None:
    source = tmp_path / "dst.log"
    source.write_text(
        "2025-11-02 01:30:00 INFO ambiguous US local time\n"
        "2025-11-02 01:30:00-04:00 INFO explicit offset\n"
        "2025-03-09 02:30:00 INFO nonexistent US local time\n"
    )
    events = list(stream_events(source, "s", "a" * 64, "America/New_York"))
    assert events[0].timestamp_utc is None
    assert events[2].timestamp_utc is None
    assert events[1].timestamp_utc.isoformat() == "2025-11-02T05:30:00+00:00"
    assert events[1].timezone_status == "explicit"


def test_long_continuations_are_split_without_lost_lines(tmp_path: Path) -> None:
    source = tmp_path / "trace.log"
    source.write_bytes(b"2025-05-14 00:00:00 ERROR issue\n" + (b"  " + b"x" * 1000 + b"\n") * 100)
    events = list(stream_events(source, "s", "a" * 64))
    assert sum(event.line_end - event.line_start + 1 for event in events) == 101
    assert sum(event.byte_length for event in events) == source.stat().st_size
    assert all(event.byte_length <= 65536 for event in events)
    assert all(len(event.raw_excerpt) <= 8192 for event in events)
    assert any(event.parse_status == "quarantined" for event in events)
