from pathlib import Path

from upf_agent.log_correlation import correlate_events
from upf_agent.log_tools import LogStore, ingest_log


def store_for(tmp_path: Path, text: str, timezone: str | None = "UTC") -> LogStore:
    source = tmp_path / "a.log"
    source.write_text(text)
    report = ingest_log(source, tmp_path / "store", timezone, "UPF-A")
    return LogStore(tmp_path / "store", report["generation"])


def test_pair_requires_real_peer_and_timezone_and_keeps_missing_topology(tmp_path: Path) -> None:
    store = store_for(
        tmp_path,
        "2025-05-14 00:00:00 ERROR #Alarm# report PFCP node unreachable! peer ip: 192.0.2.1\n"
        "2025-05-14 00:00:01 ERROR #Alarm# clear PFCP node unreachable. peer ip: 192.0.2.1\n",
    )
    result = correlate_events(store)
    assert len(result.data["alarm_pairs"]) == 1
    assert result.data["alarm_pairs"][0]["duration_seconds"] == 1.0
    assert result.data["topology_checks"] == []
    assert any("topology" in warning.lower() for warning in result.warnings)


def test_masked_peer_cannot_prove_same_entity(tmp_path: Path) -> None:
    store = store_for(
        tmp_path,
        "2025-05-14 00:00:00 ERROR #Alarm# report PFCP node unreachable! peer ip: IP_ADDR\n"
        "2025-05-14 00:00:01 ERROR #Alarm# clear PFCP node unreachable. peer ip: IP_ADDR\n",
    )
    result = correlate_events(store)
    assert result.data["alarm_pairs"] == []
    assert result.data["identity_gap_messages"] == 2


def test_repeated_raise_is_ambiguous_and_not_counted_as_separate_faults(tmp_path: Path) -> None:
    store = store_for(
        tmp_path,
        "2025-05-14 00:00:00 ERROR Report license not installed alarm\n"
        "2025-05-14 00:00:01 ERROR Report license not installed alarm\n"
        "2025-05-14 00:00:02 ERROR Clear license not installed alarm\n",
    )
    result = correlate_events(store)
    assert result.data["alarm_pairs"] == []
    assert len(result.data["unpaired"]) == 3
    assert result.data["count_unit"] == "messages_not_independent_faults"


def test_unknown_timezone_does_not_infer_duration(tmp_path: Path) -> None:
    store = store_for(
        tmp_path,
        "2025-05-14 00:00:00 ERROR Report license not installed alarm\n"
        "2025-05-14 00:00:01 ERROR Clear license not installed alarm\n",
        None,
    )
    assert correlate_events(store).data["alarm_pairs"] == []
