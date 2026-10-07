from pathlib import Path
from types import SimpleNamespace

import pytest

from upf_agent.p6_contracts import EvalCase, validate_partition
from upf_agent.p6_truth_validation import validate_reviewed_sources


def make_case(**updates):
    return EvalCase(
        **{
            "id": "test-1",
            "category": "knowledge",
            "split": "dev",
            "question": "测试问题",
            "source_groups": ["section-old"],
            "index_generation": "a" * 20,
            "label_status": "reviewed",
            "reviewer": "测试复核人",
            "review_date": "2026-09-29",
            **updates,
        }
    )


@pytest.fixture
def sources(monkeypatch, tmp_path):
    import upf_agent.p6_truth_validation as validation

    monkeypatch.setattr(
        validation,
        "load_active_index",
        lambda root: (tmp_path, {"index_generation": "a" * 20}),
    )
    monkeypatch.setattr(validation, "load_chunks", lambda path: {})
    return validation


def test_corrected_chunk_cannot_escape_original_partition_group(sources, monkeypatch, tmp_path):
    monkeypatch.setattr(
        sources, "load_chunks", lambda path: {"chunk-new": SimpleNamespace(parent_id="section-new")}
    )
    case = make_case(evidence_ids=["chunk-new"])
    with pytest.raises(ValueError, match="exceeds the declared source groups"):
        validate_reviewed_sources(tmp_path, [case])


def test_same_actual_log_event_cannot_hide_behind_different_declared_groups(
    sources, monkeypatch, tmp_path
):
    class Store:
        def __init__(self, *_args):
            pass

        def context(self, eid):
            return SimpleNamespace(data={"event": {"event_id": eid}})

    monkeypatch.setattr(sources, "LogStore", Store)
    common = {"category": "log", "log_generation": "b" * 20, "evidence_ids": ["event-1"]}
    first = make_case(**common)
    second = make_case(
        **common,
        id="test-2",
        question="另一个问题",
        split="heldout",
        source_groups=["section-other"],
    )
    validate_partition([first, second])
    with pytest.raises(ValueError, match="Actual log evidence crosses"):
        validate_reviewed_sources(tmp_path, [first, second])


def test_same_actual_topology_file_cannot_cross_folds(sources, tmp_path):
    fixture = tmp_path / "eval/p6/topology.csv"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("node\nSYN-UPF-0\n")
    common = {
        "category": "topology",
        "synthetic": True,
        "evidence_ids": ["eval/p6/topology.csv:row2"],
    }
    first = make_case(**common)
    second = make_case(
        **common,
        id="test-2",
        question="另一个问题",
        split="heldout",
        source_groups=["section-other"],
    )
    validate_partition([first, second])
    with pytest.raises(ValueError, match="Actual topology file crosses"):
        validate_reviewed_sources(tmp_path, [first, second])


def test_canonical_csv_locator_counts_the_header(sources):
    root = Path(__file__).resolve().parents[1]
    case = make_case(
        category="topology",
        synthetic=True,
        evidence_ids=["eval/p6/topology_synthetic_0.csv:row2"],
    )
    assert validate_reviewed_sources(root, [case])["topology_files_resolved"] == 1
    case.evidence_ids = ["eval/p6/topology_synthetic_0.csv:row1"]
    with pytest.raises(ValueError, match="outside the fixture"):
        validate_reviewed_sources(root, [case])
