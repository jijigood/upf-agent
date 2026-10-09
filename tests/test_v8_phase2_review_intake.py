import json
from pathlib import Path
from runpy import run_path

import pytest

from upf_agent.v8.models import digest
from upf_agent.v8.plan_cases import prepare_case

PACKAGE = run_path(str(Path(__file__).resolve().parents[1] / "scripts/v8_phase2_review.py"))[
    "review_package"
]


def pilot(root):
    root.mkdir()
    return [prepare_case(root / "cases" / f"pilot-{i + 1:02d}", i)[1] for i in range(10)]


def records(gold, second="approve"):
    return [
        {
            "case_id": gold.case_id,
            "reference_sha256": digest(gold),
            "reviewer": f"unit-test-reviewer-{seat}",
            "decision": decision,
            "notes": ["test fixture only"],
            "review_version": "unit-test-1",
            "review_source": "unit test, not a real expert review",
        }
        for seat, decision in [("A", "approve"), ("B", second)]
    ]


def test_exported_empty_forms_do_not_count_as_reviews(tmp_path):
    source = tmp_path / "pilot"
    pilot(source)
    output = tmp_path / "empty"
    status = PACKAGE(source, output)
    assert status["actual_review_records"] == 0
    assert not status["calibration_pass"] and not status["all_references_reviewed"]
    assert len(list((output / "empty-forms").glob("*/*.json"))) == 20
    assert all(
        json.loads(p.read_text())["reviewer"] is None
        for p in (output / "empty-forms").glob("*/*.json")
    )


@pytest.mark.parametrize("decision", ["approve", "request_changes"])
def test_intake_approval_and_adjudication_preserve_original_reference(tmp_path, decision):
    source = tmp_path / "pilot"
    gold = pilot(source)[0]
    ref = source / "cases/pilot-01/case-reference.json"
    before = ref.read_bytes()
    review = tmp_path / "records.json"
    review.write_text(json.dumps(records(gold, decision)))
    status = PACKAGE(source, tmp_path / "received", review)
    assert ref.read_bytes() == before
    assert status["actual_review_records"] == 2
    assert status["approved_cases"] == (["pilot-01"] if decision == "approve" else [])
    assert status["adjudication_cases"] == ([] if decision == "approve" else ["pilot-01"])
    assert not status["calibration_pass"]


@pytest.mark.parametrize("fault", ["hash", "duplicate"])
def test_invalid_review_rejected_before_writing_any_output(tmp_path, fault):
    source = tmp_path / "pilot"
    pair = records(pilot(source)[0])
    if fault == "hash":
        pair[0]["reference_sha256"] = "0" * 64
    else:
        pair[1]["reviewer"] = pair[0]["reviewer"]
    review = tmp_path / "records.json"
    review.write_text(json.dumps(pair))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        PACKAGE(source, output, review)
    assert not output.exists()
