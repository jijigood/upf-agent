"""Independent review intake and calibration; no review is fabricated by the pilot runner."""

from typing import Literal

from pydantic import Field

from upf_agent.v8.models import Model, digest
from upf_agent.v8.plan_ir import CaseGold


class GoldReview(Model):
    case_id: str
    reviewer: str = Field(min_length=1)
    reference_sha256: str
    decision: Literal["approve", "request_changes"]
    notes: list[str] = Field(default_factory=list)
    review_version: str = Field(min_length=1)
    review_source: str = Field(min_length=1)


def apply_reviews(gold: CaseGold, reviews: list[GoldReview]):
    if len(reviews) != 2 or len({r.reviewer for r in reviews}) != 2:
        raise ValueError("exactly two independent review records required")
    if any(r.case_id != gold.case_id or r.reference_sha256 != digest(gold) for r in reviews):
        raise ValueError("review does not match frozen case/reference")
    if any(r.decision != "approve" for r in reviews):
        raise ValueError(
            "review disagreement/change request requires adjudication and new reference"
        )
    result = gold.model_dump(mode="json")
    result.update(
        reviewed=True, reviewers=[r.reviewer for r in reviews], review_version=digest(reviews)
    )
    return CaseGold.model_validate(result)


def calibration_report(reviews):
    cases = {}
    for r in reviews:
        cases.setdefault(r.case_id, []).append(r)
    paired = [
        rs
        for rs in cases.values()
        if len(rs) == 2
        and len({r.reviewer for r in rs}) == 2
        and len({r.reference_sha256 for r in rs}) == 1
    ]
    agree = sum(rs[0].decision == rs[1].decision for rs in paired)
    return {
        "paired_cases": len(paired),
        "agreement_count": agree,
        "agreement_rate": agree / len(paired) if paired else None,
        "adjudication_cases": [
            rs[0].case_id for rs in paired if any(r.decision != "approve" for r in rs)
        ],
        "unpaired_cases": sorted(set(cases) - {rs[0].case_id for rs in paired}),
    }


def validate_splits(golds):
    for field in ["source_group", "structure_family"]:
        splits = {}
        for gold in golds:
            splits.setdefault(getattr(gold, field), set()).add(gold.split)
        if any(len(values) > 1 for values in splits.values()):
            raise ValueError(f"case family leakage across split: {field}")
