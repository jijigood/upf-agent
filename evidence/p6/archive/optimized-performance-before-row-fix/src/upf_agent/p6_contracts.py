"""Evaluation contracts separating drafts, reviewed labels and one-use heldout runs."""

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    category: Literal["command", "knowledge", "troubleshooting", "log", "topology", "missing"]
    split: Literal["dev", "heldout"]
    question: str = Field(min_length=1, max_length=2000)
    source_groups: list[str] = Field(min_length=1)
    product: str | None = None
    version: str | None = None
    index_generation: str
    label_status: Literal["draft", "reviewed"] = "draft"
    synthetic: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    expected_extracts: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    expected_statuses: list[str] = Field(default_factory=lambda: ["completed", "partial"])
    required_event_types: list[str] = Field(default_factory=list)
    log_generation: str | None = None
    reviewer: str | None = None
    review_date: str | None = None

    @model_validator(mode="after")
    def reviewed_has_provenance(self) -> "EvalCase":
        if self.label_status == "reviewed" and (not self.reviewer or not self.review_date):
            raise ValueError("Reviewed truth requires reviewer and date")
        if self.category == "topology" and not self.synthetic:
            raise ValueError("P6 currently has no validated real topology")
        return self


def read_suite(path: Path, split: str = "dev", allow_drafts: bool = False) -> list[EvalCase]:
    cases = [EvalCase.model_validate_json(line) for line in path.read_text().splitlines() if line]
    validate_partition(cases)
    selected = [case for case in cases if case.split == split]
    if not selected:
        raise ValueError("Empty evaluation partition")
    if not allow_drafts and any(case.label_status != "reviewed" for case in selected):
        raise ValueError("Human truth is not reviewed; use explicit draft development mode")
    if split == "heldout" and allow_drafts:
        raise ValueError("Heldout cannot run as draft development")
    return selected


def validate_partition(cases: list[EvalCase]) -> None:
    ids, questions, assignment = set(), set(), {}
    for case in cases:
        normalized = "".join(case.question.split()).casefold()
        if case.id in ids or normalized in questions:
            raise ValueError("Duplicate evaluation id/question")
        ids.add(case.id)
        questions.add(normalized)
        for group in case.source_groups:
            if group in assignment and assignment[group] != case.split:
                raise ValueError("Source/command family crosses dev and heldout")
            assignment[group] = case.split


def ranking_metrics(ranking: list[str], grades: dict[str, int]) -> dict[str, float | None]:
    if len(set(ranking)) != len(ranking) or any(value < 0 for value in grades.values()):
        raise ValueError("Ranking must be unique and relevance nonnegative")
    positive = {key for key, grade in grades.items() if grade > 0}
    if not positive:
        return {"recall_at_20": None, "hit_at_20": None, "ndcg_at_8": None}
    hit = set(ranking[:20]) & positive
    dcg = sum((2 ** grades.get(key, 0) - 1) / math.log2(i + 2) for i, key in enumerate(ranking[:8]))
    ideal = sum(
        (2**grade - 1) / math.log2(i + 2)
        for i, grade in enumerate(sorted(grades.values(), reverse=True)[:8])
    )
    return {
        "recall_at_20": len(hit) / len(positive),
        "hit_at_20": float(bool(hit)),
        "ndcg_at_8": dcg / ideal,
    }


def percentiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "p50": ordered[math.ceil(len(values) * 0.5) - 1],
        "p95": ordered[math.ceil(len(values) * 0.95) - 1],
    }
