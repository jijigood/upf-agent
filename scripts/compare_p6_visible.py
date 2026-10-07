"""Compare the user-visible P6 answer payloads of two development runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

STOP_NOTES = {
    "补查没有新增信息，已终止重复工具循环。",
    "调查已达到最多 3 轮的业务限制。",
}


def visible_payload(path: Path, category: str) -> dict:
    result = json.loads(path.read_text())["result"]
    if category in {"command", "knowledge", "troubleshooting"}:
        answer = result.get("answer", {})
        return {
            key: answer.get(key)
            for key in ("status", "claims", "citations", "missing_information", "warnings")
        }
    if "report" in result:
        report = dict(result["report"])
        report.pop("run_id", None)
        for key in ("supporting_evidence_ids", "opposing_evidence_ids"):
            report[key] = [
                evidence_id.split(":", 1)[1] if ":" in evidence_id else evidence_id
                for evidence_id in report.get(key, [])
            ]
        return report
    return result


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def classification(before: dict, after: dict, fields: list[str], category: str) -> str:
    if not fields:
        return "unchanged"
    if fields == ["missing_information"] and category == "log":
        old, new = before.get("missing_information"), after.get("missing_information")
        if (
            isinstance(old, list)
            and isinstance(new, list)
            and old
            and new
            and old[0] in STOP_NOTES
            and new[0] in STOP_NOTES
            and old[1:] == new[1:]
        ):
            return "procedural_stop_only"
    return "semantic_change"


def compare_runs(old: Path, new: Path) -> dict:
    previous, current = (
        json.loads((directory / "summary.json").read_text()) for directory in (old, new)
    )
    if previous.get("suite_sha256") != current.get("suite_sha256"):
        raise ValueError("Runs use different reviewed suites")
    if previous.get("split") != current.get("split") or current.get("split") != "dev":
        raise ValueError("Only matching development runs may be compared")
    old_rows = {row["case"]: row for row in previous["rows"]}
    new_rows = {row["case"]: row for row in current["rows"]}
    if old_rows.keys() != new_rows.keys() or len(old_rows) != len(previous["rows"]):
        raise ValueError("Runs contain different or duplicate case IDs")
    rows = []
    for cid, row in new_rows.items():
        category = row["category"]
        if old_rows[cid]["category"] != category:
            raise ValueError(f"Category changed: {cid}")
        before = visible_payload(old / f"{cid}.json", category)
        after = visible_payload(new / f"{cid}.json", category)
        fields = sorted(
            key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
        )
        kind = classification(before, after, fields, category)
        record = {
            "case": cid,
            "category": category,
            "classification": kind,
            "changed_fields": fields,
            "old_visible_sha256": digest(before),
            "current_visible_sha256": digest(after),
        }
        if fields:
            record["details"] = {
                key: {"old": before.get(key), "current": after.get(key)} for key in fields
            }
        rows.append(record)
    return {
        "old_summary_sha256": hashlib.sha256((old / "summary.json").read_bytes()).hexdigest(),
        "current_summary_sha256": hashlib.sha256((new / "summary.json").read_bytes()).hexdigest(),
        "scope": "machine-visible comparison only; no human judgement carried forward",
        "changed": sum(item["classification"] != "unchanged" for item in rows),
        "unchanged": sum(item["classification"] == "unchanged" for item in rows),
        "procedural_stop_only": sum(
            item["classification"] == "procedural_stop_only" for item in rows
        ),
        "semantic_change": sum(item["classification"] == "semantic_change" for item in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing comparison; choose a fresh output")
    report = compare_runs(args.old, args.new)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("changed", "unchanged", "procedural_stop_only", "semantic_change")
            }
        )
    )


if __name__ == "__main__":
    main()
