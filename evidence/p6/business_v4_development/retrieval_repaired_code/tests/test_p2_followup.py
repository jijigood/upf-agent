import runpy
from pathlib import Path

import pytest

parse = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/sync_p2_review.py"))[
    "followup_records"
]


def sheet():
    return "\n".join(
        f"## 截图 {number:02d}\n\n"
        "- [x] 修正后的可见内容与原图一致\n"
        "- 定位：`p[1]/image[1]`\n"
        "- 结论：通过（本轮通过，附残留问题不影响签署）\n"
        "- 问题：PEDM 应为 PFDM\n"
        for number in (1, 7, 17, 20)
    )


def test_qualified_pass_preserves_note_and_exact_source():
    records = parse(sheet())
    assert records[7]["normalized_conclusion"] == "通过"
    assert records[7]["问题"] == "PEDM 应为 PFDM"
    assert records[7]["定位"] == "`p[1]/image[1]`"
    assert "残留问题不影响签署" in records[7]["submitted_section"]


@pytest.mark.parametrize("failure", ["duplicate", "missing", "checkbox", "ambiguous", "locator"])
def test_invalid_followup_does_not_become_pass(failure):
    text = sheet()
    if failure == "duplicate":
        text = text.replace("## 截图 20", "## 截图 07")
    elif failure == "missing":
        text = text.split("## 截图 20")[0]
    elif failure == "checkbox":
        text = text.replace("- [x]", "- [ ]", 1)
    elif failure == "ambiguous":
        text = text.replace("通过（本轮通过，附残留问题不影响签署）", "通过不通过", 1)
    else:
        text = text.replace("- 定位：`p[1]/image[1]`", "", 1)
    with pytest.raises(ValueError):
        parse(text)
