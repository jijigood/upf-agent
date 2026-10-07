import copy

import pytest

from upf_agent.quality_repair import apply_corrections


def fixture():
    original = {
        "generation": "g",
        "images": [
            {
                "lines": [
                    {"text": "GRUB", "box": [0, 0, 10, 10], "score": 0.9},
                    {"text": "DEFAULT=0", "box": [11, 0, 30, 10], "score": 0.8},
                ]
            }
        ],
    }
    ledger = {
        "generation": "g",
        "version": "v2",
        "occlusions": [],
        "incomplete_images": [],
        "patches": [
            {
                "screenshot": 1,
                "line": 1,
                "before": "GRUB",
                "after": "GRUB_",
                "usable": True,
                "disposition": "corrected",
            }
        ],
        "reconstructed_rows": [{"screenshot": 1, "lines": [1, 2], "text": "GRUB_DEFAULT=0"}],
    }
    return original, ledger


def test_row_reconstruction_preserves_fragment_provenance():
    original, ledger = fixture()
    saved = copy.deepcopy(original)
    result = apply_corrections(original, ledger)
    image = result["images"][0]
    assert original == saved
    assert image["reconstructed_rows"][0]["text"] == "GRUB_DEFAULT=0"
    assert image["reconstructed_rows"][0]["boxes"] == [[0, 0, 10, 10], [11, 0, 30, 10]]
    assert image["lines"][0]["original_text"] == "GRUB"
    assert image["lines"][0]["score"] == 0.9
    assert not result["human_signoff"]


@pytest.mark.parametrize("error", ["bounds", "duplicate", "order", "text", "quarantined"])
def test_invalid_row_reconstruction_rejected(error):
    original, ledger = fixture()
    row = ledger["reconstructed_rows"][0]
    if error == "bounds":
        row["lines"] = [0, 2]
    elif error == "duplicate":
        row["lines"] = [1, 1]
    elif error == "order":
        row["lines"] = [2, 1]
    elif error == "text":
        row["text"] = "invented"
    else:
        ledger["occlusions"] = [{"screenshot": 1, "line": 2, "reason": "unreadable"}]
    with pytest.raises(ValueError):
        apply_corrections(original, ledger)
