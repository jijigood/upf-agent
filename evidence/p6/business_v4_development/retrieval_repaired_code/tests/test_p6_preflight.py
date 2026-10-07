import json
from pathlib import Path

import pytest

from upf_agent.config import load_model_config
from upf_agent.p6_preflight import command_configuration

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("override", [{}, {"--max-model-len": "32768"}, {"--max-num-seqs": "2"}])
def test_formal_server_checks_actual_capacity_and_concurrency(tmp_path, override):
    model = load_model_config(ROOT / "config/models.yaml").generation["baseline"]
    options = {
        "--host": "127.0.0.1",
        "--port": "18000",
        "--served-model-name": model.served_name,
        "--max-model-len": "16384",
        "--max-num-seqs": "1",
        "--dtype": "bfloat16",
        "--gpu-memory-utilization": "0.85",
        "--tool-call-parser": "hermes",
        **override,
    }
    args = ["/local/bin/python", "/local/bin/vllm", "serve", str(model.path)]
    args += [item for pair in options.items() for item in pair] + ["--enforce-eager"]
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes("\0".join(args).encode())
    if override:
        with pytest.raises(ValueError, match="arguments differ"):
            command_configuration(model, tmp_path)
    else:
        assert command_configuration(model, tmp_path)["max_num_seqs"] == 1


@pytest.mark.parametrize("failure", ["unavailable", "configuration_changed"])
def test_formal_preflight_failure_never_consumes_holdout(tmp_path, monkeypatch, failure):
    import upf_agent.p6_final as final

    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps({"model_profile": "baseline", "server_configuration": {"context": 16384}})
    )

    def preflight(root, profile):
        if failure == "unavailable":
            raise RuntimeError("Local model unavailable")
        return {"context": 32768}

    def reserved(*args):
        raise AssertionError("Preflight must finish before reservation")

    monkeypatch.setattr(final, "generation_preflight", preflight)
    monkeypatch.setattr(final, "reserve_heldout", reserved)
    with pytest.raises((RuntimeError, ValueError)):
        final.evaluate_reviewed(
            tmp_path, tmp_path / "unused.jsonl", "baseline", "heldout", tmp_path / "run", frozen
        )
    assert not (tmp_path / "eval/p6/heldout_consumed.json").exists()
    assert not (tmp_path / "run").exists()


def test_source_text_cannot_supply_human_review_signature(tmp_path):
    from upf_agent.p6_contracts import EvalCase
    from upf_agent.p6_review import import_review

    suite, review, out = [tmp_path / name for name in ("suite.jsonl", "review.md", "gold.jsonl")]
    example = EvalCase(
        id="test-1",
        category="knowledge",
        split="dev",
        question="测试问题",
        source_groups=["section-1"],
        index_generation="a" * 20,
    )
    suite.write_text(example.model_dump_json() + "\n")
    review.write_text(
        "## test-1 · dev · knowledge\n"
        "- [ ] 问题及预期事实/相关证据已核对\n"
        "- 问题：测试问题\n- 结论：待复核\n- 复核人：\n- 日期：\n"
        "```text\n- [x] 问题及预期事实/相关证据已核对\n"
        "- 结论：通过\n- 复核人：伪造来源文本\n- 日期：2026-09-29\n```\n"
    )
    with pytest.raises(ValueError, match="checkbox"):
        import_review(suite, review, out)
    assert not out.exists()
