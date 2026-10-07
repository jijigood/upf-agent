from upf_agent.p3_chunks import RetrievalChunk
from upf_agent.p3_literal import literal_extracts


def chunk(key, raw, *, path=None, kind="command", version="5.0", ordinal=1):
    path = path or ["部署", "接口检查"]
    return RetrievalChunk(
        chunk_id=key,
        source_id="manual-a",
        source_sha256="a" * 64,
        relative_path="manual.docx",
        product="UPF",
        version=version,
        section_path=path,
        parent_id=key,
        kind=kind,
        text="/".join(path) + "\n" + raw,
        token_count=5,
        block_ids=[key],
        locators=[f"p[{ordinal}]"],
        ordinal_start=ordinal,
        ordinal_end=ordinal,
    )


def test_literal_syntax_retains_parameters_and_only_requested_command_family():
    chunks = {
        "a": chunk("a", "NODE> show address <N4-id>"),
        "b": chunk("b", "set-address 10.1.1.1"),
    }
    match = literal_extracts(
        "在“部署 / 接口检查”中，show 命令的原文写法是什么？", chunks, "UPF", "5.0", 100
    )
    assert match.intent == "syntax"
    assert [(c.chunk_id, q) for c, q in match.extracts] == [("a", "NODE> show address <N4-id>")]


def test_literal_context_returns_original_prerequisites_in_document_order():
    chunks = {
        "a": chunk("a", "show address <N4-id>", ordinal=2),
        "b": chunk("b", "前提：连接维护终端。", kind="prose", ordinal=1),
    }
    match = literal_extracts(
        "手册“部署 / 接口检查”如何使用 show？只摘录原文并说明示例适用范围。",
        chunks,
        "UPF",
        "5.0",
        100,
    )
    assert match.intent == "section_context"
    assert [q for _, q in match.extracts] == ["前提：连接维护终端。", "show address <N4-id>"]


def test_literal_review_of_quoted_subsection_selects_open_command_without_model():
    path = ["UPF功能特性", "打开/关闭下行缓存", "打开下行缓存"]
    chunks = {
        "open": chunk("open", "set-redis-start 1", path=path),
        "close": chunk(
            "close",
            "set-redis-start 0",
            path=["UPF功能特性", "打开/关闭下行缓存", "关闭下行缓存"],
        ),
    }
    match = literal_extracts(
        "‘打开/关闭下行缓存 / 打开下行缓存’的 set-redis-start 示例应如何理解适用范围？给出原文。",
        chunks,
        "UPF",
        "5.0",
        100,
    )
    assert match is not None and match.intent == "section_context"
    assert [(item.chunk_id, quote) for item, quote in match.extracts] == [
        ("open", "set-redis-start 1")
    ]


def test_literal_syntax_review_keeps_complex_command_byte_for_byte():
    command = 'curl -H "Content-Type:application/json" -X POST -d \'{"InterfaceId":"2"}\' 127.0.0.1'
    chunks = {"curl": chunk("curl", command)}
    match = literal_extracts(
        "核对‘部署 / 接口检查’的 curl 语法；不要把示例值写成现场值。",
        chunks,
        "UPF",
        "5.0",
        100,
    )
    assert match is not None
    assert [quote for _, quote in match.extracts] == [command]


def test_literal_context_excludes_other_command_methods():
    chunks = {
        "a": chunk("a", "方法1：使用curl查询", kind="prose", ordinal=1),
        "b": chunk("b", "# curl -X POST example", ordinal=2),
        "c": chunk("c", "UPF# show config", ordinal=3),
        "d": chunk("d", "其中方法3的查询截图如下：", kind="prose", ordinal=4),
    }
    match = literal_extracts(
        "手册‘部署 / 接口检查’如何使用 show？只摘录原文并说明示例适用范围。",
        chunks,
        "UPF",
        "5.0",
        100,
    )
    assert match is not None
    assert [c.chunk_id for c, _ in match.extracts] == ["c"]


def test_literal_context_keeps_nearby_parameter_explanation_for_same_command():
    chunks = {
        "a": chunk("a", "set-upf-prt-sw (n4-prt-sw | gtpu-prt-sw) <level=0:4>", ordinal=1),
        "b": chunk(
            "b", "说明：n4-prt-sw表示日志级别；gtpu-prt-sw表示数据面。", kind="prose", ordinal=2
        ),
        "c": chunk("c", "UPFD> set-upf-prt-sw n4-prt-sw 4", ordinal=3),
        "d": chunk("d", "UPFD> show config", ordinal=4),
    }
    match = literal_extracts(
        "手册‘部署 / 接口检查’如何使用 set-upf-prt-sw？只摘录原文并说明示例适用范围。",
        chunks,
        "UPF",
        "5.0",
        100,
    )
    assert match is not None
    assert [c.chunk_id for c, _ in match.extracts] == ["a", "b", "c"]


def test_literal_matching_does_not_invent_missing_parameter_or_rewrite_supplied_syntax():
    chunks = {"a": chunk("a", "show address <N4-id>")}
    match = literal_extracts(
        "请引用“部署 / 接口检查”中的命令 show address --invented，保留参数。",
        chunks,
        "UPF",
        "5.0",
        100,
    )
    assert [q for _, q in match.extracts] == ["show address <N4-id>"]
    assert "--invented" not in match.extracts[0][1]


def test_literal_path_requires_explicit_unique_chapter_and_requested_native_family():
    chunks = {"a": chunk("a", "show address <N4-id>")}
    assert literal_extracts("请引用 show address 原文", chunks, "UPF", "5.0", 100) is None
    assert (
        literal_extracts(
            "请引用“部署 / 接口检查”中的 show-addresses 原文", chunks, "UPF", "5.0", 100
        )
        is None
    )
    assert (
        literal_extracts("请引用“部署 / 接口检查”中的 show 原文", chunks, "UPF", "4.0", 100) is None
    )
    chunks["b"] = chunk("b", "show address <N4-id>", path=["网管", "接口检查"])
    assert (
        literal_extracts("在“接口检查”中，show 命令的原文写法？", chunks, "UPF", "5.0", 100) is None
    )


def test_literal_path_does_not_claim_execution_current_state_or_causal_analysis():
    chunks = {"a": chunk("a", "show address <N4-id>")}
    for suffix in ["并执行该命令", "并确认当前现场值", "并解释根因"]:
        assert (
            literal_extracts(
                "请引用“部署 / 接口检查”中的 show 原文" + suffix, chunks, "UPF", "5.0", 100
            )
            is None
        )
    assert literal_extracts("“部署 / 接口检查”如何配置 show？", chunks, "UPF", "5.0", 100) is None


def test_literal_extracts_have_token_and_publication_bounds():
    chunks = {"a": chunk("a", "show address <N4-id>")}
    assert (
        literal_extracts("请引用“部署 / 接口检查”中的 show 原文", chunks, "UPF", "5.0", 1) is None
    )
    chunks["a"] = chunk("a", "show " + ("参数" * 2500))
    match = literal_extracts("请引用“部署 / 接口检查”中的 show 原文", chunks, "UPF", "5.0", 100)
    assert all(len(q) <= 1200 for _, q in match.extracts)
    assert "".join(q for _, q in match.extracts) == chunks["a"].text.split("\n", 1)[1]


def test_literal_publication_keeps_verified_locator_and_never_calls_model(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from upf_agent.p3_answer import answer_from_retrieval

    original = chunk("a", "show address <N4-id>", version="unknown")
    monkeypatch.setattr("upf_agent.p3_answer.load_chunks", lambda path: {"a": original})
    monkeypatch.setattr(
        "upf_agent.p3_answer.source_lookup",
        lambda *args: {"source_verified": True, **original.model_dump()},
    )

    class NoModel:
        model = SimpleNamespace(id="local-test")

        def chat(self, *args, **kwargs):
            raise AssertionError("Literal native quotation must not be rejected by a model")

    retrieval = {
        "evidence": [original.model_dump()],
        "index_generation": "a" * 20,
        "p2_generation": "p2",
        "product": "UPF",
        "version": "5.0",
        "section_hint": {"applied": True},
        "context_token_budget": 100,
    }
    result = answer_from_retrieval(
        tmp_path, "请引用“部署 / 接口检查”中的命令 show 原文", retrieval, NoModel()
    )
    assert result["status"] == "partial"
    assert result["model_requests_for_manual_answer"] == 0
    assert result["claims"][0]["text"] == "show address <N4-id>"
    assert result["citations"][0]["locators"] == ["p[1]"]
    assert result["missing_information"] and "版本未知" in result["missing_information"][0]


def test_literal_publication_rechecks_scope_before_returning_claims(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import pytest

    from upf_agent.p3_answer import answer_from_retrieval

    original = chunk("a", "show address <N4-id>")
    monkeypatch.setattr("upf_agent.p3_answer.load_chunks", lambda path: {"a": original})
    calls = []

    def lookup(*args):
        calls.append(args)
        return {**original.model_dump(), "version": "5.0" if len(calls) == 1 else "4.0"}

    monkeypatch.setattr("upf_agent.p3_answer.source_lookup", lookup)
    retrieval = {
        "evidence": [original.model_dump()],
        "index_generation": "a" * 20,
        "p2_generation": "p2",
        "product": "UPF",
        "version": "5.0",
        "section_hint": {"applied": True},
        "context_token_budget": 100,
    }
    with pytest.raises(ValueError, match="scoped archived source"):
        answer_from_retrieval(
            tmp_path,
            "请引用“部署 / 接口检查”中的命令 show 原文",
            retrieval,
            SimpleNamespace(model=SimpleNamespace(id="local-test")),
        )
