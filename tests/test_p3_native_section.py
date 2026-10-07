from upf_agent.p3_chunks import RetrievalChunk
from upf_agent.p3_native_section import (
    NativeSectionExtracts,
    adjacent_case_overlap,
    empty_collection_sections,
    interface_conflicts,
    native_section_extracts,
    native_source_gap_notes,
)


def chunk(key, path, body, *, kind="prose", source="manual-a", ordinal=1, tokens=10):
    return RetrievalChunk(
        chunk_id=key,
        source_id=source,
        source_sha256="a" * 64,
        relative_path="manual.docx",
        product="UPF",
        version="unknown",
        section_path=path,
        parent_id=key,
        kind=kind,
        text="/".join(path) + "\n" + body,
        token_count=tokens,
        block_ids=[key],
        locators=[f"p[{ordinal}]"],
        ordinal_start=ordinal,
        ordinal_end=ordinal,
    )


def test_native_section_keeps_entire_related_table_and_ignores_delivery_sibling():
    chunks = {
        "delivery": chunk("delivery", ["交付配置", "接口地址配置"], "此处仅是交付步骤"),
        "prose": chunk(
            "prose", ["产品配置", "网管配置管理", "接口地址配置"], "配置接口地址功能。", ordinal=2
        ),
        "table1": chunk(
            "table1",
            ["产品配置", "网管配置管理", "接口地址配置", "功能配置"],
            "参数名称：接口ID",
            kind="table",
            ordinal=3,
        ),
        "table2": chunk(
            "table2",
            ["产品配置", "网管配置管理", "接口地址配置", "功能配置"],
            "参数名称：IPv6前缀长度",
            kind="table",
            ordinal=4,
        ),
    }
    result = native_section_extracts(
        "依据手册原生文字，UPF接口地址配置功能有哪些说明与适用范围？仅见截图或缺少字段时说明缺口；不要报告当前现场数值。",
        chunks,
        "UPF",
        None,
        100,
    )
    assert result is not None
    assert [c.chunk_id for c, _ in result.extracts] == ["prose", "table1", "table2"]
    assert result.method == "same_name_native_table_disambiguation"


def test_native_section_preserves_historical_symptoms_collection_and_solution():
    prefix = ["故障案例", "某历史故障案例"]
    chunks = {
        "symptom": chunk("symptom", [*prefix, "症状现象"], "upfd pod crash", ordinal=1),
        "collect": chunk("collect", [*prefix, "信息收集"], "检查license授权文件", ordinal=2),
        "solve": chunk("solve", [*prefix, "解决方法"], "重新授权并重启", ordinal=3),
        "other": chunk("other", ["故障案例", "另一个案例"], "不相关", ordinal=4),
    }
    result = native_section_extracts(
        "在手册的‘某历史故障案例’中，症状、信息收集和处理建议是什么？区分历史案例与当前故障。",
        chunks,
        "UPF",
        None,
        100,
    )
    assert result is not None
    assert [c.chunk_id for c, _ in result.extracts] == ["symptom", "collect", "solve"]


def test_explicit_historical_case_works_without_word_manual():
    path = ["系统平台故障案例", "k8s 异常：大部分pod处于Evicted状态"]
    chunks = {
        "symptom": chunk("symptom", [*path, "症状现象"], "pod Evicted", ordinal=1),
        "collection": chunk("collection", [*path, "信息收集"], "kubectl get pods -A", ordinal=2),
        "solution": chunk("solution", [*path, "解决方法"], "清理磁盘", ordinal=3),
        "other": chunk("other", ["系统平台故障案例", "其他故障"], "无关", ordinal=4),
    }
    result = native_section_extracts(
        "引用‘k8s 异常：大部分pod处于Evicted状态’的排查依据与处理前提，"
        "指出它不能直接证明当前故障。",
        chunks,
        "UPF",
        None,
        100,
    )
    assert result is not None
    assert [item.chunk_id for item, _ in result.extracts] == [
        "symptom",
        "collection",
        "solution",
    ]


def test_native_case_marks_heading_only_information_collection(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from upf_agent.p3_answer import answer_from_retrieval

    path = ["业务故障案例", "upfmgr n4switch后重启upfd pod一直crash"]
    chunks = {
        "symptom": chunk("symptom", [*path, "症状现象"], "重启后一直crash", ordinal=1),
        "collect": chunk("collect", [*path, "3.5.2. 信息收集"], "3.5.2. 信息收集", ordinal=2),
        "solve": chunk("solve", [*path, "3.5.3. 解决方法"], "检查license", ordinal=3),
    }
    question = "手册‘upfmgr n4switch后重启upfd pod一直crash’：症状、信息收集和处理建议？"
    native = native_section_extracts(question, chunks, "UPF", None, 100)
    assert native is not None
    assert empty_collection_sections(native) == [
        {"chunk_id": "collect", "section_title": "3.5.2. 信息收集"}
    ]
    monkeypatch.setattr("upf_agent.p3_answer.load_chunks", lambda path: chunks)
    monkeypatch.setattr(
        "upf_agent.p3_answer.source_lookup",
        lambda root, generation, cid: {"source_verified": True, **chunks[cid].model_dump()},
    )

    class NoModel:
        model = SimpleNamespace(id="local-test")

        def chat(self, *args, **kwargs):
            raise AssertionError("Native case publication must not need a model")

    retrieval = {
        "evidence": [item.model_dump() for item in chunks.values()],
        "index_generation": "a" * 20,
        "p2_generation": "p2",
        "product": "UPF",
        "version": None,
        "context_token_budget": 100,
        "native_section": {
            "source_id": native.source_id,
            "path": list(native.path),
            "complete_chunk_ids": list(chunks),
        },
    }
    answer = answer_from_retrieval(tmp_path, question, retrieval, NoModel())
    assert "只有标题、没有步骤正文" in answer["missing_information"][1]
    assert answer["audit"]["native_section"]["empty_source_sections"] == [
        {"chunk_id": "collect", "section_title": "3.5.2. 信息收集"}
    ]


def test_native_collection_with_text_is_not_marked_empty():
    path = ["故障案例", "某历史故障案例"]
    chunks = {
        "collect": chunk("collect", [*path, "信息收集"], "检查license授权文件"),
    }
    native = native_section_extracts(
        "在手册的‘某历史故障案例’中，信息收集是什么？", chunks, "UPF", None, 100
    )
    assert native is not None
    assert empty_collection_sections(native) == []


def test_native_source_gaps_distinguish_blank_cells_and_unreadable_figure():
    path = ("产品配置", "网管配置管理", "DNN带宽配置")
    cells = chunk(
        "cells",
        [*path, "功能配置"],
        "表头：参数名称\t是否必选\t默认值\t参数说明\n"
        "参数名称：DNN | 默认值：无\n"
        "参数名称：带宽 | 是否必选：可选 | 默认值：1024 | 参数说明：限速",
        kind="table",
    )
    figure = chunk("figure", [*path, "功能配置"], "在如下图中配置字段。")
    native = NativeSectionExtracts(
        source_id="manual-a",
        path=path,
        method="test",
        extracts=[
            (cells, cells.text.split("\n", 1)[1]),
            (figure, figure.text.split("\n", 1)[1]),
        ],
    )
    notes = native_source_gap_notes(native)
    assert any("DNN" in note and "是否必选、参数说明栏为空" in note for note in notes)
    assert not any("带宽" in note for note in notes)
    assert any("图中未写出" in note for note in notes)


def test_adjacent_history_cases_with_shared_command_keep_distinct_errors():
    own = chunk(
        "own",
        ["系统平台故障案例", "upfmgr hc执行报错", "症状现象"],
        "upfmgr hc\n/root/example_IP.conf: No such file or directory",
        ordinal=20,
    )
    neighbor = chunk(
        "neighbor",
        ["系统平台故障案例", "upfmgr hc显示N3/N6地址为空", "信息收集"],
        "upfmgr hc\nGet GTPU Node IP Failed!\nN3/N6地址异常",
        ordinal=30,
    )
    unrelated = chunk(
        "unrelated",
        ["系统平台故障案例", "upfmgr hc另一个环境", "信息收集"],
        "upfmgr hc\nGet GTPU Node IP Failed!",
        source="other-manual",
        ordinal=21,
    )
    native = NativeSectionExtracts(
        "manual-a", ("系统平台故障案例", "upfmgr hc执行报错"), "test",
        [(own, own.text.split("\n", 1)[1])],
    )
    overlap = adjacent_case_overlap(
        native, {c.chunk_id: c for c in (own, neighbor, unrelated)}
    )
    assert len(overlap) == 1
    assert overlap[0]["neighbor_chunk_id"] == "neighbor"
    assert "/root/example_IP.conf" in overlap[0]["note"]
    assert "未确认两节属于同一次故障" in overlap[0]["note"]
    assert adjacent_case_overlap(native, {"own": own}) == []


def test_native_section_fails_closed_for_ambiguous_tables_budget_and_unscoped_query():
    first = chunk("first", ["产品", "接口地址配置"], "A", kind="table", tokens=25)
    second = chunk("second", ["交付", "接口地址配置"], "B", kind="table", tokens=25)
    chunks = {"first": first, "second": second}
    question = "依据手册原生文字，接口地址配置功能有哪些说明？"
    assert native_section_extracts(question, chunks, "UPF", None, 100) is None
    assert native_section_extracts(question, {"first": first}, "UPF", None, 20) is None
    assert (
        native_section_extracts("当前接口地址配置是什么？", {"first": first}, "UPF", None, 100)
        is None
    )


def test_native_section_publishes_every_archived_parameter_without_model(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from upf_agent.p3_answer import answer_from_retrieval

    chunks = {
        "intro": chunk("intro", ["网管", "接口配置"], "接口配置功能描述。", ordinal=1),
        "ipv4": chunk(
            "ipv4", ["网管", "接口配置", "配置项"], "IPv4地址、子网掩码。", kind="table", ordinal=2
        ),
        "ipv6": chunk(
            "ipv6", ["网管", "接口配置", "配置项"], "IPv6地址、前缀长度。", kind="table", ordinal=3
        ),
    }
    question = "依据手册原生文字，接口配置功能有哪些说明与适用范围？"
    native = native_section_extracts(question, chunks, "UPF", None, 100)
    assert native is not None
    monkeypatch.setattr("upf_agent.p3_answer.load_chunks", lambda path: chunks)
    monkeypatch.setattr(
        "upf_agent.p3_answer.source_lookup",
        lambda root, generation, cid: {"source_verified": True, **chunks[cid].model_dump()},
    )

    class NoModel:
        model = SimpleNamespace(id="local-test")

        def chat(self, *args, **kwargs):
            raise AssertionError("Complete native chapter must not need model selection")

    retrieval = {
        "evidence": [item.model_dump() for item in chunks.values()],
        "index_generation": "a" * 20,
        "p2_generation": "p2",
        "product": "UPF",
        "version": None,
        "context_token_budget": 100,
        "native_section": {
            "source_id": native.source_id,
            "path": list(native.path),
            "complete_chunk_ids": ["intro", "ipv4", "ipv6"],
        },
    }
    answer = answer_from_retrieval(tmp_path, question, retrieval, NoModel())
    assert answer["status"] == "partial"
    assert answer["answer_mode"] == "native_complete_section_extract"
    assert answer["model_requests_for_manual_answer"] == 0
    text = "\n".join(claim["text"] for claim in answer["claims"])
    assert "子网掩码" in text and "前缀长度" in text
    assert "示例适用范围" in answer["missing_information"][0]
    assert {c["chunk_id"] for c in answer["citations"]} == set(chunks)


def test_native_case_flags_solution_interface_mismatch_without_rewriting_source(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from upf_agent.p3_answer import answer_from_retrieval

    path = ["故障案例", "UPF N3 IPv4 接口地址到网关链路异常"]
    chunks = {
        "symptom": chunk("symptom", [*path, "症状现象"], "N3网关不通", ordinal=1),
        "collect": chunk("collect", [*path, "信息收集"], "检查N4邻接作为参考", ordinal=2),
        "solution": chunk(
            "solution",
            [*path, "解决方法"],
            "若N3网关不通，排查交换机接口有无收到 N4接口的数据报文。",
            ordinal=3,
        ),
    }
    question = (
        "在手册的‘UPF N3 IPv4 接口地址到网关链路异常’案例中，"
        "症状和处理建议是什么？原件有接口矛盾时指出。"
    )
    native = native_section_extracts(question, chunks, "UPF", None, 100)
    assert native is not None
    assert interface_conflicts(native) == [
        {
            "chunk_id": "solution",
            "title_interface": "N3",
            "solution_interface": "N4",
            "source_quote": "排查交换机接口有无收到 N4接口的数据报文",
        }
    ]
    monkeypatch.setattr("upf_agent.p3_answer.load_chunks", lambda path: chunks)
    monkeypatch.setattr(
        "upf_agent.p3_answer.source_lookup",
        lambda root, generation, cid: {"source_verified": True, **chunks[cid].model_dump()},
    )

    class NoModel:
        model = SimpleNamespace(id="local-test")

        def chat(self, *args, **kwargs):
            raise AssertionError("Native case publication must not need a model")

    retrieval = {
        "evidence": [item.model_dump() for item in chunks.values()],
        "index_generation": "a" * 20,
        "p2_generation": "p2",
        "product": "UPF",
        "version": None,
        "context_token_budget": 100,
        "native_section": {
            "source_id": native.source_id,
            "path": list(native.path),
            "complete_chunk_ids": list(chunks),
        },
    }
    answer = answer_from_retrieval(tmp_path, question, retrieval, NoModel())
    assert "原文疑点" in answer["missing_information"][1]
    assert "N3" in answer["missing_information"][1]
    assert "N4" in answer["missing_information"][1]
    assert answer["audit"]["native_section"]["source_conflicts"][0]["chunk_id"] == "solution"
    assert "N4接口" in "\n".join(claim["text"] for claim in answer["claims"])


def test_native_case_does_not_flag_matching_interface_or_other_collection_mentions():
    path = ["故障案例", "UPF N4 IPv6 接口地址到网关链路异常"]
    chunks = {
        "collect": chunk("collect", [*path, "信息收集"], "也检查N6链路", ordinal=1),
        "solution": chunk(
            "solution", [*path, "解决方法"], "排查交换机接口有无收到 N4接口的数据报文。", ordinal=2
        ),
    }
    native = native_section_extracts(
        "在手册的‘UPF N4 IPv6 接口地址到网关链路异常’案例中，症状和处理建议是什么？",
        chunks,
        "UPF",
        None,
        100,
    )
    assert native is not None
    assert interface_conflicts(native) == []
