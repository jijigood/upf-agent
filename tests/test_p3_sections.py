from upf_agent.p3_chunks import RetrievalChunk
from upf_agent.p3_sections import belongs, resolve_section, section_routes


def chunk(
    key,
    path,
    *,
    product="UPF",
    version="5.0",
    source="manual-a",
    text="原生内容",
    kind="prose",
    ordinal=1,
):
    return RetrievalChunk(
        chunk_id=key,
        source_id=source,
        source_sha256="a" * 64,
        relative_path="manual.docx",
        product=product,
        version=version,
        section_path=path,
        parent_id=key,
        kind=kind,
        text=text,
        token_count=5,
        block_ids=[key],
        locators=[f"p[{ordinal}]"],
        ordinal_start=ordinal,
        ordinal_end=ordinal,
    )


def test_explicit_hierarchy_distinguishes_same_leaf_and_includes_descendants():
    chunks = {
        "a": chunk("a", ["部署", "接口检查", "N4检查"]),
        "b": chunk("b", ["故障", "N4检查"]),
        "c": chunk("c", ["部署", "接口检查", "N4检查", "示例"]),
    }
    hint = resolve_section("在“接口检查 / N4检查”中查看配置", chunks, "UPF", "5.0")
    assert hint.path == ("部署", "接口检查", "N4检查")
    assert hint.method == "explicit_section"
    assert belongs(chunks["a"], hint) and belongs(chunks["c"], hint)
    assert not belongs(chunks["b"], hint)


def test_unique_native_title_ignores_spacing_and_case():
    chunks = {
        "a": chunk("a", ["网管", "CP IP地址配置", "功能说明"]),
        "b": chunk("b", ["网管", "其他配置"]),
    }
    hint = resolve_section("解释 cp ip地址配置的说明", chunks, "UPF", None)
    assert hint.path == ("网管", "CP IP地址配置")
    assert hint.method == "unique_section_title"


def test_numbered_manual_heading_matches_unnumbered_query_without_historical_collision():
    chunks = {
        "active": chunk(
            "active",
            ["7 故障管理", "7.1 实时告警", "7.1.1 告警查询"],
            product="5GC网管",
            version="V1.3",
        ),
        "history": chunk(
            "history",
            ["7 故障管理", "7.2 历史告警", "7.2.1 历史告警查询"],
            product="5GC网管",
            version="V1.3",
        ),
    }
    hint = resolve_section("5GC网管实时告警查询有哪些说明？", chunks, "5GC网管", "V1.3")
    assert hint is not None
    assert hint.path == ("7 故障管理", "7.1 实时告警", "7.1.1 告警查询")


def test_ambiguous_section_stays_unresolved_without_native_command():
    chunks = {
        "a": chunk("a", ["CLI", "系统能力", "功能配置"]),
        "b": chunk("b", ["网管", "系统能力", "功能配置"]),
    }
    assert resolve_section("在‘系统能力 / 功能配置’中查看", chunks, "UPF", None) is None
    assert resolve_section("配置说明", chunks, "UPF", None) is None


def test_literal_native_command_disambiguates_duplicate_section():
    chunks = {
        "a": chunk(
            "a", ["CLI", "系统能力", "功能配置"], text="set-test-mode <level=0:4>", kind="command"
        ),
        "b": chunk("b", ["网管", "系统能力", "功能配置"], text="图形界面设置"),
    }
    hint = resolve_section(
        "在‘系统能力 / 功能配置’中，set-test-mode 原文是什么", chunks, "UPF", None
    )
    assert hint.path == ("CLI", "系统能力", "功能配置")
    assert hint.method == "explicit_section_native_command"
    chunks["b"] = chunk(
        "b", ["网管", "系统能力", "功能配置"], text="set-test-mode <level=0:4>", kind="command"
    )
    assert (
        resolve_section("在‘系统能力 / 功能配置’中，set-test-mode 原文是什么", chunks, "UPF", None)
        is None
    )
    assert (
        resolve_section("在‘系统能力 / 功能配置’中，set-test-modes 原文是什么", chunks, "UPF", None)
        is None
    )


def test_section_resolution_enforces_product_version_and_source_identity():
    chunks = {
        "a": chunk("a", ["配置", "接口检查"]),
        "b": chunk("b", ["配置", "接口检查"], version="4.0"),
        "c": chunk("c", ["配置", "接口检查"], product="5GC网管"),
    }
    hint = resolve_section("接口检查", chunks, "UPF", "5.0")
    assert hint.source_id == "manual-a"
    chunks["d"] = chunk("d", ["配置", "接口检查"], source="manual-b")
    assert resolve_section("接口检查", chunks, "UPF", "5.0") is None
    assert resolve_section("接口检查", chunks, "UPF", "9.9") is None


def test_section_route_is_bounded_ordered_and_preserves_fallback():
    chunks = {
        "a": chunk("a", ["部署", "接口检查"], ordinal=2),
        "b": chunk("b", ["部署", "接口检查"], ordinal=1),
        "c": chunk("c", ["其他"], ordinal=3),
    }
    routes = {
        "exact": [("c", 9)],
        "bm25": [("a", 1), ("c", 0.5)],
        "dense": [("c", 0.8), ("b", 0.7)],
    }
    hint = resolve_section("在“部署 / 接口检查”中查看", chunks, "UPF", "5.0")
    bounded = section_routes(routes, chunks, hint, 2)
    assert bounded["section"] == [("b", 1.0), ("a", 1.0)]
    assert bounded["exact"] == [] and bounded["bm25"] == [("a", 1)]
    assert bounded["dense"] == [("b", 0.7)]
    assert section_routes(routes, chunks, None, 2) == routes
    assert section_routes(routes, chunks, hint, 1) == routes


def test_section_route_does_not_reintroduce_excluded_version():
    chunks = {
        "a": chunk("a", ["配置", "接口检查"]),
        "b": chunk("b", ["配置", "接口检查"], version="4.0"),
    }
    hint = resolve_section("接口检查", chunks, "UPF", "5.0")
    routes = section_routes({"dense": [("a", 0.9)]}, chunks, hint, 2, "UPF", "5.0")
    assert routes["section"] == [("a", 1.0)]


def test_multiple_explicit_chapters_do_not_collapse_into_one_command_branch():
    chunks = {
        "a": chunk(
            "a", ["CLI", "系统能力", "功能配置"], text="set-test-mode <level=0:4>", kind="command"
        ),
        "b": chunk("b", ["网管", "其他配置"]),
    }
    assert (
        resolve_section(
            "比较“CLI / 系统能力 / 功能配置”与“网管 / 其他配置”的 set-test-mode",
            chunks,
            "UPF",
            None,
        )
        is None
    )


def test_same_title_candidates_preserve_both_chapters_and_original_routes():
    from upf_agent.p3_sections import related_section_routes

    chunks = {
        "a": chunk("a", ["交付", "接口检查"]),
        "b": chunk("b", ["产品", "接口检查"]),
        "c": chunk("c", ["其他"]),
    }
    original = {"dense": [("c", 0.9)]}
    routes, hints = related_section_routes("介绍接口检查", original, chunks, "UPF", "5.0", 2)
    assert len(hints) == 2
    assert routes["dense"] == original["dense"]
    assert {key for key, _ in routes["section"]} == {"a", "b"}
    assert related_section_routes("介绍接口检查", original, chunks, "UPF", "5.0", 1) == (
        original,
        [],
    )


def test_related_candidates_respect_scope_and_different_titles_do_not_get_collapsed():
    from upf_agent.p3_sections import related_section_routes

    chunks = {
        "a": chunk("a", ["交付", "接口检查"]),
        "b": chunk("b", ["产品", "接口检查"], version="4.0"),
        "c": chunk("c", ["产品", "告警检查"]),
    }
    original = {"dense": [("a", 0.9)]}
    assert related_section_routes("介绍接口检查", original, chunks, "UPF", "5.0", 3) == (
        original,
        [],
    )
    assert related_section_routes("比较接口检查和告警检查", original, chunks, "UPF", None, 3) == (
        original,
        [],
    )
