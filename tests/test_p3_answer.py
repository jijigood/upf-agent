import json
from types import SimpleNamespace

import pytest

from upf_agent.p3_ablation import ranking_metrics
from upf_agent.p3_answer import DraftAnswer, answer_from_retrieval, source_lookup, validate_quotes


def draft(chunk_id="c1", quote="show cp-node", text="使用 `show cp-node` 查询。"):
    return DraftAnswer.model_validate(
        {
            "status": "completed",
            "claims": [{"text": text, "citations": [{"chunk_id": chunk_id, "quote": quote}]}],
            "missing_information": [],
        }
    )


def test_quote_and_command_must_be_supported():
    evidence = [{"chunk_id": "c1", "text": "UPFD> show cp-node"}]
    validate_quotes(draft(), evidence)
    with pytest.raises(ValueError, match="citation"):
        validate_quotes(draft(chunk_id="invented"), evidence)
    with pytest.raises(ValueError, match="citation"):
        validate_quotes(draft(quote="show cp-node --force"), evidence)
    with pytest.raises(ValueError, match="command"):
        validate_quotes(draft(text="执行 `show cp-node --force`"), evidence)
    with pytest.raises(ValueError, match="no supported"):
        validate_quotes(
            DraftAnswer(status="completed", claims=[], missing_information=[]), evidence
        )


class FakeClient:
    model = SimpleNamespace(id="local-test")

    def __init__(self, replies, gate=None):
        self.gate = gate or {
            "has_relevant_evidence": True,
            "relevant_quote_ids": ["S1Q1"],
            "missing_information": [],
        }
        self.replies = iter(replies)
        self.calls = 0

    def chat(self, messages, **_kwargs):
        self.calls += 1
        if _kwargs.get("response_format", {}).get("json_schema", {}).get("name") == "evidence_gate":
            return {"content": json.dumps(self.gate)}
        reply = next(self.replies)
        if "claims" in reply and any("citations" in c for c in reply["claims"]):
            evidence = json.loads(messages[1]["content"])["evidence"]
            units = [unit for item in evidence for unit in item["quotes"]]
            reply = {
                **reply,
                "claims": [
                    {
                        "text": c["text"],
                        "quote_ids": [
                            next(
                                (u["quote_id"] for u in units if cite["quote"] in u["text"]),
                                "invalid",
                            )
                            for cite in c["citations"]
                        ],
                    }
                    for c in reply["claims"]
                ],
            }
        return {"content": json.dumps(reply)}


def retrieval(evidence):
    return {
        "evidence": evidence,
        "index_generation": "a" * 20,
        "p2_generation": "p2",
        "product": "UPF",
        "version": "5.0",
    }


def test_empty_retrieval_never_calls_model(tmp_path):
    client = FakeClient([])
    result = answer_from_retrieval(tmp_path, "question", retrieval([]), client)
    assert result["status"] == "missing_source"
    assert client.calls == 0


@pytest.mark.parametrize("requested_version", ["9.9", "8.7"])
@pytest.mark.parametrize("suffix", ["", "；请引用原文并执行配置"])
def test_unknown_source_version_cannot_publish_target_version_configuration(
    monkeypatch, tmp_path, requested_version, suffix
):
    original = {
        "chunk_id": "c1",
        "text": "配置N3地址，子网掩码填入255.255.255.0。",
        "product": "UPF",
        "version": "unknown",
        "section_path": ["接口配置"],
    }
    monkeypatch.setattr("upf_agent.p3_answer.source_lookup", lambda *args: original)
    client = FakeClient(
        [
            draft(quote=original["text"], text=original["text"]).model_dump(),
            {"supported": [True]},
        ]
    )
    result = answer_from_retrieval(
        tmp_path,
        f"{requested_version} 版本如何配置接口？{suffix}",
        {**retrieval([original]), "version": requested_version},
        client,
    )
    assert result["status"] == "missing_source"
    assert result["claims"] == result["citations"] == []
    assert requested_version in " ".join(result["missing_information"])
    assert client.calls == 0


def test_matching_source_version_still_allows_configuration(monkeypatch, tmp_path):
    original = {
        "chunk_id": "c1",
        "text": "show cp-node",
        "product": "UPF",
        "version": "5.0",
        "section_path": ["接口配置"],
    }
    monkeypatch.setattr("upf_agent.p3_answer.source_lookup", lambda *args: original)
    client = FakeClient([draft().model_dump(), {"supported": [True]}])
    result = answer_from_retrieval(tmp_path, "如何配置接口", retrieval([original]), client)
    assert result["status"] == "completed"
    assert result["claims"]


def test_historical_case_cannot_be_presented_as_current_fault(monkeypatch, tmp_path):
    original = {
        "chunk_id": "c1",
        "text": "upfmgr hc执行显示N3和N6地址为空",
        "product": "UPF",
        "version": "5.0",
        "section_path": ["系统平台故障案例", "N3和N6地址为空"],
    }
    monkeypatch.setattr("upf_agent.p3_answer.source_lookup", lambda *args: original)
    client = FakeClient(
        [
            draft(quote=original["text"], text=original["text"]).model_dump(),
            {"supported": [True]},
        ]
    )
    result = answer_from_retrieval(
        tmp_path, "该历史案例如何排查？能否认定为当前故障？", retrieval([original]), client
    )
    assert result["status"] == "partial"
    assert result["claims"]
    assert "历史故障案例" in " ".join(result["missing_information"])
    assert "当前日志" in " ".join(result["missing_information"])


def test_unknown_version_partial_and_semantic_rejection(monkeypatch, tmp_path):
    evidence = {
        "chunk_id": "c1",
        "text": "show cp-node",
        "product": "UPF",
        "version": "unknown",
        "section_path": ["查询cp node信息"],
    }
    monkeypatch.setattr("upf_agent.p3_answer.source_lookup", lambda *args: evidence)
    result = answer_from_retrieval(
        tmp_path,
        "query",
        retrieval([evidence]),
        FakeClient([draft().model_dump(), {"supported": [True]}]),
    )
    assert result["status"] == "partial"
    assert result["citations"] == [evidence]
    assert "版本未知" in result["missing_information"][0]
    rejected = answer_from_retrieval(
        tmp_path,
        "query",
        retrieval([evidence]),
        FakeClient([draft().model_dump(), {"supported": [False]}]),
    )
    assert rejected["status"] == "missing_source"
    assert not rejected["claims"] and not rejected["citations"]


def test_hallucinated_quote_and_bad_json_fail_closed(monkeypatch, tmp_path):
    evidence = {
        "chunk_id": "c1",
        "text": "show cp-node",
        "product": "UPF",
        "version": "5.0",
        "section_path": ["test"],
    }
    monkeypatch.setattr("upf_agent.p3_answer.source_lookup", lambda *args: evidence)
    for reply in [draft(quote="set all").model_dump(), {"nonsense": True}]:
        client = FakeClient([reply, reply])
        result = answer_from_retrieval(tmp_path, "query", retrieval([evidence]), client)
        assert result["status"] == "failed"
        assert result["claims"] == []
        assert client.calls == 3


def test_source_lookup_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="generation"):
        source_lookup(tmp_path, "../../../etc", "c1")


def test_recall_hit_ndcg_are_distinct_and_deduplicated():
    scores = ranking_metrics(["a", "a", "x"], {"a": 1, "b": 1})
    assert scores["recall_at_20"] == 0.5
    assert scores["hit_at_20"] == 1
    assert 0 < scores["ndcg_at_8"] < 1
    assert ranking_metrics(["b", "a"], {"a": 1, "b": 2})["ndcg_at_8"] == 1
    with pytest.raises(ValueError):
        ranking_metrics([], {})


def test_source_hash_guard_detects_modified_manual(tmp_path):
    import hashlib

    from upf_agent.p3_chunks import RetrievalChunk

    generation = "a" * 20
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    raw = raw_dir / "manual.docx"
    raw.write_bytes(b"original manual")
    chunk = RetrievalChunk(
        chunk_id="c1",
        source_id="manual-1",
        source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        relative_path="manual.docx",
        product="UPF",
        version="5.0",
        section_path=["chapter"],
        parent_id="p1",
        kind="prose",
        text="show cp-node",
        token_count=4,
        block_ids=["b1"],
        locators=["p[1]"],
        ordinal_start=1,
        ordinal_end=1,
    )
    index = tmp_path / "data/p3/indexes" / generation
    index.mkdir(parents=True)
    content = chunk.model_dump_json() + "\n"
    (index / "chunks.jsonl").write_text(content)
    (index / "manifest.json").write_text(
        json.dumps(
            {"p2_generation": "p2", "chunks_sha256": hashlib.sha256(content.encode()).hexdigest()}
        )
    )
    assert source_lookup(tmp_path, generation, "c1")["source_verified"]
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        source_lookup(tmp_path, generation, "c1")


def test_clipped_neighbor_can_be_cited_without_inventing_text(monkeypatch, tmp_path):
    parent = {
        "chunk_id": "c0",
        "text": "context",
        "product": "UPF",
        "version": "5.0",
        "section_path": ["test"],
        "neighbors": [{"chunk_id": "c1", "text": "show cp-node"}],
    }
    neighbor = {
        "chunk_id": "c1",
        "text": "show cp-node\nmore text",
        "product": "UPF",
        "version": "5.0",
        "section_path": ["test"],
    }
    monkeypatch.setattr(
        "upf_agent.p3_answer.source_lookup",
        lambda root, gen, cid: neighbor if cid == "c1" else parent,
    )
    result = answer_from_retrieval(
        tmp_path,
        "query",
        retrieval([parent]),
        FakeClient([draft().model_dump(), {"supported": [True]}]),
    )
    assert result["status"] == "completed"
    assert result["citations"][0]["chunk_id"] == "c1"
    parent["neighbors"][0]["text"] = "invented"
    with pytest.raises(ValueError, match="neighbor"):
        answer_from_retrieval(tmp_path, "query", retrieval([parent]), FakeClient([]))


def test_quote_handles_preserve_raw_text_and_reject_invented_ids():
    from upf_agent.p3_answer import SelectedAnswer, quote_catalog, resolve_selection

    evidence = [
        {
            "chunk_id": "c1",
            "text": "路径 A\n步骤 B",
            "product": "UPF",
            "version": "5.0",
            "section_path": [],
        }
    ]
    context, catalog = quote_catalog(evidence)
    assert context[0]["quotes"][0]["text"] == "路径 A\n"
    selected = SelectedAnswer(
        status="completed",
        claims=[{"text": "路径 A", "quote_ids": ["S1Q1"]}],
        missing_information=[],
    )
    resolved = resolve_selection(selected, catalog)
    assert resolved.claims[0].citations[0].quote == "路径 A\n"
    validate_quotes(resolved, evidence)
    selected.claims[0].quote_ids = ["invented"]
    with pytest.raises(ValueError, match="unknown quote_id"):
        resolve_selection(selected, catalog)


def test_irrelevant_nonempty_evidence_refuses_before_generation(monkeypatch, tmp_path):
    evidence = {
        "chunk_id": "c1",
        "text": "show cp-node",
        "product": "UPF",
        "version": "5.0",
        "section_path": ["查询cp node信息"],
    }
    monkeypatch.setattr("upf_agent.p3_answer.source_lookup", lambda *args: evidence)
    client = FakeClient(
        [],
        gate={
            "has_relevant_evidence": False,
            "relevant_quote_ids": [],
            "missing_information": ["没有所请求功能的手册证据。"],
        },
    )
    result = answer_from_retrieval(tmp_path, "unrelated function", retrieval([evidence]), client)
    assert result["status"] == "missing_source"
    assert not result["claims"]
    assert client.calls == 1


def test_evidence_gate_cannot_invent_source_ids(monkeypatch, tmp_path):
    evidence = {
        "chunk_id": "c1",
        "text": "show cp-node",
        "product": "UPF",
        "version": "5.0",
        "section_path": ["查询cp node信息"],
    }
    monkeypatch.setattr("upf_agent.p3_answer.source_lookup", lambda *args: evidence)
    client = FakeClient(
        [],
        gate={
            "has_relevant_evidence": True,
            "relevant_quote_ids": ["invented"],
            "missing_information": [],
        },
    )
    result = answer_from_retrieval(tmp_path, "query", retrieval([evidence]), client)
    assert result["status"] == "failed"
    assert not result["claims"]
    assert client.calls == 1


def test_named_term_gate_handles_5gc_boundaries_and_absent_features():
    from upf_agent.p3_answer import unsupported_named_terms

    source = "5GC网管 V1.3 UPF N4 IPv6 show cp-node set-upf-prt-sw CPU"
    assert unsupported_named_terms("5GC网管 V1.3 实时告警在哪里查询？", source) == []
    assert unsupported_named_terms("UPF Wi-Fi 7 如何切换？", source) == ["Wi-Fi"]
    assert unsupported_named_terms("UPF N40 地址？", source) == ["N40"]
    assert unsupported_named_terms("HOW CAN I show cp-node?", source) == []
    assert unsupported_named_terms("set-upf-prt-sw 参数？", source) == []


def test_absent_named_feature_refuses_without_calling_model(monkeypatch, tmp_path):
    from upf_agent.p3_chunks import RetrievalChunk

    evidence = {
        "chunk_id": "c1",
        "text": "show cp-node",
        "product": "UPF",
        "version": "5.0",
        "section_path": ["查询cp node信息"],
    }
    monkeypatch.setattr("upf_agent.p3_answer.source_lookup", lambda *args: evidence)
    monkeypatch.setattr(
        "upf_agent.p3_answer.load_chunks",
        lambda *args: {
            "c1": RetrievalChunk(
                product="UPF",
                version="5.0",
                section_path=["查询cp node信息"],
                text="show cp-node",
                chunk_id="c1",
                source_id="s",
                source_sha256="a" * 64,
                relative_path="x.docx",
                parent_id="p",
                kind="command",
                token_count=3,
                block_ids=["b"],
                locators=["p[1]"],
                ordinal_start=1,
                ordinal_end=1,
            )
        },
    )
    client = FakeClient([])
    result = answer_from_retrieval(tmp_path, "UPF Wi-Fi 7 channel?", retrieval([evidence]), client)
    assert result["status"] == "missing_source"
    assert result["source_term_gate"]["unsupported_terms"] == ["Wi-Fi"]
    assert not result["claims"] and client.calls == 0


def test_publication_preserves_source_syntax_and_does_not_publish_inferred_prerequisite():
    from upf_agent.p3_answer import publish_source_extracts

    original = DraftAnswer(
        status="completed",
        missing_information=[],
        claims=[
            {
                "text": "下载前必须查询或刷新",
                "citations": [{"chunk_id": "c1", "quote": "点击Query Alarm-->点击Refresh"}],
            },
            {
                "text": "echo 'core_pattern=value'",
                "citations": [{"chunk_id": "c2", "quote": 'echo "core_pattern=value"'}],
            },
        ],
    )
    published = publish_source_extracts(original)
    assert published.claims[0].text == "点击Query Alarm-->点击Refresh"
    assert published.claims[1].text == 'echo "core_pattern=value"'
    assert "必须" not in published.claims[0].text
    assert original.claims[0].text == "下载前必须查询或刷新"


def test_incomplete_heredoc_cannot_be_published_as_complete_script():
    from upf_agent.p3_answer import publish_source_extracts

    original = DraftAnswer(
        status="completed",
        missing_information=[],
        claims=[
            {
                "text": "完整脚本",
                "citations": [{"chunk_id": "c", "quote": "cat >file <<EOF\nvalue\n"}],
            }
        ],
    )
    published = publish_source_extracts(original)
    assert published.status == "partial"
    assert published.missing_information
    assert original.status == "completed"
    original.claims[0].citations[0].quote += "EOF\n"
    assert publish_source_extracts(original).status == "completed"
