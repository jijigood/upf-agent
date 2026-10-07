from pathlib import Path
from tempfile import TemporaryDirectory

from qdrant_client import QdrantClient, models
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers import models as token_models

from upf_agent.p3_chunks import RetrievalChunk, _redact, build_chunks
from upf_agent.p3_index import dense_search
from upf_agent.p3_lexical import TechnicalTokenizer, build_bm25, scope_allows, search_bm25
from upf_agent.p3_search import _candidate_budget, _inferred_scope, expand_context, fuse_rrf


def _chunk(
    chunk_id: str, text: str, *, version: str = "5.0", parent: str = "section-a", ordinal: int = 1
) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        source_id="manual-1",
        source_sha256="a" * 64,
        relative_path="manual.docx",
        product="UPF",
        version=version,
        section_path=["N4配置"],
        parent_id=parent,
        kind="prose",
        text=text,
        token_count=5,
        block_ids=[f"block-{chunk_id}"],
        locators=[f"p[{ordinal}]"],
        ordinal_start=ordinal,
        ordinal_end=ordinal,
    )


def _tiny_tokenizer(tmp_path: Path) -> Tokenizer:
    tokenizer = Tokenizer(
        token_models.WordLevel(
            vocab={"[UNK]": 0, "N4": 1, "接口": 2, "配置": 3, "命令": 4}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def test_rrf_deduplicates_and_keeps_exact_in_candidate_budget():
    routes = {
        "exact": [("a", 9.0)],
        "bm25": [("b", 4.0), ("a", 3.0)],
        "dense": [("b", 0.9), ("c", 0.8)],
    }
    fused = fuse_rrf(routes, k=60)
    assert len(fused) == 3
    a = next(item for item in fused if item["chunk_id"] == "a")
    assert set(a["routes"]) == {"exact", "bm25"}
    assert _candidate_budget(fused, 1)[0]["chunk_id"] == "a"


def test_scope_filters_known_wrong_version_and_query_ip_is_not_version():
    assert scope_allows(_chunk("a", "N4", version="unknown"), "UPF", "5.0")
    assert not scope_allows(_chunk("a", "N4", version="4.0"), "UPF", "5.0")
    assert not scope_allows(_chunk("a", "N4"), "5GC网管", None)
    assert _inferred_scope("N4 地址 10.5.0.1", None, None) == ("UPF", None)


def test_bm25_preserves_technical_identifiers_and_scope(tmp_path: Path):
    dictionary = tmp_path / "terms.txt"
    dictionary.write_text("N4 100000\nPFCP 100000\n")
    tokenizer = TechnicalTokenizer(dictionary)
    tokens = tokenizer.tokens("PFCP N4 show cp-node 10.1.1.2")
    assert {"pfcp", "n4", "cp-node", "10.1.1.2", "show cp-node"} <= set(tokens)
    chunks = [
        _chunk("a", "PFCP N4 链路", version="5.0"),
        _chunk("b", "PFCP N4 链路", version="4.0"),
    ]
    build_bm25(chunks, dictionary, tmp_path / "bm25")
    hits = search_bm25(
        tmp_path / "bm25", {c.chunk_id: c for c in chunks}, tokenizer, "PFCP N4", "UPF", "5.0", 10
    )
    assert [chunk_id for chunk_id, _ in hits] == ["a"]


def test_qdrant_local_dimension_and_payload_filter():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        client = QdrantClient(path=str(root / "qdrant"))
        client.create_collection(
            "small", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
        client.upsert(
            "small",
            [
                models.PointStruct(
                    id=1,
                    vector=[1.0, 0.0],
                    payload={"chunk_id": "a", "role": "manual", "product": "UPF", "version": "5.0"},
                ),
                models.PointStruct(
                    id=2,
                    vector=[0.9, 0.1],
                    payload={"chunk_id": "b", "role": "manual", "product": "UPF", "version": "4.0"},
                ),
            ],
        )
        assert client.get_collection("small").config.params.vectors.size == 2
        client.close()
        assert dense_search(root, "small", [1.0, 0.0], "UPF", "5.0", 10) == [("a", 1.0)]


def test_parent_expansion_is_bounded_and_keeps_locators(tmp_path: Path):
    tokenizer = _tiny_tokenizer(tmp_path)
    chunks = {
        "a": _chunk("a", "N4 接口 配置", ordinal=1),
        "b": _chunk("b", "N4 配置 命令", ordinal=2),
    }
    evidence, tokens = expand_context(
        [{"chunk_id": "a", "rrf_score": 0.1}], chunks, tokenizer, total_budget=10, neighbor_budget=4
    )
    assert tokens <= 10
    assert evidence[0]["locators"] == ["p[1]"]
    assert evidence[0]["neighbors"][0]["locators"] == ["p[2]"]


def test_jwt_is_redacted_from_indexable_text():
    token = "eyJ" + "a" * 35 + "." + "b" * 30 + "." + "c" * 20
    redacted, flags = _redact("token=" + token)
    assert token not in redacted and "[REDACTED_JWT]" in redacted
    assert flags == ["sensitive_token_redacted"]


def test_command_example_keeps_its_own_source_locator(tmp_path, monkeypatch):
    import upf_agent.p3_chunks as chunking
    from upf_agent.contracts import CommandRecord, DocumentBlock

    source = {
        "sha256": "a" * 64,
        "relative_path": "manual.docx",
        "product": "UPF",
        "version": "5.0",
    }

    def block(identifier, text, paragraph):
        return DocumentBlock(
            source_id="manual-1",
            block_id=identifier,
            kind="paragraph",
            section_path=["功能配置"],
            raw_text=text,
            normalized_text=text,
            locator=f"word/document.xml/body/p[{paragraph}]",
            extraction_method="native_ooxml",
            ordinal=paragraph,
        )

    syntax = "set-upf-prt-sw (n4-prt-sw | gtpu-prt-sw) <level=0:4>"
    example = "UPFD> set-upf-prt-sw n4-prt-sw 4"
    blocks = [block("syntax", syntax, 177), block("example", example, 183)]
    command = CommandRecord(
        command_id="cmd-1",
        source_id="manual-1",
        family="CLI",
        syntax=syntax,
        syntax_block_id="syntax",
        example_block_ids=["example"],
    )
    monkeypatch.setattr(
        chunking, "_load_p2", lambda root: ("a" * 20, blocks, [command], {"manual-1": source})
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    _tiny_tokenizer(tmp_path).save(str(tokenizer_path))

    _, chunks = build_chunks(tmp_path, tokenizer_path)

    syntax_chunk = next(c for c in chunks if c.command_id == "cmd-1")
    assert syntax_chunk.text == f"功能配置\n{syntax}"
    assert syntax_chunk.block_ids == ["syntax"]
    assert syntax_chunk.locators == ["word/document.xml/body/p[177]"]
    example_chunk = next(c for c in chunks if example in c.text)
    assert example_chunk.locators == ["word/document.xml/body/p[183]"]
    assert all("相关说明：" not in c.text for c in chunks)
