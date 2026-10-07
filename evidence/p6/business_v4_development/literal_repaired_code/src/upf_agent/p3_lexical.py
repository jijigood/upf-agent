"""Chinese technical BM25 and literal command lookup with one shared scope rule."""

from __future__ import annotations

import json
import re
from pathlib import Path

import bm25s
import jieba

from upf_agent.contracts import CommandRecord
from upf_agent.p3_chunks import RetrievalChunk

TECH_TOKEN = re.compile(
    r"(?i)(?<![\w])(?:0x[0-9a-f]+|(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?|"
    r"[a-z][a-z0-9]*(?:[-_/][a-z0-9]+)+|[a-z][a-z0-9]*)(?![\w])"
)
COMMAND_PHRASE = re.compile(r"(?i)\b(?:show|set|display|upfmgr|vppctl)\s+[a-z0-9_-]+")


def scope_allows(chunk: RetrievalChunk | dict, product: str | None, version: str | None) -> bool:
    candidate_product = chunk.product if isinstance(chunk, RetrievalChunk) else chunk.get("product")
    candidate_version = chunk.version if isinstance(chunk, RetrievalChunk) else chunk.get("version")
    if product and candidate_product != product:
        return False
    if version and candidate_version not in {version, "unknown", None}:
        return False
    return True


class TechnicalTokenizer:
    def __init__(self, dictionary: Path) -> None:
        self.jieba = jieba.Tokenizer()
        self.jieba.load_userdict(str(dictionary))

    def tokens(self, text: str) -> list[str]:
        words = [token.strip().casefold() for token in self.jieba.lcut(text, HMM=False)]
        words.extend(match.group().casefold() for match in TECH_TOKEN.finditer(text))
        words.extend(
            re.sub(r"\s+", " ", match.group().casefold()) for match in COMMAND_PHRASE.finditer(text)
        )
        return [
            word
            for word in words
            if word and not all(char in "，。；：、/\t\n .,:;()（）【】[]<>|" for char in word)
        ]


def build_bm25(chunks: list[RetrievalChunk], dictionary: Path, output: Path) -> dict:
    tokenizer = TechnicalTokenizer(dictionary)
    corpus = [{"chunk_id": chunk.chunk_id} for chunk in chunks]
    tokens = [tokenizer.tokens(chunk.text) for chunk in chunks]
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(tokens, show_progress=False)
    output.mkdir(parents=True, exist_ok=True)
    retriever.save(str(output), corpus=corpus, show_progress=False)
    return {"documents": len(chunks), "vocabulary": len(retriever.vocab_dict)}


def search_bm25(
    index_dir: Path,
    chunks: dict[str, RetrievalChunk],
    tokenizer: TechnicalTokenizer,
    query: str,
    product: str | None,
    version: str | None,
    limit: int = 20,
) -> list[tuple[str, float]]:
    if not chunks:
        return []
    retriever = bm25s.BM25.load(str(index_dir), load_corpus=True)
    tokens = tokenizer.tokens(query)
    if not tokens:
        return []
    result = retriever.retrieve([tokens], k=len(chunks), show_progress=False)
    hits = []
    for item, score in zip(result.documents[0], result.scores[0], strict=True):
        chunk_id = item["chunk_id"]
        if score > 0 and scope_allows(chunks[chunk_id], product, version):
            hits.append((chunk_id, float(score)))
            if len(hits) >= limit:
                break
    return hits


def _command_alias(syntax: str) -> str:
    raw = re.sub(r"^(?:\[[^\]]+\][#$]|[\w-]{2,30}[>#])\s*", "", syntax.strip())
    raw = re.split(r"[<{（]", raw, maxsplit=1)[0].strip()
    raw = re.sub(r"\s+", " ", raw)
    tokens = raw.split()
    if len(tokens) > 5:
        tokens = tokens[:5]
    return " ".join(tokens).casefold()


def build_exact(commands: list[CommandRecord], chunks: list[RetrievalChunk], output: Path) -> dict:
    chunk_by_command = {chunk.command_id: chunk.chunk_id for chunk in chunks if chunk.command_id}
    entries = []
    for command in commands:
        chunk_id = chunk_by_command.get(command.command_id)
        alias = _command_alias(command.syntax)
        if not chunk_id or not alias:
            continue
        entries.append(
            {
                "alias": alias,
                "chunk_id": chunk_id,
                "command_id": command.command_id,
                "source_id": command.source_id,
                "product": command.product,
                "version": command.version,
                "family": command.family,
            }
        )
    entries.sort(key=lambda entry: (-len(entry["alias"]), entry["command_id"]))
    output.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"commands": len(entries), "distinct_aliases": len({x["alias"] for x in entries})}


def search_exact(
    path: Path, query: str, product: str | None, version: str | None, limit: int = 20
) -> list[tuple[str, float]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    normalized = re.sub(r"\s+", " ", query.casefold())
    seen, hits = set(), []
    for entry in entries:
        alias = entry["alias"]
        if len(alias) < 5 or not scope_allows(entry, product, version):
            continue
        pattern = rf"(?<![a-z0-9_-]){re.escape(alias)}(?![a-z0-9_-])"
        if re.search(pattern, normalized) and entry["chunk_id"] not in seen:
            seen.add(entry["chunk_id"])
            hits.append((entry["chunk_id"], float(len(alias))))
            if len(hits) >= limit:
                break
    return hits
