"""Resolve literal native chapter hints without using evaluation labels."""

from __future__ import annotations

import re
from dataclasses import dataclass

from upf_agent.p3_chunks import RetrievalChunk
from upf_agent.p3_lexical import scope_allows

QUOTED = re.compile(r"‘([^’]+)’|“([^”]+)”|\"([^\"]+)\"")
COMMAND_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_/-])[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)+(?![A-Za-z0-9_/-])"
)


@dataclass(frozen=True)
class SectionHint:
    source_id: str
    path: tuple[str, ...]
    method: str


def _normalized(value: str) -> str:
    return "".join(value.split()).casefold()


def _normalized_title(value: str) -> str:
    return _normalized(re.sub(r"^\s*\d+(?:\.\d+)*[.、\s]*", "", value))


def belongs(chunk: RetrievalChunk, hint: SectionHint) -> bool:
    return (
        chunk.source_id == hint.source_id
        and tuple(chunk.section_path[: len(hint.path)]) == hint.path
    )


def _choose(
    matches: set[tuple[str, tuple[str, ...]]],
    query: str,
    eligible: list[RetrievalChunk],
    method: str,
) -> SectionHint | None:
    if len(matches) == 1:
        source, path = next(iter(matches))
        return SectionHint(source, path, method)
    tokens = {match.group().casefold() for match in COMMAND_TOKEN.finditer(query)}
    if not tokens:
        return None
    supported = set()
    for source, path in matches:
        hint = SectionHint(source, path, method)
        native_tokens = {
            match.group().casefold()
            for chunk in eligible
            if chunk.kind == "command" and belongs(chunk, hint)
            for match in COMMAND_TOKEN.finditer(chunk.text)
        }
        if tokens & native_tokens:
            supported.add((source, path))
    if len(supported) == 1:
        source, path = next(iter(supported))
        return SectionHint(source, path, method + "_native_command")
    return None


def resolve_section(
    query: str, chunks: dict[str, RetrievalChunk], product: str | None, version: str | None
) -> SectionHint | None:
    eligible = [chunk for chunk in chunks.values() if scope_allows(chunk, product, version)]
    paths = {(chunk.source_id, tuple(chunk.section_path)) for chunk in eligible}
    explicit = set()
    explicit_groups = []
    for alternatives in QUOTED.findall(query):
        value = next(part for part in alternatives if part)
        segments = tuple(_normalized(part) for part in value.split("/"))
        if not all(segments):
            continue
        group = set()
        for source, path in paths:
            normalized = tuple(_normalized(part) for part in path)
            for end in range(len(segments), len(path) + 1):
                if normalized[end - len(segments) : end] == segments:
                    group.add((source, path[:end]))
        if group:
            explicit_groups.append(group)
            explicit |= group
    if len(explicit_groups) > 1 and any(
        group != explicit_groups[0] for group in explicit_groups[1:]
    ):
        return None
    if explicit:
        return _choose(explicit, query, eligible, "explicit_section")
    normalized_query = _normalized(query)
    matches = {
        (source, path[:end])
        for source, path in paths
        for end, title in enumerate(path, 1)
        if len(_normalized_title(title)) >= 4 and _normalized_title(title) in normalized_query
    }
    deepest = {
        (source, path)
        for source, path in matches
        if not any(
            other_source == source and len(other) > len(path) and other[: len(path)] == path
            for other_source, other in matches
        )
    }
    return _choose(deepest, query, eligible, "unique_section_title") if deepest else None


def section_routes(
    routes: dict[str, list[tuple[str, float]]],
    chunks: dict[str, RetrievalChunk],
    hint: SectionHint | None,
    limit: int,
    product: str | None = None,
    version: str | None = None,
) -> dict[str, list[tuple[str, float]]]:
    if hint is None:
        return routes
    section = sorted(
        (
            chunk
            for chunk in chunks.values()
            if belongs(chunk, hint) and scope_allows(chunk, product, version)
        ),
        key=lambda chunk: (chunk.ordinal_start, chunk.chunk_id),
    )
    # A large chapter cannot be replaced by an arbitrary truncated prefix.
    if not section or len(section) > limit:
        return routes
    allowed = {chunk.chunk_id for chunk in section}
    return {
        **{
            name: [(key, score) for key, score in hits if key in allowed]
            for name, hits in routes.items()
        },
        "section": [(chunk.chunk_id, 1.0) for chunk in section],
    }


def related_section_routes(
    query: str,
    routes: dict[str, list[tuple[str, float]]],
    chunks: dict[str, RetrievalChunk],
    product: str | None,
    version: str | None,
    limit: int,
) -> tuple[dict[str, list[tuple[str, float]]], list[SectionHint]]:
    """Supplement a small set of same-name chapters without selecting one as truth."""
    eligible = [chunk for chunk in chunks.values() if scope_allows(chunk, product, version)]
    normalized_query = _normalized(query)
    matches = {
        (chunk.source_id, tuple(chunk.section_path[:end]))
        for chunk in eligible
        for end, title in enumerate(chunk.section_path, 1)
        if len(_normalized_title(title)) >= 4 and _normalized_title(title) in normalized_query
    }
    deepest = {
        (source, path)
        for source, path in matches
        if not any(
            other_source == source and len(other) > len(path) and other[: len(path)] == path
            for other_source, other in matches
        )
    }
    if len(deepest) < 2 or len({_normalized_title(path[-1]) for _, path in deepest}) != 1:
        return routes, []
    hints = [
        SectionHint(source, path, "ambiguous_native_title_candidates")
        for source, path in sorted(deepest)
    ]
    candidates = sorted(
        (chunk for chunk in eligible if any(belongs(chunk, hint) for hint in hints)),
        key=lambda chunk: (chunk.source_id, chunk.ordinal_start, chunk.chunk_id),
    )
    if not candidates or len(candidates) > limit:
        return routes, []
    return {
        **routes,
        "section": [(chunk.chunk_id, 1.0) for chunk in candidates],
    }, hints
