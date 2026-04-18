"""SEARCH/REPLACE diff parsing and application."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from alphaevolve.errors import DiffApplyError, DiffParseError
from alphaevolve.models import Diff, DiffBlock

_DIFF_PATTERN = re.compile(
    r"<<<<<<< SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n>>>>>>> REPLACE",
    re.DOTALL,
)
_OUTER_FENCE_PATTERN = re.compile(r"^```[^\n`]*\r?\n(?P<body>.*)\r?\n```$", re.DOTALL)
_SEARCH_MARKER_PATTERN = re.compile(r"^<+\s*SEARCH\s*$")
_DIVIDER_MARKER_PATTERN = re.compile(r"^={3,}\s*$")
_REPLACE_MARKER_PATTERN = re.compile(r"^>+\s*REPLACE\s*$")
_ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\u200b\u200c\u200d\u2060\ufeff")


@dataclass(frozen=True, slots=True)
class _Match:
    start: int
    end: int
    strategy: str
    base_indent: str = ""


def parse_diff(raw_text: str) -> Diff:
    """Extract SEARCH/REPLACE blocks from a raw model response."""
    normalized_text = _normalize_diff_text(raw_text)
    matches = list(_DIFF_PATTERN.finditer(normalized_text))
    if not matches:
        raise DiffParseError("No SEARCH/REPLACE blocks were found in the model response.")
    cursor = 0
    for match in matches:
        if normalized_text[cursor : match.start()].strip():
            raise DiffParseError("Model response contained text outside SEARCH/REPLACE blocks.")
        cursor = match.end()
    if normalized_text[cursor:].strip():
        raise DiffParseError("Model response contained text outside SEARCH/REPLACE blocks.")
    blocks = tuple(
        DiffBlock(search=match.group(1), replace=match.group(2))
        for match in matches
    )
    return Diff(raw_text=normalized_text, blocks=blocks)


def apply_diff(source: str, diff: Diff) -> str:
    """Apply diff blocks sequentially using tolerant but safe matching."""
    updated = source
    for block in diff.blocks:
        updated = _apply_block(updated, block)
    return updated


def _apply_block(source: str, block: DiffBlock) -> str:
    match = (
        _find_exact_match(source, block.search)
        or _find_normalized_line_ending_match(source, block.search)
        or _find_indentation_insensitive_match(source, block.search)
    )
    if match is None:
        raise DiffApplyError("SEARCH block could not be matched uniquely in the current program.")

    newline_style = _dominant_newline(source)
    if match.strategy == "indent":
        replacement = _reindent_replacement(block.replace, match.base_indent, newline_style)
    else:
        replacement = _convert_newlines(block.replace, newline_style)
    return f"{source[:match.start]}{replacement}{source[match.end:]}"


def _find_exact_match(source: str, search: str) -> _Match | None:
    starts = _all_occurrences(source, search)
    if len(starts) > 1:
        raise DiffApplyError("SEARCH block matched multiple exact regions.")
    if not starts:
        return None
    start = starts[0]
    return _Match(start=start, end=start + len(search), strategy="exact")


def _find_normalized_line_ending_match(source: str, search: str) -> _Match | None:
    normalized_source, source_map = _normalize_line_endings_with_map(source)
    normalized_search, _ = _normalize_line_endings_with_map(search)
    starts = _all_occurrences(normalized_source, normalized_search)
    if len(starts) > 1:
        raise DiffApplyError("SEARCH block matched multiple regions after line-ending normalization.")
    if not starts:
        return None
    start = starts[0]
    end = start + len(normalized_search)
    return _Match(
        start=source_map[start],
        end=source_map[end],
        strategy="line_endings",
    )


def _find_indentation_insensitive_match(source: str, search: str) -> _Match | None:
    search_lines = search.splitlines()
    if not search_lines:
        raise DiffApplyError("SEARCH block cannot be empty.")

    source_lines = list(_split_lines_with_spans(source))
    window_size = len(search_lines)
    if len(source_lines) < window_size:
        return None

    canonical_search = _canonicalize_block(search_lines)
    matches: list[_Match] = []
    for start_index in range(len(source_lines) - window_size + 1):
        window = source_lines[start_index : start_index + window_size]
        candidate_lines = [line for line, _, _ in window]
        if _canonicalize_block(candidate_lines) != canonical_search:
            continue
        matches.append(
            _Match(
                start=window[0][1],
                end=window[-1][2],
                strategy="indent",
                base_indent=_common_indent_prefix(candidate_lines),
            )
        )

    if len(matches) > 1:
        raise DiffApplyError("SEARCH block matched multiple regions after indentation normalization.")
    return matches[0] if matches else None


def _all_occurrences(haystack: str, needle: str) -> list[int]:
    if not needle:
        raise DiffApplyError("SEARCH block cannot be empty.")
    starts: list[int] = []
    start = haystack.find(needle)
    while start != -1:
        starts.append(start)
        start = haystack.find(needle, start + 1)
    return starts


def _normalize_line_endings_with_map(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    index_map: list[int] = []
    index = 0
    while index < len(text):
        index_map.append(index)
        char = text[index]
        if char == "\r":
            if index + 1 < len(text) and text[index + 1] == "\n":
                normalized_chars.append("\n")
                index += 2
                continue
            normalized_chars.append("\n")
            index += 1
            continue
        normalized_chars.append(char)
        index += 1
    index_map.append(len(text))
    return "".join(normalized_chars), index_map


def _normalize_diff_text(raw_text: str) -> str:
    text = raw_text.strip()
    fenced_body = _strip_outer_markdown_fence(text)
    if fenced_body is not None:
        text = fenced_body
    lines = text.splitlines()
    normalized_lines = [_normalize_marker_line(line) for line in lines]
    return "\n".join(normalized_lines)


def _strip_outer_markdown_fence(text: str) -> str | None:
    match = _OUTER_FENCE_PATTERN.match(text)
    if match is None:
        return None
    body = match.group("body").strip()
    if _contains_diff_markers(body):
        return body
    return None


def _contains_diff_markers(text: str) -> bool:
    saw_search = False
    saw_replace = False
    for line in text.splitlines():
        normalized = _normalize_marker_line(line)
        if normalized == "<<<<<<< SEARCH":
            saw_search = True
        elif normalized == ">>>>>>> REPLACE":
            saw_replace = True
    return saw_search and saw_replace


def _normalize_marker_line(line: str) -> str:
    candidate = line.translate(_ZERO_WIDTH_TRANSLATION).strip()
    if _SEARCH_MARKER_PATTERN.fullmatch(candidate):
        return "<<<<<<< SEARCH"
    if _DIVIDER_MARKER_PATTERN.fullmatch(candidate):
        return "======="
    if _REPLACE_MARKER_PATTERN.fullmatch(candidate):
        return ">>>>>>> REPLACE"
    return line


def _split_lines_with_spans(text: str) -> Iterable[tuple[str, int, int]]:
    offset = 0
    for chunk in text.splitlines(keepends=True):
        line = chunk.rstrip("\r\n")
        start = offset
        offset += len(chunk)
        yield line, start, offset


def _canonicalize_block(lines: list[str]) -> list[str]:
    indent = _common_indent_prefix(lines)
    canonical: list[str] = []
    for line in lines:
        if line.strip():
            canonical.append(line[len(indent) :].rstrip())
        else:
            canonical.append("")
    return canonical


def _common_indent_prefix(lines: list[str]) -> str:
    indents = [re.match(r"[ \t]*", line).group(0) for line in lines if line.strip()]
    if not indents:
        return ""
    prefix = indents[0]
    for indent in indents[1:]:
        while prefix and not indent.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break
    return prefix


def _reindent_replacement(replacement: str, base_indent: str, newline_style: str) -> str:
    raw_lines = replacement.splitlines()
    if not raw_lines:
        return replacement
    replace_indent = _common_indent_prefix(raw_lines)
    adjusted: list[str] = []
    for line in raw_lines:
        if line.strip():
            adjusted.append(f"{base_indent}{line[len(replace_indent):]}")
        else:
            adjusted.append("")
    return newline_style.join(adjusted)


def _dominant_newline(text: str) -> str:
    crlf_count = text.count("\r\n")
    lf_count = text.count("\n")
    if crlf_count and crlf_count >= lf_count / 2:
        return "\r\n"
    return "\n"


def _convert_newlines(text: str, newline_style: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", newline_style)
