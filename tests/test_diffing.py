from __future__ import annotations

import pytest

from alphaevolve.diffing import apply_diff, parse_diff
from alphaevolve.errors import DiffApplyError, DiffParseError


def test_exact_match_diff_application() -> None:
    source = "def score():\n    return 1\n"
    diff = parse_diff(
        """<<<<<<< SEARCH
    return 1
=======
    return 2
>>>>>>> REPLACE"""
    )
    updated = apply_diff(source, diff)
    assert "return 2" in updated


def test_line_ending_normalization_match() -> None:
    source = "def score():\r\n    return 1\r\n"
    diff = parse_diff(
        """<<<<<<< SEARCH
def score():
    return 1
=======
def score():
    return 3
>>>>>>> REPLACE"""
    )
    updated = apply_diff(source, diff)
    assert "return 3" in updated


def test_indentation_insensitive_match_reindents_replacement() -> None:
    source = "def outer():\n        value = 1\n        return value\n"
    diff = parse_diff(
        """<<<<<<< SEARCH
    value = 1
    return value
=======
    value = 2
    return value
>>>>>>> REPLACE"""
    )
    updated = apply_diff(source, diff)
    assert "        value = 2" in updated


def test_ambiguous_fuzzy_match_is_rejected() -> None:
    source = "if True:\n    value = 1\n    return value\n\nif False:\n    value = 1\n    return value\n"
    diff = parse_diff(
        """<<<<<<< SEARCH
  value = 1
  return value
=======
  value = 2
  return value
>>>>>>> REPLACE"""
    )
    with pytest.raises(DiffApplyError):
        apply_diff(source, diff)


def test_prose_wrapped_diff_is_rejected() -> None:
    with pytest.raises(DiffParseError):
        parse_diff(
            """Here is the requested diff:
<<<<<<< SEARCH
    return 1
=======
    return 2
>>>>>>> REPLACE"""
        )


def test_malformed_marker_lines_are_normalized() -> None:
    source = "def score():\n    return 1\n"
    diff = parse_diff(
        """<<<<<<<<< SEARCH
    return 1
=========
    return 2
>>>>>>>>>> REPLACE"""
    )

    updated = apply_diff(source, diff)

    assert "return 2" in updated


def test_outer_markdown_fence_is_ignored_for_diff_only_payloads() -> None:
    source = "def score():\n    return 1\n"
    diff = parse_diff(
        """```python
<<<<<<< SEARCH
    return 1
=======
    return 4
>>>>>>> REPLACE
```"""
    )

    updated = apply_diff(source, diff)

    assert "return 4" in updated
