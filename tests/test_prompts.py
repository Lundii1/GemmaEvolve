from __future__ import annotations

import pytest

from alphaevolve.errors import PromptTooLargeError
from alphaevolve.models import Program, PromptBudget
from alphaevolve.prompts import PromptBuilder


def test_current_program_is_never_truncated() -> None:
    program = Program(id="current", code="print('hello')", metrics={"score": 1.0}, primary_score=1.0)
    builder = PromptBuilder(PromptBudget(max_prompt_tokens=400, reserved_completion_tokens=50, max_history_programs=2))

    rendered = builder.build(
        system_instructions="Improve the code.",
        task_contract="Maximize score.",
        current_program=program,
        history=[],
    )

    assert "print('hello')" in rendered.text
    assert rendered.summarized_program_ids == ()


def test_history_uses_sliding_window_and_summary_card() -> None:
    builder = PromptBuilder(PromptBudget(max_prompt_tokens=400, reserved_completion_tokens=50, max_history_programs=3))
    current = Program(id="current", code="x = 1\n" * 20, metrics={"score": 1.0}, primary_score=1.0, archive_cell=(0, 0))
    history = [
        Program(id="h1", code="a = 1\n" * 20, metrics={"score": 10.0}, primary_score=10.0, archive_cell=(1, 0)),
        Program(id="h2", code="b = 1\n" * 45, metrics={"score": 9.0}, primary_score=9.0, archive_cell=(2, 0)),
        Program(id="h3", code="c = 1\n" * 60, metrics={"score": 8.0}, primary_score=8.0, archive_cell=(3, 0)),
    ]

    rendered = builder.build(
        system_instructions="Improve the code.",
        task_contract="Maximize score.",
        current_program=current,
        history=history,
    )

    assert "History Program (Summary Only)" in rendered.text
    assert len(rendered.summarized_program_ids) == 1


def test_prompt_raises_when_current_program_alone_exceeds_budget() -> None:
    builder = PromptBuilder(PromptBudget(max_prompt_tokens=100, reserved_completion_tokens=10, max_history_programs=1))
    current = Program(id="current", code="x = 1\n" * 200)

    with pytest.raises(PromptTooLargeError):
        builder.build(
            system_instructions="Improve the code.",
            task_contract="Maximize score.",
            current_program=current,
            history=[],
        )


def test_prompt_includes_authorized_edit_window_instructions_when_markers_present() -> None:
    builder = PromptBuilder(PromptBudget(max_prompt_tokens=600, reserved_completion_tokens=50, max_history_programs=1))
    current = Program(
        id="current",
        code=(
            "# EVOLVE-BLOCK-START\n"
            "def heuristic_score(weight: float, value: float) -> float:\n"
            "    return value\n"
            "# EVOLVE-BLOCK-END\n"
            "print('done')\n"
        ),
    )

    rendered = builder.build(
        system_instructions="Improve the code.",
        task_contract="Maximize score.",
        current_program=current,
        history=[],
    )

    assert "## Authorized Edit Window" in rendered.text
    assert "History programs are reference-only" in rendered.text
    assert "Only modify code between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END." in rendered.text
