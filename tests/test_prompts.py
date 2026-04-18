from __future__ import annotations

import pytest

from alphaevolve.errors import PromptTooLargeError
from alphaevolve.models import Program, PromptArtifactContext, PromptBudget
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


def test_history_order_preserves_curated_diverse_selection() -> None:
    builder = PromptBuilder(PromptBudget(max_prompt_tokens=600, reserved_completion_tokens=50, max_history_programs=3))
    current = Program(id="current", code="x = 1\n", primary_score=1.0, archive_cell=(1, 1))
    best = Program(id="best", code="best = 1\n", primary_score=10.0, archive_cell=(1, 1))
    diverse = Program(id="diverse", code="diverse = 1\n", primary_score=7.0, archive_cell=(4, 1))
    recent_novel = Program(id="recent", code="recent = 1\n", primary_score=6.0, archive_cell=(2, 3))

    rendered = builder.build(
        system_instructions="Improve the code.",
        task_contract="Maximize score.",
        current_program=current,
        history=[best, diverse, recent_novel],
    )

    best_index = rendered.text.index("Program ID: best")
    diverse_index = rendered.text.index("Program ID: diverse")
    recent_index = rendered.text.index("Program ID: recent")
    assert best_index < diverse_index < recent_index


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


def test_prompt_includes_feedback_and_artifact_summaries() -> None:
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
        artifact_context=(
            PromptArtifactContext(name="report", type="json", summary="score=10"),
        ),
        evaluator_feedback="Try improving the ratio heuristic.",
    )

    assert "## Authorized Edit Window" in rendered.text
    assert "History programs are reference-only" in rendered.text
    assert "Your response must start with <<<<<<< SEARCH and end with >>>>>>> REPLACE." in rendered.text
    assert "Prefer one semantically complete edit unit." in rendered.text
    assert "The SEARCH block must include a semantically complete logical chain needed to make the change valid." in rendered.text
    assert "Never patch a single conditional branch, a single line in a repeated chain, or a partial loop body." in rendered.text
    assert "It is valid to replace a full helper function, loop, or repeated conditional chain when the change spans that unit." in rendered.text
    assert "If a smaller chain would make the edit incomplete, replace the entire helper function." in rendered.text
    assert "Emit one SEARCH/REPLACE block per logical edit unit; only emit multiple blocks for disjoint units." in rendered.text
    assert "## Artifact Summaries" in rendered.text
    assert "## Evaluator Feedback" in rendered.text


def test_full_file_prompt_omits_authorized_edit_window() -> None:
    builder = PromptBuilder(
        PromptBudget(max_prompt_tokens=600, reserved_completion_tokens=50, max_history_programs=1),
        mutation_scope="full_file",
    )
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

    assert "## Authorized Edit Window" not in rendered.text
    assert "Only modify code between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END." not in rendered.text
    assert "You may modify any part of the current program file." in rendered.text
    assert "It is valid to replace imports, helper functions, and the overall algorithm structure when needed." in rendered.text
