"""Prompt construction with a bounded history window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from alphaevolve.errors import PromptTooLargeError
from alphaevolve.models import Program, PromptArtifactContext, PromptBudget
from alphaevolve.token_estimator import CharacterTokenEstimator, TokenEstimator

EVOLVE_BLOCK_START_MARKER = "# EVOLVE-BLOCK-START"
EVOLVE_BLOCK_END_MARKER = "# EVOLVE-BLOCK-END"


def _head_tail_excerpt(code: str, head_lines: int = 8, tail_lines: int = 8) -> str:
    lines = code.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return code
    head = "\n".join(lines[:head_lines])
    tail = "\n".join(lines[-tail_lines:])
    return f"{head}\n...\n{tail}"


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """A fully rendered prompt plus prompt-budget metadata."""

    text: str
    estimated_tokens: int
    included_program_ids: tuple[str, ...]
    summarized_program_ids: tuple[str, ...]
    artifact_context: tuple[PromptArtifactContext, ...]
    evaluator_feedback_used: str | None


@dataclass(frozen=True, slots=True)
class EditWindow:
    """Absolute character offsets for the editable code region."""

    start: int
    end: int
    content: str


def locate_edit_window(code: str) -> EditWindow | None:
    """Return the code between EVOLVE-BLOCK markers when present."""
    start_marker_index = code.find(EVOLVE_BLOCK_START_MARKER)
    end_marker_index = code.find(EVOLVE_BLOCK_END_MARKER)
    if start_marker_index == -1 and end_marker_index == -1:
        return None
    if start_marker_index == -1 or end_marker_index == -1 or start_marker_index >= end_marker_index:
        return None

    start_line_end = code.find("\n", start_marker_index)
    if start_line_end == -1:
        return None

    editable_start = start_line_end + 1
    editable_end = end_marker_index
    return EditWindow(
        start=editable_start,
        end=editable_end,
        content=code[editable_start:editable_end],
    )


class PromptBuilder:
    """Construct prompts that stay within a configured prompt budget."""

    def __init__(
        self,
        budget: PromptBudget,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        self._budget = budget
        self._token_estimator = token_estimator or CharacterTokenEstimator()

    def build(
        self,
        *,
        system_instructions: str,
        task_contract: str,
        current_program: Program,
        history: Sequence[Program],
        artifact_context: Sequence[PromptArtifactContext] = (),
        evaluator_feedback: str | None = None,
    ) -> RenderedPrompt:
        edit_window = locate_edit_window(current_program.code)
        system_section = self._render_system_section(
            system_instructions,
            has_edit_window=edit_window is not None,
        )
        task_section = self._render_task_section(task_contract)
        boundary_section = self._render_edit_window(edit_window)
        current_section = self._render_current_program(current_program)
        feedback_section = self._render_feedback(evaluator_feedback)
        artifact_section, used_artifacts = self._render_artifacts(artifact_context)
        base_sections = [system_section, task_section]
        if feedback_section is not None:
            base_sections.append(feedback_section)
        if artifact_section is not None:
            base_sections.append(artifact_section)
        if boundary_section is not None:
            base_sections.append(boundary_section)
        base_sections.append(current_section)
        base_prompt = "\n\n".join(base_sections)
        base_tokens = self._token_estimator.estimate(base_prompt)
        if base_tokens > self._budget.usable_prompt_tokens:
            raise PromptTooLargeError(
                "Current program is too large for configured model budget."
            )

        remaining_budget = self._budget.usable_prompt_tokens - base_tokens
        history_sections: list[str] = []
        included: list[str] = []
        summarized: list[str] = []

        ranked_history = self._rank_history(current_program, history)[: self._budget.max_history_programs]
        if ranked_history and remaining_budget > 0:
            history_header = "## Prior High-Performing Programs"
            header_tokens = self._token_estimator.estimate(history_header)
            if header_tokens <= remaining_budget:
                history_sections.append(history_header)
                remaining_budget -= header_tokens

        for program in ranked_history:
            full_card = self._render_history_program(program)
            full_tokens = self._token_estimator.estimate(full_card)
            if full_tokens <= remaining_budget:
                history_sections.append(full_card)
                included.append(program.id)
                remaining_budget -= full_tokens
                continue

            summary_card = self._render_summary_program(program)
            summary_tokens = self._token_estimator.estimate(summary_card)
            if summary_tokens <= remaining_budget:
                history_sections.append(summary_card)
                included.append(program.id)
                summarized.append(program.id)
            break

        prompt = "\n\n".join(base_sections + history_sections)
        estimated_tokens = self._token_estimator.estimate(prompt)
        if estimated_tokens > self._budget.usable_prompt_tokens:
            raise PromptTooLargeError("Prompt exceeded usable token budget after rendering.")
        return RenderedPrompt(
            text=prompt,
            estimated_tokens=estimated_tokens,
            included_program_ids=tuple(included),
            summarized_program_ids=tuple(summarized),
            artifact_context=tuple(used_artifacts),
            evaluator_feedback_used=evaluator_feedback,
        )

    def _render_system_section(self, system_instructions: str, *, has_edit_window: bool) -> str:
        rules = [
            "Respond with SEARCH/REPLACE diff blocks only.",
            "Do not include prose, explanations, bullet lists, or markdown fences.",
            "History programs are reference-only; use them for ideas, not for SEARCH text.",
            "Copy every SEARCH block exactly from the Current Program section.",
        ]
        if has_edit_window:
            rules.append(
                f"Only modify code between {EVOLVE_BLOCK_START_MARKER} and {EVOLVE_BLOCK_END_MARKER}."
            )
            rules.append("If an idea requires edits outside that boundary, do not propose it.")
        rules_text = "\n".join(f"- {rule}" for rule in rules)
        return (
            "## System Instructions\n"
            f"{system_instructions.strip()}\n\n"
            f"{rules_text}\n\n"
            "Use this exact structure for every change:\n"
            "<<<<<<< SEARCH\n"
            "<old code>\n"
            "=======\n"
            "<new code>\n"
            ">>>>>>> REPLACE\n"
            "Make changes that are internally consistent across the full program."
        )

    def _render_task_section(self, task_contract: str) -> str:
        return f"## Task and Evaluation Contract\n{task_contract.strip()}"

    def _render_feedback(self, evaluator_feedback: str | None) -> str | None:
        if not evaluator_feedback:
            return None
        return f"## Evaluator Feedback\n{evaluator_feedback.strip()}"

    def _render_artifacts(
        self,
        artifact_context: Sequence[PromptArtifactContext],
    ) -> tuple[str | None, tuple[PromptArtifactContext, ...]]:
        limited = tuple(artifact_context[: self._budget.max_artifact_context])
        if not limited:
            return None, ()
        lines = ["## Artifact Summaries"]
        for artifact in limited:
            lines.append(f"- {artifact.name} ({artifact.type}): {artifact.summary}")
        return "\n".join(lines), limited

    def _render_edit_window(self, edit_window: EditWindow | None) -> str | None:
        if edit_window is None:
            return None
        return (
            "## Authorized Edit Window\n"
            f"Only mutate code between {EVOLVE_BLOCK_START_MARKER} and {EVOLVE_BLOCK_END_MARKER}.\n\n"
            "```python\n"
            f"{edit_window.content.rstrip()}\n"
            "```"
        )

    def _render_current_program(self, program: Program) -> str:
        metrics = ", ".join(
            f"{name}={value:.3f}" for name, value in sorted(program.metrics.items())
        ) or "unscored"
        features = ", ".join(
            f"{name}={value:.3f}" for name, value in sorted(program.features.items())
        ) or "none"
        return (
            "## Current Program\n"
            f"Program ID: {program.id}\n"
            f"Parent ID: {program.parent_id or 'none'}\n"
            f"Island ID: {program.island_id}\n"
            f"Latest metrics: {metrics}\n"
            f"Latest features: {features}\n\n"
            "```python\n"
            f"{program.code.rstrip()}\n"
            "```"
        )

    def _render_history_program(self, program: Program) -> str:
        metrics = ", ".join(
            f"{name}={value:.3f}" for name, value in sorted(program.metrics.items())
        ) or "unscored"
        artifact_summary = self._artifact_summary(program)
        artifact_line = f"\nArtifact summaries: {artifact_summary}" if artifact_summary else ""
        return (
            "### History Program\n"
            f"Program ID: {program.id}\n"
            f"Primary score: {program.primary_score:.3f}\n"
            f"Island ID: {program.island_id}\n"
            f"Archive cell: {program.archive_cell}\n"
            f"Metrics: {metrics}{artifact_line}\n\n"
            "```python\n"
            f"{program.code.rstrip()}\n"
            "```"
        )

    def _render_summary_program(self, program: Program) -> str:
        excerpt = _head_tail_excerpt(program.code)
        return (
            "### History Program (Summary Only)\n"
            f"Program ID: {program.id}\n"
            f"Primary score: {program.primary_score:.3f}\n"
            f"Island ID: {program.island_id}\n"
            f"Archive cell: {program.archive_cell}\n"
            "Compact excerpt because the prompt budget is nearly full.\n\n"
            "```python\n"
            f"{excerpt.rstrip()}\n"
            "```"
        )

    def _artifact_summary(self, program: Program) -> str | None:
        summaries = [record.summary for record in program.artifact_records if record.summary]
        if not summaries:
            return None
        return " | ".join(summaries[:2])

    def _rank_history(self, current_program: Program, history: Sequence[Program]) -> list[Program]:
        current_cell = current_program.archive_cell

        def diversity(program: Program) -> int:
            if current_cell is None or program.archive_cell is None:
                return abs(len(program.code) - len(current_program.code))
            return sum(
                abs(program_dim - current_dim)
                for program_dim, current_dim in zip(program.archive_cell, current_cell)
            )

        deduped = {program.id: program for program in history if program.id != current_program.id}
        return sorted(
            deduped.values(),
            key=lambda program: (program.primary_score, diversity(program)),
            reverse=True,
        )
