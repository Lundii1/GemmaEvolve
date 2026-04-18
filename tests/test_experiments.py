from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from alphaevolve.config import load_experiment_config
from alphaevolve.models import Program
from alphaevolve.prompts import PromptBuilder


def test_turan_tetrahedron_seed_initializer_stays_in_5286_basin() -> None:
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "experiments" / "turan_tetrahedron_seed.py"))

    result = module["evaluate"]()
    config = load_experiment_config(root / "experiments" / "turan_tetrahedron.toml")

    assert result["seed_score"] == 5286.0
    assert result["seed_signature_111_edges"] == 2408.0
    assert result["seed_signature_210_edges"] == 2413.0
    assert result["seed_signature_300_edges"] == 465.0
    assert result["score"] >= result["seed_score"]
    assert config.target_score == 5502.0
    assert result["seed_score"] < config.target_score


def test_turan_seed_improver_never_returns_worse_than_initializer() -> None:
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "experiments" / "turan_tetrahedron_seed.py"))
    num_vertices = 24
    part_of = module["_balanced_part_assignment"](num_vertices)
    initial_edges = module["_construct_seed_hypergraph"](num_vertices)

    improved_edges = module["improve_construction"](
        num_vertices,
        part_of,
        initial_edges,
        module["MAX_STEPS_BY_N"][num_vertices],
    )

    assert len(improved_edges) >= len(initial_edges)


def test_turan_invalid_improver_output_is_rejected() -> None:
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "experiments" / "turan_tetrahedron_seed.py"))

    def invalid_improver(num_vertices, part_of, initial_edges, max_steps):
        del num_vertices, part_of, initial_edges, max_steps
        return ((0, 1, 2), (0, 1, 2))

    module["evaluate"].__globals__["improve_construction"] = invalid_improver

    with pytest.raises(ValueError):
        module["evaluate"]()


def test_turan_evaluate_is_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "experiments" / "turan_tetrahedron_seed.py"))

    assert module["evaluate"]() == module["evaluate"]()


def test_turan_prompt_mentions_search_mode_contract_and_5502_hint() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "experiments" / "turan_tetrahedron.toml")
    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    rendered = PromptBuilder(config.model.prompt_budget).build(
        system_instructions=config.system_instructions,
        task_contract=f"{config.task_description}\n\n{config.evaluation_contract}",
        current_program=Program.seed(seed_code),
        history=[],
    )

    assert "5502-class reference construction is known" in rendered.text
    assert "Only the bounded improve_construction heuristic inside the EVOLVE block should change." in rendered.text
    assert "initial_edges" in rendered.text
    assert "max_steps" in rendered.text


def test_turan_full_file_prompt_allows_file_wide_mutation() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "experiments" / "turan_tetrahedron_full_file.toml")
    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    rendered = PromptBuilder(
        config.model.prompt_budget,
        mutation_scope=config.mutation_scope,
    ).build(
        system_instructions=config.system_instructions,
        task_contract=f"{config.task_description}\n\n{config.evaluation_contract}",
        current_program=Program.seed(seed_code),
        history=[],
    )

    assert config.mutation_scope == "full_file"
    assert "## Authorized Edit Window" not in rendered.text
    assert "You may modify any part of the current program file." in rendered.text
    assert "rewrite any helper, import, or algorithmic phase in the file" in rendered.text


def test_turan_score_evaluator_reports_delta_metrics_and_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    evaluator_module = runpy.run_path(str(root / "experiments" / "score_evaluator.py"))
    evaluate_stage2 = evaluator_module["evaluate_stage2"]

    class DummyExecution:
        def to_dict(self):
            return {"status": "success"}

    class DummyPreviousResult:
        def __init__(self):
            self.status = "success"
            self.metrics = {
                "seed_score": 5286.0,
                "score": 5291.0,
                "n13_edges": 173.0,
                "n14_edges": 216.0,
                "n16_edges": 327.0,
                "n17_edges": 395.0,
                "n19_edges": 556.0,
                "n20_edges": 650.0,
                "n22_edges": 872.0,
                "n23_edges": 997.0,
                "n24_edges": 1105.0,
                "delta_n13_edges": 1.0,
                "delta_n14_edges": 0.0,
                "delta_n16_edges": 0.0,
                "delta_n17_edges": 0.0,
                "delta_n19_edges": 0.0,
                "delta_n20_edges": 0.0,
                "delta_n22_edges": 2.0,
                "delta_n23_edges": 0.0,
                "delta_n24_edges": 2.0,
                "signature_111_edges": 2408.0,
                "signature_210_edges": 2418.0,
                "signature_300_edges": 465.0,
            }
            self.features = {"duration_ms": 100.0, "code_bytes": 1000.0}
            self.execution = DummyExecution()

    class DummyProgram:
        id = "prog_1"
        code = "print('hello')"

    class DummyContext:
        primary_metric = "score"
        program = DummyProgram()

    result = evaluate_stage2(DummyContext(), DummyPreviousResult())

    assert result["features"]["delta_score"] == 5.0
    assert result["features"]["signature_210_ratio"] > 0.0
    assert result["artifacts"][0]["summary"] == "score=5291.000 seed=5286.000 delta=+5.000"
    assert "delta=+5" in result["artifacts"][1]["summary"]
    assert "d13=+1" in result["artifacts"][2]["summary"]
