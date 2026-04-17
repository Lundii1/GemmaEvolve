from __future__ import annotations

from pathlib import Path

from alphaevolve.archive import ProgramDatabase
from alphaevolve.cli import main
from alphaevolve.config import load_experiment_config
from alphaevolve.models import Program


def test_clone_best_creates_experiment_copy_with_best_seed(tmp_path: Path) -> None:
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    seed_path = experiments_dir / "seed.py"
    evaluator_path = experiments_dir / "eval.py"
    notes_path = experiments_dir / "notes.txt"
    seed_path.write_text("def evaluate():\n    return {'score': 1.0}\n", encoding="utf-8")
    evaluator_path.write_text("def evaluate_program():\n    return {'score': 1.0}\n", encoding="utf-8")
    notes_path.write_text("keep me\n", encoding="utf-8")
    config_path = experiments_dir / "demo.toml"
    config_path.write_text(
        """
name = "demo"
description = "demo"
seed_program = "seed.py"
primary_metric = "score"
target_score = 2.0
system_instructions = "Improve."
task_description = "Task."
evaluation_contract = "Contract."

[model]
provider = "ollama"

[sandbox]
backend = "python"

[evaluator]
module = "eval.py"

[checkpoint]
enabled = true

[controller]
max_inflight = 1
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)
    run_dir = tmp_path / "runs" / "demo-run"
    database = ProgramDatabase(config.database, run_dir / "database")

    seed_program = Program.seed(seed_path.read_text(encoding="utf-8"))
    seed_program.primary_score = 1.0
    seed_program.metrics = {"score": 1.0}
    database.record_seed_across_islands(seed_program)

    best_program = Program(
        id="prog_best1234",
        code="def evaluate():\n    return {'score': 2.0}\n",
        parent_id=seed_program.id,
        generation=1,
        island_id=0,
    )
    best_program.primary_score = 2.0
    best_program.metrics = {"score": 2.0}
    database.record(best_program)

    clone_dir = tmp_path / "demo-clone"

    exit_code = main(
        [
            "clone-best",
            "--config",
            str(config_path),
            "--run",
            str(run_dir),
            "--output",
            str(clone_dir),
        ]
    )

    assert exit_code == 0
    assert (clone_dir / "demo.toml").exists()
    assert (clone_dir / "eval.py").read_text(encoding="utf-8") == evaluator_path.read_text(encoding="utf-8")
    assert (clone_dir / "notes.txt").read_text(encoding="utf-8") == "keep me\n"
    assert (clone_dir / "seed.py").read_text(encoding="utf-8") == best_program.code
    assert seed_path.read_text(encoding="utf-8") == "def evaluate():\n    return {'score': 1.0}\n"


def test_clone_best_replaces_existing_output_directory(tmp_path: Path) -> None:
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    seed_path = experiments_dir / "seed.py"
    evaluator_path = experiments_dir / "eval.py"
    seed_path.write_text("def evaluate():\n    return {'score': 1.0}\n", encoding="utf-8")
    evaluator_path.write_text("def evaluate_program():\n    return {'score': 1.0}\n", encoding="utf-8")
    config_path = experiments_dir / "demo.toml"
    config_path.write_text(
        """
name = "demo"
description = "demo"
seed_program = "seed.py"
primary_metric = "score"
target_score = 2.0
system_instructions = "Improve."
task_description = "Task."
evaluation_contract = "Contract."

[model]
provider = "ollama"

[sandbox]
backend = "python"

[evaluator]
module = "eval.py"

[checkpoint]
enabled = true

[controller]
max_inflight = 1
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)
    run_dir = tmp_path / "runs" / "demo-run"
    database = ProgramDatabase(config.database, run_dir / "database")

    seed_program = Program.seed(seed_path.read_text(encoding="utf-8"))
    seed_program.primary_score = 1.0
    seed_program.metrics = {"score": 1.0}
    database.record_seed_across_islands(seed_program)

    best_program = Program(
        id="prog_best5678",
        code="def evaluate():\n    return {'score': 3.0}\n",
        parent_id=seed_program.id,
        generation=1,
        island_id=0,
    )
    best_program.primary_score = 3.0
    best_program.metrics = {"score": 3.0}
    database.record(best_program)

    clone_dir = tmp_path / "demo-clone"
    clone_dir.mkdir()
    stale_file = clone_dir / "stale.txt"
    stale_file.write_text("old clone\n", encoding="utf-8")

    exit_code = main(
        [
            "clone-best",
            "--config",
            str(config_path),
            "--run",
            str(run_dir),
            "--output",
            str(clone_dir),
        ]
    )

    assert exit_code == 0
    assert not stale_file.exists()
    assert (clone_dir / "seed.py").read_text(encoding="utf-8") == best_program.code
