from __future__ import annotations

import logging
from pathlib import Path

import pytest

from alphaevolve.config import load_experiment_config
from alphaevolve.errors import ConfigError


def test_config_loads_new_sections(tmp_path) -> None:
    seed_path = tmp_path / "seed.py"
    eval_path = tmp_path / "eval.py"
    seed_path.write_text("print('seed')\n", encoding="utf-8")
    eval_path.write_text("def evaluate():\n    return {'score': 1.0}\n", encoding="utf-8")
    config_path = tmp_path / "experiment.toml"
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

[database]
islands = 2

[[database.feature_axes]]
name = "code_bytes"
source = "code_bytes"
scale = "log1p"
bins = [0.0, 1.0, 2.0]

[evaluator]
module = "eval.py"

[checkpoint]
enabled = true

[controller]
max_generations = 2
stagnation_patience = 1
max_inflight = 1
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.database.islands == 2
    assert config.evaluator.module == eval_path.resolve()
    assert config.sandbox.backend == "python"
    assert config.mutation_scope == "evolve_block"
    assert config.sandbox.program_filename == "program.py"
    assert config.sandbox.build_command is None
    assert config.sandbox.run_command == ("python", "{program}")
    assert config.controller.max_inflight == 1


def test_config_rejects_fake_provider_and_backend(tmp_path) -> None:
    seed_path = tmp_path / "seed.py"
    eval_path = tmp_path / "eval.py"
    seed_path.write_text("print('seed')\n", encoding="utf-8")
    eval_path.write_text("def evaluate():\n    return {'score': 1.0}\n", encoding="utf-8")
    config_path = tmp_path / "experiment.toml"
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
provider = "fake"

[sandbox]
backend = "fake"

[evaluator]
module = "eval.py"

[controller]
max_inflight = 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_experiment_config(config_path)


def test_config_accepts_parallel_max_inflight(tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    seed_path = tmp_path / "seed.py"
    eval_path = tmp_path / "eval.py"
    seed_path.write_text("print('seed')\n", encoding="utf-8")
    eval_path.write_text("def evaluate():\n    return {'score': 1.0}\n", encoding="utf-8")
    config_path = tmp_path / "experiment.toml"
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

[controller]
max_inflight = 4
""".strip(),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="alphaevolve.config"):
        config = load_experiment_config(config_path)

    assert config.controller.max_inflight == 4
    assert not caplog.records


def test_config_warns_for_very_high_max_inflight(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seed_path = tmp_path / "seed.py"
    eval_path = tmp_path / "eval.py"
    seed_path.write_text("print('seed')\n", encoding="utf-8")
    eval_path.write_text("def evaluate():\n    return {'score': 1.0}\n", encoding="utf-8")
    config_path = tmp_path / "experiment.toml"
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

[controller]
max_inflight = 64
""".strip(),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="alphaevolve.config"):
        config = load_experiment_config(config_path)

    assert config.controller.max_inflight == 64
    assert any("controller.max_inflight=64 is unusually high" in record.message for record in caplog.records)


def test_config_loads_longcat_model_settings(tmp_path) -> None:
    seed_path = tmp_path / "seed.py"
    eval_path = tmp_path / "eval.py"
    seed_path.write_text("print('seed')\n", encoding="utf-8")
    eval_path.write_text("def evaluate():\n    return {'score': 1.0}\n", encoding="utf-8")
    config_path = tmp_path / "experiment.toml"
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
provider = "longcat"
model = "LongCat-Flash-Chat"
base_url = "https://api.longcat.chat/openai"
api_key_env = "LONGCAT_API_KEY"

[sandbox]
backend = "python"

[evaluator]
module = "eval.py"

[controller]
max_inflight = 1
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.model.provider == "longcat"
    assert config.model.model == "LongCat-Flash-Chat"
    assert config.model.base_url == "https://api.longcat.chat/openai"
    assert config.model.api_key_env == "LONGCAT_API_KEY"


def test_config_loads_parent_sampling_retention_settings(tmp_path) -> None:
    seed_path = tmp_path / "seed.py"
    eval_path = tmp_path / "eval.py"
    seed_path.write_text("print('seed')\n", encoding="utf-8")
    eval_path.write_text("def evaluate():\n    return {'score': 1.0}\n", encoding="utf-8")
    config_path = tmp_path / "experiment.toml"
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

[database.retention]
parent_share_window = 32
parent_share_cap = 0.25
best_parent_cooldown_samples = 4
mutation_failure_streak_threshold = 5
mutation_failure_penalty_samples = 9
mutation_failure_weight_multiplier = 0.4

[evaluator]
module = "eval.py"

[controller]
max_inflight = 1
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.database.retention.parent_share_window == 32
    assert config.database.retention.parent_share_cap == 0.25
    assert config.database.retention.best_parent_cooldown_samples == 4
    assert config.database.retention.mutation_failure_streak_threshold == 5
    assert config.database.retention.mutation_failure_penalty_samples == 9
    assert config.database.retention.mutation_failure_weight_multiplier == 0.4


def test_config_loads_full_file_mutation_scope_and_sandbox_commands(tmp_path) -> None:
    seed_path = tmp_path / "seed.py"
    eval_path = tmp_path / "eval.py"
    seed_path.write_text("print('seed')\n", encoding="utf-8")
    eval_path.write_text("def evaluate():\n    return {'score': 1.0}\n", encoding="utf-8")
    config_path = tmp_path / "experiment.toml"
    config_path.write_text(
        """
name = "demo"
description = "demo"
seed_program = "seed.py"
primary_metric = "score"
target_score = 2.0
mutation_scope = "full_file"
system_instructions = "Improve."
task_description = "Task."
evaluation_contract = "Contract."

[model]
provider = "ollama"

[sandbox]
backend = "python"
program_filename = "candidate.src"
build_command = ["python", "-c", "print('build')"]
run_command = ["python", "{program}"]

[evaluator]
module = "eval.py"

[controller]
max_inflight = 1
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.mutation_scope == "full_file"
    assert config.sandbox.program_filename == "candidate.src"
    assert config.sandbox.build_command == ("python", "-c", "print('build')")
    assert config.sandbox.run_command == ("python", "{program}")


@pytest.mark.parametrize(
    "config_path",
    sorted((Path(__file__).resolve().parents[1] / "experiments").glob("*.toml")),
)
def test_repo_experiment_configs_load(config_path: Path) -> None:
    config = load_experiment_config(config_path)

    assert config.name
    assert config.seed_program_path.exists()
    assert config.evaluator.module.exists()
