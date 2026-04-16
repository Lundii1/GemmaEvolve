from __future__ import annotations

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
