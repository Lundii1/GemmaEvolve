from __future__ import annotations

import runpy
from pathlib import Path

from alphaevolve.config import load_experiment_config


def test_turan_tetrahedron_seed_baseline_score_below_reference_target() -> None:
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "experiments" / "turan_tetrahedron_seed.py"))

    result = module["evaluate"]()
    config = load_experiment_config(root / "experiments" / "turan_tetrahedron.toml")

    assert result["score"] == 2408.0
    assert config.target_score == 5502.0
    assert result["score"] < config.target_score
