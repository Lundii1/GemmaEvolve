# AlphaEvolve

An evolutionary code-generation pipeline inspired by AlphaEvolve and OpenEvolve-style program search.

This repo now uses:

- a filesystem-backed, island-aware program database
- generic evaluator modules with staged evaluation
- prompt logging, artifact capture, checkpoint/resume, and early stopping
- real runtimes only in production paths (`ollama`, local Python, optional Docker/gVisor)

## Requirements

- Python 3.12 or newer
- `pip`
- Optional for local model generation:
  - [Ollama](https://ollama.com/) running locally
- Optional for hardened execution:
  - Docker
  - gVisor `runsc`

## Installation

Create and activate a virtual environment if you want an isolated setup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```powershell
python -m pip install -e .
```

Install development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Install Docker support:

```powershell
python -m pip install -e ".[docker]"
```

Install everything commonly used for development:

```powershell
python -m pip install -e ".[dev,docker]"
```

On Windows, the generated `alphaevolve` console script may not be on `PATH`. The most reliable invocation is:

```powershell
python -m alphaevolve.cli --help
```

## Quick Start

The repo includes a knapsack benchmark in [experiments/knapsack.toml](/C:/Users/amine/OneDrive/Desktop/GemmaEvolve/experiments/knapsack.toml) with:

- `ollama` generation
- staged evaluator logic in [experiments/knapsack_evaluator.py](/C:/Users/amine/OneDrive/Desktop/GemmaEvolve/experiments/knapsack_evaluator.py)
- optional Docker/gVisor execution

There are also harder deterministic benchmarks you can try:

- [experiments/knapsack_portfolio.toml](/C:/Users/amine/OneDrive/Desktop/GemmaEvolve/experiments/knapsack_portfolio.toml) for multi-instance 0/1 knapsack with an exact target of `1860`
- [experiments/weighted_tardiness.toml](/C:/Users/amine/OneDrive/Desktop/GemmaEvolve/experiments/weighted_tardiness.toml) for single-machine weighted tardiness scheduling with an exact target of `3413`
- [experiments/budgeted_coverage.toml](/C:/Users/amine/OneDrive/Desktop/GemmaEvolve/experiments/budgeted_coverage.toml) for budgeted maximum coverage with an exact target of `343`

Run a fresh experiment:

```powershell
python -m alphaevolve.cli run --config experiments/knapsack.toml
```

Resume from a previous run directory:

```powershell
python -m alphaevolve.cli run --resume runs\<previous-run>
```

Check runtime availability first if needed:

```powershell
python -m alphaevolve.cli smoke-ollama --config experiments/knapsack.toml
python -m alphaevolve.cli smoke-docker --config experiments/knapsack.toml
```

## CLI

Run a new experiment:

```powershell
python -m alphaevolve.cli run --config <path-to-config>
```

Resume an existing run:

```powershell
python -m alphaevolve.cli run --resume <run-directory>
```

Store fresh runs under a custom parent directory:

```powershell
python -m alphaevolve.cli run --config <path-to-config> --output-dir <runs-directory>
```

## Experiment Configuration

Experiment configs are TOML files.

Top-level fields:

- `name`
- `description`
- `seed_program`
- `system_instructions`
- `task_description`
- `evaluation_contract`
- `primary_metric`
- `target_score`

Sections:

- `[model]`
- `[sandbox]`
- `[database]`
- `[evaluator]`
- `[checkpoint]`
- `[controller]`

Important notes:

- `model.provider='fake'` is no longer supported.
- `sandbox.backend='fake'` is no longer supported.
- The current engine is serial-first, so `controller.max_inflight` must stay `1`.
- `[archive]` is only treated as a deprecated compatibility alias for `[database]`.

### Evaluator Modules

`[evaluator].module` points to a Python file that exports either:

- `evaluate(context)`, or
- staged functions such as `evaluate_stage1(context)`, `evaluate_stage2(context, previous_result)`

Each evaluator function can return a `dict`, `StageResult`, or `EvaluationResult`. In the common `dict` case, supported keys include:

- `status`
- `metrics`
- `features`
- `primary_score`
- `should_continue`
- `rejection_reason`
- `execution`
- `artifacts`

`context.execute_candidate()` runs the candidate program through the configured backend and returns:

- parsed metrics
- an `ExecutionResult`

Artifacts can be emitted as text, JSON, or file references and are copied into the run directory.

## Run Output Layout

Each run directory contains:

- `config.toml`
- `checkpoint.json`
- `database/state.json`
- `database/programs/*.json`
- `database/prompts/*.json`
- `database/artifacts/<program-id>/...`

This is what enables prompt logs, lineage inspection, artifact reuse, and resume.

## Evolving Code Safely

This project supports editable regions marked with:

```python
# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
```

Only code inside the block is intended to be mutated. The included knapsack seed demonstrates this in [experiments/knapsack_seed.py](/C:/Users/amine/OneDrive/Desktop/GemmaEvolve/experiments/knapsack_seed.py).

LLM responses must be emitted as SEARCH/REPLACE blocks:

```text
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE
```

## Local Development

Run the test suite:

```powershell
python -m pytest -q
```

Current tests cover:

- config loading and fake-runtime rejection
- prompt construction
- diff parsing/application
- island database persistence, migration, and novelty checks
- staged evaluators, artifact capture, and evaluator feedback
- controller retries, checkpoint/resume, target stopping, and stagnation stopping

## Troubleshooting

### `ModuleNotFoundError: No module named 'alphaevolve'`

Install the package first:

```powershell
python -m pip install -e .
```

### `alphaevolve` command is not recognized

Use the module form instead:

```powershell
python -m alphaevolve.cli --help
```

### Docker smoke test reports `docker Python package is not installed`

Install the Docker extra:

```powershell
python -m pip install -e ".[docker]"
```

### Ollama run fails

Check:

- Ollama is running
- the model name in your config exists locally
- `base_url` matches your Ollama instance

## Project Layout

- [src/alphaevolve](/C:/Users/amine/OneDrive/Desktop/GemmaEvolve/src/alphaevolve) - package source
- [experiments](/C:/Users/amine/OneDrive/Desktop/GemmaEvolve/experiments) - example configs, evaluator modules, and seeds
- [tests](/C:/Users/amine/OneDrive/Desktop/GemmaEvolve/tests) - test suite
- [docs](/C:/Users/amine/OneDrive/Desktop/GemmaEvolve/docs) - papers and notes
