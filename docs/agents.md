# AlphaEvolve Open-Source Implementation: Coding Agent Blueprint

## Overview
This document serves as the architectural blueprint and execution plan for building an open-source, asynchronous evolutionary code-generation pipeline inspired by DeepMind's AlphaEvolve. The system prioritizes local execution, high throughput, and secure isolation, making it ideal for researching optimization techniques in complex neural architectures, physical simulations, and low-level system operations.

## Architectural Pillars

### 1. Core Architecture & Data Structures (Python)
The foundation of the pipeline relies on modern, asynchronous Python (`asyncio`) to maintain high throughput and non-blocking execution.
* **Initialization:** Set up a modular project structure with strict type hinting and a robust logging framework to monitor throughput and error rates.
* **Data Models:**
    * `Program`: A dataclass storing the source code string, historical evaluation score(s), execution logs, and a unique identifier.
    * `Diff`: A dataclass strictly modeling the `<<<<<<< SEARCH` and `>>>>>>> REPLACE` mutation format.

### 2. The LLM Orchestrator (Ollama Integration)
The generation engine utilizes a local open-weight model (`gemma4:26b-a4b-it-q4_K_M`) orchestrated via Ollama.
* **Asynchronous Client:** Implement a non-blocking API client (using `aiohttp` or `httpx`) pointing to the local Ollama instance (default port 11434).
* **PromptBuilder Module:** Construct dynamic prompts injecting system instructions, historical high-performing code (context), and the targeted code block.
* **Parser & Retry Logic:** Build resilient parsing to extract the specific SEARCH/REPLACE diff format, with automatic retries for malformed JSON or invalid diffs.
* **Design Note:** Keep the interface abstract. While Ollama handles the current workload, the architecture must allow seamless swapping to custom PyTorch or Hugging Face inference loops for future parameter tuning or sequence modeling experiments.

### 3. The Evaluator Sandbox (gVisor & Docker)
Executing untrusted, AI-generated code requires an impenetrable, ephemeral sandbox to prevent system instability.
* **Docker SDK Integration:** Programmatically manage container lifecycles using the `docker` Python SDK.
* **Isolation Engine:** Utilize the `runsc` (gVisor) runtime for robust kernel-level system call interception without the overhead of a full VM.
* **Environment Constraints:**
    * `network_mode='none'` to ensure a strictly air-gapped environment.
    * `read_only=True` for the root filesystem.
    * Mount a highly restricted `tmpfs` volume (e.g., 5MB at `/tmp/eval`) exclusively for capturing standard output and evaluation metrics.
* **Resource Quotas:** Enforce strict cgroup limits (`mem_limit`, `cpu_quota`) to immediately kill fork bombs or memory leaks.
* **Timeouts:** Wrap the execution command in an `asyncio.wait_for` hard wall-clock timeout (e.g., 10 seconds).

### 4. The MAP-Elites Database
The evolutionary memory of the system, balancing exploitation of high scores with exploration of novel logic.
* **Structure:** Implement a `ProgramDatabase` utilizing a multi-dimensional archive inspired by MAP-elites. Programs are binned not just by score, but by behavioral characteristics (e.g., AST depth, execution time, reliance on specific libraries).
* **Sampling Strategy:** The `sample()` method must mathematically favor a mix of elite performers and diverse, under-explored approaches to avoid collapsing into local optima.

### 5. The Asynchronous Controller Loop
The central nervous system linking all components into a concurrent pipeline.
* **Lifecycle Management:** Implement the main `async` loop: `Sample -> Prompt -> Generate -> Apply Diff -> Evaluate -> Update Database`.
* **Concurrency Control:** Deploy `asyncio.Semaphore` or queues to throttle simultaneous requests, preventing the Ollama instance or Docker daemon from being overwhelmed by the RTX 4060 Ti's generation speed.

### 6. The "Hello World" Benchmark (Knapsack Heuristic)
Validate the end-to-end pipeline using a deterministic, lightweight optimization problem.
* **Initial Seed:** Inject the Fractional Knapsack heuristic script as the initial `Program` seed.
* **Evaluation Capture:** Configure the sandbox to capture stdout, parsing the JSON string outputted by the `evaluate()` function to extract the score float.
* **Success Criteria:** Run the pipeline until the `ProgramDatabase` registers a program with the known global optimum score of 275.0, confirming that generation, mutation, execution, and selection are all functioning correctly.
