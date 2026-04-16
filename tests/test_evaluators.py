from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from alphaevolve.evaluators import build_evaluator, docker_environment_status
from alphaevolve.llm import AsyncInferenceClient
from alphaevolve.models import (
    EvaluatorArtifactsConfig,
    EvaluatorConfig,
    EvaluatorFeedbackConfig,
    EvaluatorStageConfig,
    Program,
    SandboxConfig,
)


class _StubInferenceClient(AsyncInferenceClient):
    def __init__(self, response: str) -> None:
        self._response = response

    async def generate_text(self, prompt: str, *, attempt: int = 1) -> str:
        return self._response


class _FakeDockerClient:
    def __init__(self, *, info_payload=None, ping_error: Exception | None = None) -> None:
        self._info_payload = info_payload or {}
        self._ping_error = ping_error
        self.closed = False

    def ping(self) -> None:
        if self._ping_error is not None:
            raise self._ping_error

    def info(self):
        return self._info_payload

    def close(self) -> None:
        self.closed = True


def _install_fake_docker(monkeypatch, client: _FakeDockerClient) -> None:
    fake_module = SimpleNamespace(from_env=lambda: client)
    monkeypatch.setitem(sys.modules, "docker", fake_module)


def _write_module(path: Path, source: str) -> Path:
    path.write_text(source.strip(), encoding="utf-8")
    return path


def test_module_evaluator_runs_cascade_and_captures_artifacts(tmp_path) -> None:
    evaluator_module = _write_module(
        tmp_path / "cascade_eval.py",
        """
async def evaluate_stage1(context):
    metrics, execution = await context.execute_candidate()
    return {
        "status": execution.status,
        "metrics": metrics,
        "features": {"duration_ms": execution.duration_ms},
        "primary_score": metrics.get(context.primary_metric, 0.0),
        "should_continue": execution.status == "success",
        "execution": execution.to_dict(),
        "artifacts": [{"name": "stdout", "type": "text", "content": execution.stdout, "summary": "stage1 stdout"}],
    }

def evaluate_stage2(context, previous_result):
    return {
        "status": previous_result.status,
        "metrics": previous_result.metrics,
        "features": {"stage2_feature": 1.0},
        "primary_score": previous_result.primary_score,
        "should_continue": True,
        "execution": previous_result.execution.to_dict(),
        "artifacts": [{"name": "report", "type": "json", "content": {"score": previous_result.primary_score}, "summary": "score report"}],
    }
""",
    )
    evaluator = build_evaluator(
        EvaluatorConfig(module=evaluator_module),
        SandboxConfig(backend="python", timeout_seconds=5.0),
    )
    program = Program(
        id="prog1",
        code="import json\nprint(json.dumps({'score': 3.0}))\n",
    )

    result = asyncio.run(
        evaluator.evaluate(
            program,
            primary_metric="score",
            artifact_dir=tmp_path / "artifacts",
        )
    )

    assert result.primary_score == 3.0
    assert len(result.stage_results) == 2
    assert len(result.artifacts) == 2
    assert all(record.path.exists() for record in result.artifacts)


def test_module_evaluator_short_circuits_failed_stage(tmp_path) -> None:
    evaluator_module = _write_module(
        tmp_path / "failed_eval.py",
        """
def evaluate_stage1(context):
    return {
        "status": "error",
        "metrics": {"score": 0.0},
        "primary_score": 0.0,
        "should_continue": False,
        "rejection_reason": "stage1_failed",
    }

def evaluate_stage2(context, previous_result):
    raise AssertionError("stage2 should not run")
""",
    )
    evaluator = build_evaluator(
        EvaluatorConfig(module=evaluator_module),
        SandboxConfig(backend="python", timeout_seconds=5.0),
    )
    result = asyncio.run(
        evaluator.evaluate(
            Program(id="prog2", code="print('x')\n"),
            primary_metric="score",
            artifact_dir=tmp_path / "artifacts",
        )
    )

    assert result.status == "error"
    assert len(result.stage_results) == 1
    assert result.rejection_reason == "stage1_failed"


def test_module_evaluator_feedback_uses_inference_client_when_enabled(tmp_path) -> None:
    evaluator_module = _write_module(
        tmp_path / "feedback_eval.py",
        """
def evaluate(context):
    return {
        "status": "error",
        "metrics": {"score": 0.0},
        "primary_score": 0.0,
        "execution": {"status": "error", "stderr": "boom"},
        "rejection_reason": "bad_candidate",
    }
""",
    )
    evaluator = build_evaluator(
        EvaluatorConfig(
            module=evaluator_module,
            stages=EvaluatorStageConfig(enabled=False),
            feedback=EvaluatorFeedbackConfig(enabled=True, on_failure=True, max_feedback_chars=120),
            artifacts=EvaluatorArtifactsConfig(enabled=True),
        ),
        SandboxConfig(backend="python", timeout_seconds=5.0),
        feedback_client=_StubInferenceClient("Focus on the failing path."),
    )

    result = asyncio.run(
        evaluator.evaluate(
            Program(id="prog3", code="print('x')\n"),
            primary_metric="score",
            artifact_dir=tmp_path / "artifacts",
        )
    )

    assert result.feedback == "Focus on the failing path."


def test_docker_environment_status_reports_daemon_unavailable(monkeypatch) -> None:
    client = _FakeDockerClient(ping_error=RuntimeError("daemon unavailable"))
    _install_fake_docker(monkeypatch, client)

    available, reason = docker_environment_status("runsc")

    assert available is False
    assert "daemon unavailable" in reason
    assert client.closed is True


def test_docker_environment_status_reports_missing_runsc(monkeypatch) -> None:
    client = _FakeDockerClient(info_payload={"Runtimes": {"runc": {}}})
    _install_fake_docker(monkeypatch, client)

    available, reason = docker_environment_status("runsc")

    assert available is False
    assert "runsc" in reason
    assert client.closed is True


def test_docker_environment_status_accepts_configured_runsc(monkeypatch) -> None:
    client = _FakeDockerClient(info_payload={"Runtimes": {"runc": {}, "runsc": {}}})
    _install_fake_docker(monkeypatch, client)

    available, reason = docker_environment_status("runsc")

    assert available is True
    assert reason == "ok"
    assert client.closed is True
