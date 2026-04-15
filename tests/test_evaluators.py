from __future__ import annotations

import sys
from types import SimpleNamespace

from alphaevolve.evaluators import docker_environment_status


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
