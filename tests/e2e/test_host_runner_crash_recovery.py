"""A host-owned runner crash recovers its interrupted sessions without new input."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from omnigent.runner.identity import OMNIGENT_INTERNAL_WS_ORIGIN
from tests.e2e.conftest import (
    configure_mock_llm,
    get_mock_requests,
    lookup_agent_id,
    send_user_message_to_session,
    upload_agent,
)
from tests.e2e.test_host_e2e import (
    _runner_pid_from_daemon_log,
    _spawn_host_daemon,
    _wait_for_host_online,
    _write_smoke_agent_yaml,
)


def _wait_until(predicate: Callable[[], bool], timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.25)
    assert predicate(), "recovery did not reach its expected state"


@pytest.mark.timeout(240)
@pytest.mark.parametrize("stop_target", [None, "parent", "child"])
def test_host_runner_recovers_group_only_after_crash(
    live_server: str,
    http_client: httpx.Client,
    tmp_path: Path,
    mock_llm_server_url: str,
    stop_target: str | None,
) -> None:
    daemon = _spawn_host_daemon(
        tmp_path=tmp_path, live_server=live_server, mock_llm_server_url=mock_llm_server_url
    )
    try:
        _wait_for_host_online(http_client, daemon.host_id)
        agent_name = upload_agent(http_client, _write_smoke_agent_yaml(tmp_path))
        agent_id = lookup_agent_id(http_client, agent_name)
        response = http_client.post("/v1/sessions", json={"agent_id": agent_id})
        response.raise_for_status()
        parent = response.json()["id"]
        launch = http_client.post(
            f"/v1/hosts/{daemon.host_id}/runners",
            json={"session_id": parent, "workspace": str(tmp_path)},
            timeout=60,
        )
        launch.raise_for_status()
        original_runner = launch.json()["runner_id"]
        _wait_until(
            lambda: (
                http_client.get(f"/v1/runners/{original_runner}/status").json().get("online")
                is True
            )
        )
        http_client.patch(
            f"/v1/sessions/{parent}", json={"runner_id": original_runner}
        ).raise_for_status()

        def child() -> str:
            response = http_client.post(
                "/v1/sessions",
                json={"agent_id": agent_id, "parent_session_id": parent},
                headers={"Origin": OMNIGENT_INTERNAL_WS_ORIGIN},
            )
            response.raise_for_status()
            return response.json()["id"]

        worker, completed = child(), child()

        def snapshot(session_id: str) -> dict:
            response = http_client.get(f"/v1/sessions/{session_id}")
            response.raise_for_status()
            return response.json()

        def transcript(session_id: str) -> str:
            response = http_client.get(f"/v1/sessions/{session_id}/items")
            response.raise_for_status()
            return json.dumps(response.json())

        completed_token = f"completed-{uuid.uuid4().hex}"
        configure_mock_llm(
            mock_llm_server_url, [{"text": "ALREADY_FINISHED"}], match=completed_token
        )
        send_user_message_to_session(http_client, session_id=completed, content=completed_token)
        _wait_until(
            lambda: (
                "ALREADY_FINISHED" in transcript(completed)
                and snapshot(completed)["status"] == "idle"
            )
        )

        for session_id, marker in [(parent, "PARENT_RECOVERED"), (worker, "CHILD_RECOVERED")]:
            token = f"hold-{uuid.uuid4().hex}"
            configure_mock_llm(
                mock_llm_server_url,
                [{"text": "BLOCKED", "block": True}, {"text": marker}],
                match=token,
            )
            send_user_message_to_session(http_client, session_id=session_id, content=token)
            _wait_until(
                lambda token=token: any(
                    r.get("model") == "gpt-5.4" and token in json.dumps(r)
                    for r in get_mock_requests(mock_llm_server_url)
                )
            )
            assert snapshot(session_id)["status"] == "running"
            assert snapshot(session_id)["runner_id"] == original_runner

        if stop_target is not None:
            stopped_id = parent if stop_target == "parent" else worker
            response = http_client.post(
                f"/v1/sessions/{stopped_id}/events", json={"type": "stop_session", "data": {}}
            )
            response.raise_for_status()
            assert snapshot(stopped_id)["labels"]["omnigent.runner_recovery.stopped"] == "true"
        if stop_target == "parent":
            # Parent Stop terminates the dedicated runner itself.
            time.sleep(12)
            assert snapshot(parent)["runner_id"] == original_runner
            assert snapshot(worker)["runner_id"] == original_runner
            assert "CHILD_RECOVERED" not in transcript(worker)
            assert "PARENT_RECOVERED" not in transcript(parent)
        else:
            runner_pid = _runner_pid_from_daemon_log(daemon.daemon_log)
            assert runner_pid is not None
            os.kill(runner_pid, signal.SIGKILL)
            expected = [(parent, "PARENT_RECOVERED")]
            if stop_target is None:
                expected.append((worker, "CHILD_RECOVERED"))
            for session_id, marker in expected:
                _wait_until(
                    lambda marker=marker, session_id=session_id: (
                        marker in transcript(session_id)
                        and snapshot(session_id)["status"] == "idle"
                    ),
                    timeout=90,
                )
            recovered_parent = snapshot(parent)
            assert recovered_parent["runner_id"] != original_runner
            assert recovered_parent["host_id"] == daemon.host_id
            assert recovered_parent["workspace"] == str(tmp_path)
            assert transcript(parent).count("PARENT_RECOVERED") == 1
            if stop_target == "child":
                assert snapshot(worker)["runner_id"] == original_runner
                assert "CHILD_RECOVERED" not in transcript(worker)
            else:
                assert recovered_parent["runner_id"] == snapshot(worker)["runner_id"]
                assert transcript(worker).count("CHILD_RECOVERED") == 1
        assert snapshot(completed)["runner_id"] == original_runner
        assert snapshot(completed)["status"] == "idle"
        assert transcript(completed).count("ALREADY_FINISHED") == 1
    finally:
        daemon.proc.terminate()
        try:
            daemon.proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            daemon.proc.kill()
            daemon.proc.wait(timeout=5)
