# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline CLI protocol and process-tree tests; never invokes a model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dream_rsi.agents import codex
from dream_rsi.agents.base import _ACTIVE
from dream_rsi.exploration.evaluate import run_evaluator


@pytest.mark.parametrize("plain_exec", [False, True])
def test_command_uses_exec_options_sandbox_and_stdin(tmp_path, monkeypatch, plain_exec):
    monkeypatch.setattr(codex, "_resolve_codex_bin", lambda _: "codex-test")
    agent = codex.CodexAgent(
        plain_exec=plain_exec, model="test-model",
        extra_args=["--config", "model_reasoning_effort=low", "--color", "never"],
    )
    prompt = "长 prompt\n" * 20000
    command = agent._build_command(tmp_path, prompt, tmp_path / "last.txt")
    assert command[:4] == ["codex-test", "exec", "--config", "model_reasoning_effort=low"]
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert command[-1] == "-"
    assert prompt not in command
    assert "--full-auto" not in command
    assert ("--json" in command) is not plain_exec


@pytest.mark.parametrize("args", [
    ["--dangerously-bypass-approvals-and-sandbox"], ["--yolo"], ["--full-auto"],
    ["--sandbox=danger-full-access"], ["-s", "danger-full-access"],
    ["--ignore-rules"], ["-c", "sandbox_mode='danger-full-access'"],
    ["--config=permissions='unrestricted'"], ["-capproval_policy='never'"],
])
def test_permission_overrides_rejected_before_cli_lookup(tmp_path, args):
    with pytest.raises(ValueError, match="permission"):
        codex.CodexAgent(extra_args=args)._build_command(tmp_path, "x", tmp_path / "last")


def test_usage_keeps_reported_turns_and_does_not_invent_missing_counts():
    stdout = '\n'.join([
        'not json', '[]', '{broken',
        '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":2}}',
        '{"type":"turn.completed","usage":{"input_tokens":7,"cached_input_tokens":3}}',
    ])
    usage = codex._usage_from_jsonl(stdout)
    assert usage["total"] == {"input_tokens": 17, "output_tokens": 2, "cached_input_tokens": 3}
    assert len(usage["turns"]) == 2
    assert codex._usage_from_jsonl("plain text") == {"available": False, "turns": [], "total": None}


def _fake_agent(tmp_path: Path, monkeypatch, source: str, *, timeout=5):
    script = tmp_path / "fake_cli.py"
    script.write_text(source, encoding="utf-8")
    agent = codex.CodexAgent(timeout_sec=timeout)
    monkeypatch.setattr(agent, "_build_command", lambda _cwd, _prompt, last: [
        sys.executable, "-X", "utf8", "-u", str(script), str(last),
    ])
    return agent


def _run(agent, tmp_path, prompt="hello"):
    return agent.run(
        prompt=prompt, workspace=tmp_path, node_dir=tmp_path,
        log_dir=tmp_path / "log", guide={},
    )


def test_large_unicode_prompt_and_usage_are_recorded(tmp_path, monkeypatch):
    agent = _fake_agent(tmp_path, monkeypatch, '''
import json, sys
from pathlib import Path
prompt = sys.stdin.buffer.read().decode("utf-8")
Path(sys.argv[1]).write_text(prompt, encoding="utf-8")
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 13, "output_tokens": 5}}))
print("diagnostic", file=sys.stderr)
''')
    prompt = "3D 椅子\n" * 20000
    assert _run(agent, tmp_path, prompt).status == "ok"
    log = tmp_path / "log"
    assert (log / "codex_last_message.txt").read_text(encoding="utf-8") == prompt
    assert (log / "prompt.txt").read_text(encoding="utf-8") == prompt
    assert "diagnostic" in (log / "stderr.txt").read_text()
    assert json.loads((log / "usage.json").read_text())["total"]["input_tokens"] == 13
    assert json.loads((log / "run.json").read_text())["returncode"] == 0
    assert not _ACTIVE


def test_failed_cli_preserves_stdout_stderr_and_returncode(tmp_path, monkeypatch):
    agent = _fake_agent(tmp_path, monkeypatch, '''
import sys
sys.stdin.read()
print("partial result")
print("failure detail", file=sys.stderr)
sys.exit(7)
''')
    result = _run(agent, tmp_path)
    assert result.status == "failed"
    assert result.message == "codex exit 7"
    assert "partial result" in (tmp_path / "log/stdout.txt").read_text()
    assert "failure detail" in (tmp_path / "log/stderr.txt").read_text()
    assert json.loads((tmp_path / "log/run.json").read_text())["returncode"] == 7


def test_start_failure_is_logged(tmp_path, monkeypatch):
    agent = codex.CodexAgent()
    monkeypatch.setattr(agent, "_build_command", lambda *_: [str(tmp_path / "missing")])
    assert _run(agent, tmp_path).status == "failed"
    assert json.loads((tmp_path / "log/run.json").read_text())["error"]
    assert json.loads((tmp_path / "log/usage.json").read_text())["available"] is False


def _assert_process_stopped(pid: int) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            assert ctypes.get_last_error() == 87  # ERROR_INVALID_PARAMETER: exited PID
            return
        try:
            assert kernel32.WaitForSingleObject(handle, 1000) == 0
        finally:
            kernel32.CloseHandle(handle)
    else:
        # Linux children can briefly remain zombies until their reaper runs.
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            return
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def _tree_source() -> str:
    return '''
import json, os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path("child_pid.txt").write_text(str(child.pid))
print("before timeout", flush=True)
time.sleep(60)
'''


def test_codex_timeout_kills_descendant_and_reaps_parent(tmp_path, monkeypatch):
    seen = []
    original = subprocess.Popen

    def track(*args, **kwargs):
        proc = original(*args, **kwargs)
        if "fake_cli.py" in " ".join(args[0]):
            seen.append(proc)
        return proc

    monkeypatch.setattr(codex.subprocess, "Popen", track)
    agent = _fake_agent(tmp_path, monkeypatch, _tree_source(), timeout=1.5)
    result = _run(agent, tmp_path)
    assert result.status == "failed" and "timeout" in result.message
    assert "before timeout" in (tmp_path / "log/stdout.txt").read_text()
    assert json.loads((tmp_path / "log/run.json").read_text())["timed_out"] is True
    assert seen[0].returncode is not None
    assert seen[0].stdout.closed and seen[0].stderr.closed
    _assert_process_stopped(int((tmp_path / "child_pid.txt").read_text()))
    assert not _ACTIVE


def test_evaluator_timeout_kills_descendant(tmp_path):
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("def evaluate(path):\n" + "\n".join(
        "    " + line for line in _tree_source().strip().splitlines()
    ), encoding="utf-8")
    program = tmp_path / "program.py"
    program.write_text("pass\n")
    metrics = run_evaluator(evaluator, program, timeout_sec=1.5)
    assert metrics == {"error": "evaluator timeout after 1.5s", "score": 0.0}
    _assert_process_stopped(int((tmp_path / "child_pid.txt").read_text()))
    assert not _ACTIVE
