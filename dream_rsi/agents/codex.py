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

"""OpenAI Codex CLI coding agent."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from dream_rsi.agents.base import (
    SHUTDOWN,
    AgentResult,
    kill_proc_group,
    register_proc,
    save_prompt,
    save_run_logs,
    unregister_proc,
)

_verified_codex: set[str] = set()


def _assert_openai_codex(codex_path: str) -> None:
    if codex_path in _verified_codex:
        return
    cp = subprocess.run(
        [codex_path, "exec", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        env=os.environ.copy(),
    )
    blob = ((cp.stdout or "") + (cp.stderr or "")).lower()
    if any(m in blob for m in ("granian", "django", ":9810", "listening at: http")):
        raise RuntimeError(f"Wrong `codex` binary at {codex_path} (not OpenAI Codex CLI)")
    if cp.returncode != 0 or "--sandbox" not in blob or "--output-last-message" not in blob:
        raise RuntimeError(f"Cannot verify a compatible OpenAI Codex CLI at {codex_path}")
    _verified_codex.add(codex_path)


def _resolve_codex_bin(codex_bin: str) -> str:
    if codex_bin == "codex":
        found = shutil.which("codex")
        if not found:
            raise FileNotFoundError("Cannot find `codex` in PATH")
        path = found
    else:
        path = str(Path(codex_bin).resolve())
    _assert_openai_codex(path)
    return path


def _usage_from_jsonl(stdout: str) -> dict[str, Any]:
    """Preserve reported per-turn token counts; absent usage is not zero cost."""
    turns = []
    totals: dict[str, int | float] = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "turn.completed":
            continue
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        turns.append(usage)
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    return {"available": bool(turns), "turns": turns, "total": totals if turns else None}


def _validate_extra_args(args: list[str]) -> None:
    """Extra exec options must not replace the backend's permission boundary."""
    forbidden = {"--yolo", "--full-auto", "--ignore-rules", "--sandbox", "-s"}
    permission_keys = {"sandbox_mode", "sandbox_permissions", "permissions", "approval_policy"}
    for index, arg in enumerate(args):
        option = arg.split("=", 1)[0]
        if option in forbidden or option.startswith("--dangerously-"):
            raise ValueError(f"CODEX_EXTRA_ARGS cannot override permissions: {option}")
        config = None
        if arg in ("-c", "--config") and index + 1 < len(args):
            config = args[index + 1]
        elif arg.startswith("--config="):
            config = arg.split("=", 1)[1]
        elif arg.startswith("-c") and arg != "-c":
            config = arg[2:]
        if config and config.split("=", 1)[0].split(".", 1)[0].strip() in permission_keys:
            raise ValueError("CODEX_EXTRA_ARGS cannot override permission configuration")


class CodexAgent:
    """
    OpenAI Codex CLI coding agent.

    Env: CODEX_BIN, CODEX_MODEL, CODEX_EXEC_TIMEOUT_SEC, CODEX_EXTRA_ARGS, CODEX_PLAIN_EXEC
    """

    name = "codex"

    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        model: str | None = None,
        timeout_sec: float | None = 600.0,
        extra_args: list[str] | None = None,
        plain_exec: bool = False,
    ) -> None:
        self.codex_bin = codex_bin
        self.model = model
        self.timeout_sec = timeout_sec
        self.extra_args = extra_args or []
        self.plain_exec = plain_exec

    def _build_command(self, workspace: Path, prompt: str, last_message: Path) -> list[str]:
        del prompt  # Always sent on stdin, including on Windows (command length).
        _validate_extra_args(self.extra_args)
        codex_path = _resolve_codex_bin(self.codex_bin)
        cmd: list[str] = [
            codex_path, "exec", *self.extra_args,
            "--sandbox", "workspace-write", "--skip-git-repo-check", "-C", str(workspace),
        ]
        if not self.plain_exec:
            cmd.append("--json")
        if self.model:
            cmd.extend(["-m", self.model])
        cmd.extend(["-o", str(last_message), "-"])
        return cmd

    def run(
        self,
        *,
        prompt: str,
        workspace: Path,
        node_dir: Path,
        log_dir: Path,
        guide: dict[str, Any],
        run_cwd: Path | None = None,
        stage: str | None = None,
    ) -> AgentResult:
        del stage, node_dir, guide
        save_prompt(log_dir, prompt)
        last_message = (log_dir / "codex_last_message.txt").resolve()
        cwd = (run_cwd or workspace).resolve()
        cmd: list[str] | None = None
        proc: subprocess.Popen | None = None
        stdout = stderr = b""
        started = time.monotonic()
        timed_out = False
        error = ""
        result = AgentResult(message="codex finished")
        try:
            if SHUTDOWN.is_set():
                raise RuntimeError("codex cancelled: shutdown requested")
            cmd = self._build_command(cwd, prompt, last_message)
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd),
                env=os.environ.copy(),
                start_new_session=True,
            )
            register_proc(proc)
            stdout, stderr = proc.communicate(
                input=prompt.encode("utf-8"), timeout=self.timeout_sec or None,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout, stderr = exc.output or b"", exc.stderr or b""
            error = f"codex timeout after {self.timeout_sec}s"
            result = AgentResult(status="failed", message=error)
        except Exception as exc:
            error = str(exc)
            result = AgentResult(status="failed", message=error)
        finally:
            if proc is not None:
                try:
                    if timed_out or proc.poll() is None:
                        try:
                            kill_proc_group(proc)
                        finally:
                            # Closing the Windows job is a second tree-cleanup
                            # guarantee, including if termination itself fails.
                            unregister_proc(proc)
                        stdout, stderr = proc.communicate(timeout=10)
                except Exception as exc:
                    error = f"{error}; cleanup failed: {exc}".lstrip("; ")
                    result = AgentResult(status="failed", message=error)
                    proc.kill()
                    proc.wait(timeout=10)
                finally:
                    unregister_proc(proc)
            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            save_run_logs(log_dir, stdout=out, stderr=err, command=cmd)
            (log_dir / "usage.json").write_text(
                json.dumps(_usage_from_jsonl(out), indent=2), encoding="utf-8",
            )
            (log_dir / "run.json").write_text(json.dumps({
                "returncode": proc.returncode if proc else None,
                "duration_sec": time.monotonic() - started,
                "timed_out": timed_out,
                "error": error or None,
            }, indent=2), encoding="utf-8")
        if result.status == "failed":
            return result
        if proc is not None and proc.returncode != 0:
            return AgentResult(status="failed", message=f"codex exit {proc.returncode}")
        return result


def from_env() -> CodexAgent:
    timeout_raw = os.environ.get("CODEX_EXEC_TIMEOUT_SEC", "600").strip()
    return CodexAgent(
        codex_bin=os.environ.get("CODEX_BIN", "codex").strip() or "codex",
        model=os.environ.get("CODEX_MODEL") or None,
        timeout_sec=float(timeout_raw) if timeout_raw else None,
        extra_args=shlex.split(os.environ.get("CODEX_EXTRA_ARGS", "").strip()),
        plain_exec=os.environ.get("CODEX_PLAIN_EXEC", "").lower() in ("1", "true", "yes"),
    )
