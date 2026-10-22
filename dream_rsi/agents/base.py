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

"""Shared agent types and helpers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# --- live subprocess registry (for clean Ctrl-C) ---------------------------
# Agents register every spawned subprocess here. On SIGINT the CLI calls
# shutdown_agents(): it sets SHUTDOWN (agents stop retrying/spawning) and
# SIGKILLs every live process group so no orphan keeps burning API quota.

SHUTDOWN = threading.Event()
_ACTIVE: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.Lock()


class _WindowsJob:
    """Keep Windows descendants in an owned, kill-on-close job object."""

    def __init__(self, proc: subprocess.Popen) -> None:
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        try:
            if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.api.AssignProcessToJobObject(self.handle, int(proc._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            proc.kill()
            raise

    def terminate(self) -> None:
        if self.handle and not self.api.TerminateJobObject(self.handle, 1):
            import ctypes
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def register_proc(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        proc._drsi_job = _WindowsJob(proc)
    with _ACTIVE_LOCK:
        _ACTIVE.add(proc)


def unregister_proc(proc: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.discard(proc)
    job = getattr(proc, "_drsi_job", None)
    if job is not None:
        job.close()


def kill_proc_group(proc: subprocess.Popen) -> None:
    """Stop an owned subprocess tree; its caller must still reap the process."""
    if os.name == "nt":
        job = getattr(proc, "_drsi_job", None)
        if job is not None:
            job.terminate()
            return
        if proc.poll() is not None:
            return
        # Windows has neither killpg nor getpgid. /T includes descendants such
        # as a renderer started by an evaluator; no shell or image-name match.
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode == 0 or proc.poll() is not None:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
        proc.kill()
        raise RuntimeError(f"Could not terminate the full subprocess tree for PID {proc.pid}")
    try:
        # start_new_session=True makes the PID the group ID. The parent may
        # already have exited while a descendant still holds an output pipe.
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def shutdown_agents() -> int:
    """Stop all agents: no new spawns/retries, kill every live subprocess group."""
    SHUTDOWN.set()
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE)
    killed = 0
    for proc in procs:
        if proc.poll() is None:
            kill_proc_group(proc)
            killed += 1
    return killed


@dataclass
class AgentResult:
    status: str = "ok"
    message: str = ""


class Agent(Protocol):
    name: str

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
    ) -> AgentResult: ...


def save_prompt(log_dir: Path, prompt: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "prompt.txt"
    path.write_text(prompt, encoding="utf-8")
    return path


def save_run_logs(log_dir: Path, *, stdout: str = "", stderr: str = "", command: list[str] | None = None) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    if command is not None:
        (log_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    (log_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
    (log_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")


def streaming_enabled() -> bool:
    """When true, agents echo tool calls / eval output to the terminal live."""
    return os.environ.get("DRSI_AGENT_STREAM", "").strip().lower() in ("1", "true", "yes", "on")
