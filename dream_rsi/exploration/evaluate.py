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

"""Run a user evaluator on a candidate program in an isolated subprocess.

Mirrors SimpleTES' EvaluatorWorker, but synchronous (one node at a time per thread).
The evaluator module must expose `evaluate(program_path) -> dict` and the dict
should contain at least a numeric score (``combined_score`` or ``score``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dream_rsi.exploration.config import ExploreConfig

# Runner executed in a fresh interpreter: load the evaluator module by path and
# call evaluate(program). Result is emitted as a single JSON line on stdout.
_RUNNER = textwrap.dedent(
    r"""
    import importlib.util, json, sys, traceback

    def main():
        evaluator_path, program_path = sys.argv[1], sys.argv[2]
        try:
            spec = importlib.util.spec_from_file_location("user_evaluator", evaluator_path)
            if spec is None or spec.loader is None:
                print(json.dumps({"error": "cannot load evaluator", "combined_score": 0.0}))
                sys.exit(1)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            result = mod.evaluate(program_path)
            print("\n__DRSI_RESULT__ " + json.dumps(result))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            print("\n__DRSI_RESULT__ " + json.dumps({"error": err, "combined_score": 0.0}))
            sys.exit(1)

    if __name__ == "__main__":
        main()
    """
).strip()

SCORE_KEYS = ("combined_score", "score", "primary", "value")


def _extract_score(metrics: dict[str, Any]) -> float:
    for key in SCORE_KEYS:
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def _parse_metrics(stdout: str) -> dict[str, Any]:
    marker = "__DRSI_RESULT__ "
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(marker):
            try:
                data = json.loads(line[len(marker) :])
            except json.JSONDecodeError:
                break
            if isinstance(data, dict):
                return data
            return {"error": f"evaluator returned non-dict: {type(data).__name__}", "combined_score": 0.0}
    # Fallback: last JSON-looking line
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return {"error": "no JSON result from evaluator", "combined_score": 0.0}


def run_evaluator(
    evaluator_path: Path,
    program_path: Path,
    *,
    timeout_sec: float = 600.0,
    python_executable: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate ``program_path`` with ``evaluator_path`` in a killable subprocess.

    Returns the evaluator's metrics dict, always augmented with a numeric
    ``score`` key so downstream heuristics can read it uniformly.
    """
    evaluator_path = Path(evaluator_path).resolve()
    program_path = Path(program_path).resolve()
    if not evaluator_path.is_file():
        return {"error": f"evaluator not found: {evaluator_path}", "score": 0.0}
    if not program_path.is_file():
        return {"error": f"program not found: {program_path}", "score": 0.0}

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    cmd = [
        python_executable or sys.executable,
        "-c",
        _RUNNER,
        str(evaluator_path),
        str(program_path),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(evaluator_path.parent),
            start_new_session=True,
        )
    except OSError as exc:
        return {"error": f"failed to spawn evaluator: {exc}", "score": 0.0}

    # Register so a Ctrl-C in the CLI also kills in-flight evaluations.
    from dream_rsi.agents.base import kill_proc_group, register_proc, unregister_proc

    try:
        register_proc(proc)
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            kill_proc_group(proc)
        finally:
            # Drain/close both pipes and reap the child after tree termination.
            unregister_proc(proc)
            proc.communicate(timeout=10)
        return {"error": f"evaluator timeout after {timeout_sec}s", "score": 0.0}
    except BaseException:
        try:
            kill_proc_group(proc)
        finally:
            unregister_proc(proc)
            proc.communicate(timeout=10)
        raise
    finally:
        unregister_proc(proc)

    metrics = _parse_metrics(stdout or "")
    if proc.returncode != 0 and "error" not in metrics:
        tail = (stderr or stdout or "")[-500:]
        metrics = {"error": f"evaluator exit {proc.returncode}: {tail}", **metrics}

    metrics["score"] = _extract_score(metrics)
    return metrics


def evaluate_node(node_dir: Path, config: "ExploreConfig | None" = None) -> dict[str, Any] | None:
    """Score the node's program and write eval/score.json (no-op if eval disabled).

    The candidate program is node_dir / config.eval_program. Returns the metrics
    dict, or None when no evaluator is configured.
    """
    from dream_rsi.exploration.config import ExploreConfig

    config = config or ExploreConfig.from_env()
    if not config.eval_enabled():
        return None

    program = node_dir / config.eval_program
    metrics = run_evaluator(
        config.evaluator,
        program,
        timeout_sec=config.eval_timeout_sec,
        python_executable=config.eval_python,
    )

    eval_dir = node_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "score.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
