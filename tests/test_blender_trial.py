"""Offline checks of trial accounting, selection, and paid-phase boundaries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def agent_log(directory, *, tokens=10, duration=2, available=True):
    dump(directory / "run.json", {
        "returncode": 0, "duration_sec": duration, "timed_out": False, "error": None,
    })
    if available:
        dump(directory / "usage.json", {
            "available": True, "total": {"input_tokens": tokens, "output_tokens": 2},
        })


def candidate(directory, name, *, dev, heldout):
    node = directory / "explore_b0" / name
    program = node / "artifact" / "program.py"
    dump(program, {"dev": dev, "heldout": heldout})
    dump(node / "eval" / "score.json", {"combined_score": dev})
    return program


@pytest.fixture
def trial(monkeypatch):
    spec = importlib.util.spec_from_file_location("trial_under_test", ROOT / "scripts/run_blender_trial.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # setup changes environment variables; isolate them from the rest of pytest.
    monkeypatch.setattr(module.os, "environ", dict(os.environ))

    def unexpected_live_call(*args, **kwargs):
        raise AssertionError("offline trial tests must not launch an agent or live evaluator")

    monkeypatch.setattr(module, "_run_explore_stage", unexpected_live_call)
    monkeypatch.setattr(module, "run_cycle", unexpected_live_call)
    monkeypatch.setattr(module, "_bootstrap_workspace", unexpected_live_call)
    return module


@pytest.mark.parametrize("backend", ["codex", "codex_json"])
def test_usage_counts_learning_and_unique_rounds_without_copied_history(tmp_path, trial, backend):
    e0 = tmp_path / "e0"
    agent_log(e0 / f"explore_b0/attempt_01/agent_log/{backend}", tokens=10)
    agent_log(e0 / f"explore_b0/history/iter099/explore_b0/attempt_01/agent_log/{backend}", tokens=900)
    agent_log(e0 / f"explore_b0/baseline/agent_log/{backend}", tokens=900)
    for arm in ("adaptive", "fixed"):
        shutil.copytree(e0, tmp_path / arm / "history/iter000")
        agent_log(tmp_path / arm / f"history/iter001/explore_b0/attempt_01/agent_log/{backend}", tokens=30)
    learning = tmp_path / "adaptive/learning/iter000"
    agent_log(learning / "logs/r0001_proposal", tokens=20)
    agent_log(learning / "heuristic_history/r0001_copy/logs/r0001_proposal", tokens=900)

    groups = [trial.aggregate_usage([directory]) for directory in (
        e0, learning, tmp_path / "adaptive/history/iter001", tmp_path / "fixed/history/iter001",
    )]
    assert [group["episodes"] for group in groups] == [1, 1, 1, 1]
    assert sum(group["reported_token_totals"]["input_tokens"] for group in groups) == 90
    assert sum(group["duration_sec"] for group in groups) == 8
    assert all(group["usage_available"] == 1 for group in groups)


def test_missing_usage_is_counted_as_an_episode_but_not_as_known_zero_cost(tmp_path, trial):
    agent_log(tmp_path / "logs/r0001_reported", tokens=7)
    agent_log(tmp_path / "logs/r0002_missing", available=False, duration=5)
    result = trial.aggregate_usage([tmp_path])
    assert result["episodes"] == 2
    assert result["usage_available"] == 1
    assert result["reported_token_totals"] == {"input_tokens": 7, "output_tokens": 2}
    assert result["duration_sec"] == 7
    assert result["runs"][1]["usage"]["available"] is False


@pytest.mark.parametrize("incumbent,valid", [(False, True), (False, False), (True, True), (True, False)])
def test_report_checks_used_policy_hash_and_selects_on_dev_only(tmp_path, monkeypatch, trial, incumbent, valid):
    task = tmp_path / "task"
    task.mkdir()
    (task / "problem.txt").write_text("test task", encoding="utf-8")
    dump(task / "init_program.py", {"dev": 0.2, "heldout": 0.3})
    # A deterministic fake evaluator: no candidate execution, model, or Blender.
    (task / "evaluator.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def evaluate_split(program_path, split='dev'):\n"
        "    return {'combined_score': json.loads(Path(program_path).read_text())[split]}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(trial, "TASK", task)
    dest = tmp_path / "run"
    cfg = trial.PipelineConfig(workspace=dest)
    dump(dest / "plan.json", {"task_hashes": trial.task_hashes()})
    winners = {}
    for name, directory in (
        ("e0", dest / "e0"),
        ("adaptive_e1", dest / "adaptive/history/iter001"),
        ("fixed_e1", dest / "fixed/history/iter001"),
    ):
        winners[name] = candidate(directory, "attempt_01", dev=0.9, heldout=0.1)
        candidate(directory, "attempt_02", dev=0.4, heldout=1.0)

    # The incumbent wrapper hash is deliberately different from baseline.py.
    selected = {"incumbent": incumbent, "method_sha256": "wrapper" if incumbent else "learned"}
    dump(dest / "learning_result.json", {"discover": {"best_round": selected}})
    baseline_source = str((ROOT / "dream_rsi/policy/baseline.py").resolve())
    dump(dest / "e0/live_cycle_manifest.json", {"policy": {"source": baseline_source, "sha256": "baseline"}})
    digest = "baseline" if incumbent else "learned"
    source = baseline_source if incumbent else str(dest / "adaptive/policy/method.py")
    dump(dest / "adaptive/history/iter001/live_cycle_manifest.json", {
        "policy": {"source": source, "sha256": digest if valid else "wrong-policy"},
    })
    for path in (dest / "e0/round_0_result.json", dest / "adaptive/round_1_result.json",
                 dest / "fixed/round_1_result.json"):
        dump(path, {"failed": [], "nodes": 2, "best_score": 0.9})

    if valid:
        trial.phase_report(dest, cfg)
    else:
        with pytest.raises(RuntimeError, match="hash verification failed"):
            trial.phase_report(dest, cfg)
    report = trial.read(dest / "report.json")
    assert report["policy"]["verified"] is valid
    for name, expected in winners.items():
        assert report["results"][name]["program"] == str(expected)
        assert report["results"][name]["heldout"]["combined_score"] == 0.1


def test_mock_protocol_shares_e0_and_runs_one_learning_phase_with_13_episode_cap(tmp_path, monkeypatch, trial):
    dest, cfg = trial.setup(argparse.Namespace(output=tmp_path, agent_timeout=12))
    events = []
    episodes = []

    def explore(cfg, ws, *, label, cycle):
        events.append(("explore", ws.name, cycle))
        episodes.append(cfg.explore_max_steps)
        for index in range(4):
            candidate(ws, f"attempt_{index + 1:02d}", dev=0.5, heldout=0.5)
        dump(ws / "raw_scores.json", {"nodes": [
            {"raw": {"evaluated": True, "combined_score": 0.5}} for _ in range(4)
        ]})
        return {"failed": [], "nodes": 4, "best_score": 0.5}

    def archive(ws, *, cycle):
        archived = ws / "history" / f"iter{cycle:03d}"
        archived.mkdir(parents=True)
        for path in [*ws.glob("explore_b*"), ws / "raw_scores.json"]:
            if path.exists():
                shutil.move(str(path), str(archived / path.name))
        return str(archived)

    def learn(cfg, *, cycle, start_stage):
        events.append(("learning", cfg.workspace.name, cycle))
        assert start_stage == "learning"
        assert cfg.evaluate_incumbent and cfg.archive_each_cycle
        assert cfg.discover_retry_rounds == cfg.discover_max_retry_batches == 0
        assert cfg.direction_mode == "off"
        assert cfg.explore_max_width == 2 and cfg.explore_max_refine == 1
        episodes.append(cfg.discover_rounds)
        archive(cfg.workspace, cycle=cycle)
        return {"discover": {"best_round": {"incumbent": True}}, "promoted_method": None}

    monkeypatch.setattr(trial, "_bootstrap_workspace", lambda ws, config: ws.mkdir(parents=True))
    monkeypatch.setattr(trial, "_run_explore_stage", explore)
    monkeypatch.setattr(trial, "_archive_and_reset_workspace", archive)
    monkeypatch.setattr(trial, "run_cycle", learn)
    trial.phase_e0(dest, cfg)
    assert (dest / "e0/raw_scores.json").read_bytes() == (dest / "adaptive/raw_scores.json").read_bytes()
    assert (dest / "e0/raw_scores.json").read_bytes() == (dest / "fixed/raw_scores.json").read_bytes()
    trial.phase_learn(dest, cfg)
    trial.phase_e1(dest, cfg)
    assert events == [
        ("explore", "e0", 0), ("learning", "adaptive", 0),
        ("explore", "adaptive", 1), ("explore", "fixed", 1),
    ]
    assert sum(episodes) == trial.read(dest / "plan.json")["max_agent_episodes"] == 13
    assert (dest / "e0/raw_scores.json").is_file()
    # Repeating E1 must not spend the same paid runs again.
    trial.phase_e1(dest, cfg)
    assert len(events) == 4


def test_failed_e0_stops_before_copy_and_existing_attempt_cannot_be_rerun(tmp_path, monkeypatch, trial):
    cfg = trial.PipelineConfig(workspace=tmp_path / "e0")
    monkeypatch.setattr(trial, "_bootstrap_workspace", lambda ws, config: ws.mkdir(parents=True))

    def failed(cfg, ws, **kwargs):
        dump(ws / "raw_scores.json", {"nodes": [{"raw": {"evaluated": True, "combined_score": 0}}]})
        return {"failed": ["b0_a0"], "nodes": 1}

    monkeypatch.setattr(trial, "_run_explore_stage", failed)
    with pytest.raises(RuntimeError):
        trial.phase_e0(tmp_path, cfg)
    assert not (tmp_path / "adaptive").exists()
    assert not (tmp_path / "fixed").exists()
    assert (tmp_path / "e0/round_0_result.json").exists()
    with pytest.raises(RuntimeError, match="Refusing to repeat"):
        trial.checked_round(cfg, tmp_path / "e0", 0)


@pytest.mark.parametrize("failures", [[], ["b1_a0"]])
def test_e1_accepts_single_zero_score_probe_as_an_outcome(tmp_path, monkeypatch, trial, failures):
    cfg = trial.PipelineConfig(workspace=tmp_path)

    def early_stop(cfg, ws, **kwargs):
        candidate(ws, "attempt_01", dev=0, heldout=0)
        dump(ws / "raw_scores.json", {"nodes": [{"raw": {"evaluated": True, "combined_score": 0}}]})
        return {"failed": failures, "nodes": 1, "best_score": 0}

    monkeypatch.setattr(trial, "_run_explore_stage", early_stop)
    result = trial.checked_round(cfg, tmp_path, 1)
    assert result["nodes"] == 1
    recorded = trial.read(tmp_path / "round_1_result.json")
    assert recorded["best_score"] == 0
    assert recorded["failed"] == failures


@pytest.mark.parametrize("variable,initial,changed", [
    ("CODEX_MODEL", "test-model-a", "test-model-b"),
    ("CODEX_EXTRA_ARGS", "--config model_reasoning_effort=low", "--config model_reasoning_effort=high"),
])
def test_separate_phases_cannot_silently_change_agent_settings(tmp_path, monkeypatch, trial, variable, initial, changed):
    monkeypatch.setenv(variable, initial)
    args = argparse.Namespace(output=tmp_path, agent_timeout=12)
    trial.setup(args)
    monkeypatch.setenv(variable, changed)
    with pytest.raises(RuntimeError, match="changed after trial was frozen"):
        trial.setup(args)
