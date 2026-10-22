"""Small, auditable shared-E0 DREAM-RSI versus fixed-policy 3D trial.

Run from the repository root after `pip install -e .`. Each phase refuses to
rerun a live round. Logs stay under ignored workspace/. No Blender or model
call is needed for --phase report. See docs/3D_BASELINE.md.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

from dream_rsi.exploration.config import ExploreConfig
from dream_rsi.orchestrate import (
    PipelineConfig, _bootstrap_workspace, _run_explore_stage,
    _archive_and_reset_workspace, run_cycle,
)

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "blender_layout"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_hashes():
    return {p.name: sha(p) for p in sorted(TASK.iterdir()) if p.suffix in (".py", ".txt")}


def checked_round(cfg, ws, cycle):
    if (ws / "raw_scores.json").exists() or (ws / "history" / f"iter{cycle:03d}").exists():
        raise RuntimeError(f"Refusing to repeat an existing round: {ws}, E{cycle}")
    result = _run_explore_stage(cfg, ws, label=f"E{cycle}", cycle=cycle)
    write(ws / f"round_{cycle}_result.json", result)
    # Upstream explore's exit status alone does not establish successful runs.
    raw = read(ws / "raw_scores.json")
    nodes = raw.get("nodes", [])
    evaluated = [n for n in nodes if n.get("raw", {}).get("evaluated")]
    if cycle == 0 and (result["failed"] or len(evaluated) < 2 or not any(
        (n["raw"].get("combined_score") or 0) > 0 for n in evaluated
    )):
        raise RuntimeError(f"Round did not yield a usable positive trace; inspect {ws}")
    # E1 early stopping and zero scores are valid experimental outcomes.
    if cycle != 0 and not evaluated:
        raise RuntimeError(f"No evaluated candidate in E{cycle}; inspect {ws}")
    return result


def setup(args):
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "CODEX_EXEC_TIMEOUT_SEC": str(args.agent_timeout),
        "DRSI_REPLAY_MAX_WORKERS": "1",
        "DRSI_HISTORY_SEED_MODE": "copy",
        "DRSI_HISTORY_ARCHIVE_FILTER": "all",
        "DRSI_EVAL_PYTHON": sys.executable,
    })
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "shared E0 -> L0 -> adaptive E1 versus fixed E1; no unused L1",
        "max_width": 2, "max_refine": 1, "max_steps_per_round": 4,
        "max_workers": 1, "policy_rewrites": 1, "max_agent_episodes": 13,
        "agent_timeout_sec": args.agent_timeout,
        "coding_agent": getattr(args, "agent", "codex"), "model_override": os.environ.get("CODEX_MODEL"),
        "codex_extra_args": os.environ.get("CODEX_EXTRA_ARGS", ""),
        "model_note": "No override means the existing Codex CLI default; inspect logs for identity.",
        "task_hashes": task_hashes(),
        "upstream_commit": "8ec35bd2538220517ce1bfca4282db21b8c518db",
        "budget_note": "Episode/evaluation caps are not equal tokens or compute. Adaptive also pays for L0.",
        "heldout_note": "Never supplied as feedback; local source is readable, not a secure hidden test.",
    }
    plan_path = dest / "plan.json"
    if plan_path.exists():
        old = read(plan_path)
        if old["task_hashes"] != plan["task_hashes"] or old["agent_timeout_sec"] != args.agent_timeout:
            raise RuntimeError("Task or timeout changed after trial was frozen; use a new output directory")
        for key in ("coding_agent", "model_override", "codex_extra_args"):
            if old.get(key, "" if key == "codex_extra_args" else None) != plan[key]:
                raise RuntimeError(f"{key} changed after trial was frozen; use a new output directory")
    else:
        write(plan_path, plan)
    config = ExploreConfig(problem=TASK / "problem.txt", evaluator=TASK / "evaluator.py",
                           init_program=TASK / "init_program.py", eval_timeout_sec=30,
                           eval_python=sys.executable)
    cfg = PipelineConfig(
        workspace=dest / "e0", agent=getattr(args, "agent", "codex"), cycles=1,
        discover_rounds=1, discover_retry_rounds=0, discover_max_retry_batches=0,
        evaluate_incumbent=True, eval_timeout_sec=120, resume=False, stream=False,
        archive_each_cycle=True, update_baseline_champion=False,
        explore_max_steps=4, explore_max_workers=1, explore_max_width=2,
        explore_max_refine=1, explore_config=config, direction_mode="off",
    )
    return dest, cfg


def phase_e0(dest, cfg):
    ws = dest / "e0"
    _bootstrap_workspace(ws, cfg.explore_config)
    checked_round(cfg, ws, 0)
    for name in ("adaptive", "fixed"):
        target = dest / name
        if target.exists():
            raise RuntimeError(f"Refusing to replace {target}")
        shutil.copytree(ws, target)


def phase_learn(dest, cfg):
    ws = dest / "adaptive"
    if not (ws / "raw_scores.json").exists() or (ws / "learning").exists():
        raise RuntimeError("Learning needs the fresh shared E0 copy; will not repeat a paid phase")
    result = run_cycle(replace(cfg, workspace=ws), cycle=0, start_stage="learning")
    write(dest / "learning_result.json", result)
    if not (result.get("discover") or {}).get("best_round"):
        raise RuntimeError("No replay-qualified policy: inspect learning logs")


def phase_e1(dest, cfg):
    if not (dest / "learning_result.json").exists():
        raise RuntimeError("Complete L0 first")
    for name in ("adaptive", "fixed"):
        ws = dest / name
        if (ws / "history" / "iter001").exists():
            continue
        if name == "fixed":
            if (ws / "policy" / "method.py").exists():
                raise RuntimeError("Fixed control must not have a promoted policy")
            if (ws / "raw_scores.json").exists() and not (ws / "history" / "iter000").exists():
                _archive_and_reset_workspace(ws, cycle=0)
        checked_round(replace(cfg, workspace=ws), ws, 1)
        _archive_and_reset_workspace(ws, cycle=1)


def aggregate_usage(directories):
    # Exact round/job roots only: copied E0 histories must never count again.
    rows = []
    for directory in directories:
        paths = list(directory.rglob("agent_log/codex*/run.json"))
        paths.extend(directory.glob("logs/r*/run.json"))
        for path in sorted(set(paths)):
            # Seeded history inside each branch is another copy, not a new run.
            relative = path.relative_to(directory)
            if "history" in relative.parts or "baseline" in relative.parts:
                continue
            row = read(path)
            row["log"] = str(path)
            usage = path.with_name("usage.json")
            row["usage"] = read(usage) if usage.exists() else {"available": False}
            rows.append(row)
    totals = {}
    for row in rows:
        for key, value in (row["usage"].get("total") or {}).items():
            totals[key] = totals.get(key, 0) + value
    return {"episodes": len(rows), "usage_available": sum(bool(r["usage"].get("available")) for r in rows),
            "reported_token_totals": totals, "duration_sec": sum(r["duration_sec"] for r in rows), "runs": rows}


def phase_report(dest, cfg):
    if read(dest / "plan.json")["task_hashes"] != task_hashes():
        raise RuntimeError("Task files changed during the trial")
    spec = importlib.util.spec_from_file_location("blender_layout_evaluator", TASK / "evaluator.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    report = {"plan": read(dest / "plan.json"), "results": {}}
    if (dest / "warning_recovery.json").exists():
        report["engineering_recovery"] = read(dest / "warning_recovery.json")
    if (dest / "arm_concurrency.json").exists():
        report["arm_concurrency"] = read(dest / "arm_concurrency.json")
    candidates = {"initial": TASK / "init_program.py"}
    for name, directory in (("e0", dest / "e0"),
                            ("adaptive_e1", dest / "adaptive" / "history" / "iter001"),
                            ("fixed_e1", dest / "fixed" / "history" / "iter001")):
        # Select solely on dev score, no heldout feedback. Include zero-score
        # candidates rather than silently dropping them as find_champion does.
        scored = []
        for score in sorted(directory.glob("explore_b*/attempt_*/eval/score.json")):
            program = score.parent.parent / cfg.explore_config.eval_program
            if program.exists():
                scored.append((float(read(score).get("combined_score", 0)), str(program), program))
        if not scored:
            raise RuntimeError(f"No scored candidates in {directory}")
        candidates[name] = max(scored)[2]
    for name, program in candidates.items():
        report["results"][name] = {"program": str(program), "sha256": sha(program),
                                  "dev": evaluator.evaluate_split(program, "dev"),
                                  "heldout": evaluator.evaluate_split(program, "heldout")}
    selected = read(dest / "learning_result.json")["discover"]["best_round"]
    e0 = read(dest / "e0" / "live_cycle_manifest.json")
    e1 = read(dest / "adaptive" / "history" / "iter001" / "live_cycle_manifest.json")
    if selected.get("incumbent"):
        match = e1["policy"]["source"] == e0["policy"]["source"] and e1["policy"]["sha256"] == e0["policy"]["sha256"]
    else:
        match = e1["policy"]["sha256"] == selected["method_sha256"]
    report["policy"] = {"selected": selected, "used_in_e1": e1["policy"], "verified": match}
    report["usage"] = {
        "shared_e0": aggregate_usage([dest / "e0"]),
        "adaptive_learning": aggregate_usage([dest / "adaptive" / "learning" / "iter000"]),
        "adaptive_e1": aggregate_usage([dest / "adaptive" / "history" / "iter001"]),
        "fixed_e1": aggregate_usage([dest / "fixed" / "history" / "iter001"]),
    }
    report["rounds"] = {
        "e0": read(dest / "e0" / "round_0_result.json"),
        "adaptive_e1": read(dest / "adaptive" / "round_1_result.json"),
        "fixed_e1": read(dest / "fixed" / "round_1_result.json"),
    }
    report["cumulative_best"] = {
        arm: max(("e0", arm + "_e1"), key=lambda name: report["results"][name]["dev"]["combined_score"])
        for arm in ("adaptive", "fixed")
    }
    report["limitations"] = [
        "Single shared starting trace and one policy rewrite: feasibility pilot, not evidence of RSI gains.",
        "Equal online episode caps do not equal token/compute budgets; policy rewrite adds cost.",
        "Heldout solver performance is not evidence of policy transfer to new task families.",
        "Cuboid constraints and mesh roundtrip do not measure arbitrary asset quality.",
    ]
    write(dest / "report.json", report)
    if not match:
        raise RuntimeError("Selected-to-used policy hash verification failed; see report.json")
    print(json.dumps({k: {s: v[s]["combined_score"] for s in ("dev", "heldout")}
                      for k, v in report["results"].items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "workspace" / "blender-trial")
    parser.add_argument("--phase", choices=("all", "e0", "learn", "e1", "report"), default="all")
    parser.add_argument("--agent-timeout", type=int, default=240)
    parser.add_argument("--agent", choices=("codex", "codex_json"), default="codex")
    args = parser.parse_args()
    if args.agent_timeout <= 0:
        parser.error("--agent-timeout must be positive")
    dest, cfg = setup(args)
    for name, fn in (("e0", phase_e0), ("learn", phase_learn), ("e1", phase_e1), ("report", phase_report)):
        if args.phase in ("all", name):
            print(f"[trial] {name}", flush=True)
            fn(dest, cfg)


if __name__ == "__main__":
    main()
