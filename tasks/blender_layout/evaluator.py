"""Independent numeric evaluator for immutable axis-aligned 3D geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
from scenes import VERSION, get_scenes

TOL = 1e-6


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate output key: {key}")
        result[key] = value
    return result


def run_candidate(program_path, scenes, timeout=None):
    """Candidate receives serialized copies, never host geometry objects."""
    timeout = float(timeout if timeout is not None else os.environ.get("EVALUATOR_TIMEOUT_SECONDS", "8"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    # A configurable workspace cache also works in restricted Windows runners
    # whose system TEMP is not writable by spawned Python processes.
    temporary_root = Path(os.environ.get("BLENDER_LAYOUT_TMPDIR", str(TASK_DIR / ".runtime"))).resolve()
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="layout-eval-", dir=temporary_root) as directory:
        candidate = Path(directory) / "candidate.py"
        shutil.copyfile(Path(program_path).resolve(), candidate)
        proc = subprocess.run([sys.executable, "-I", str(TASK_DIR / "worker.py"), str(candidate)],
                              input=json.dumps(scenes), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", cwd=directory, timeout=timeout)
        if proc.returncode:
            raise ValueError(f"candidate exited {proc.returncode}: {proc.stderr[-1500:]}")
        if len(proc.stdout) > 1_000_000:
            raise ValueError("candidate output exceeds 1 MB")
        outputs = json.loads(proc.stdout, object_pairs_hook=_unique_pairs,
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid constant {value}")))
        if not isinstance(outputs, list) or len(outputs) != len(scenes):
            raise ValueError("candidate must return one placement mapping per scene")
        return outputs


def validate_placements(scene, placements):
    if not isinstance(placements, dict):
        raise ValueError("placements must be a mapping")
    expected = {obj["id"] for obj in scene["objects"]}
    if set(placements) != expected:
        raise ValueError("missing or extra object ids")
    for obj in scene["objects"]:
        xyz = placements[obj["id"]]
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            raise ValueError(f"{obj['id']}: expected exactly [x,y,z]")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in xyz):
            raise ValueError(f"{obj['id']}: coordinates must be finite numbers")
        if any(xyz[k]-obj["size"][k]/2 < -TOL or
               xyz[k]+obj["size"][k]/2 > scene["room"][k]+TOL for k in range(3)):
            raise ValueError(f"{obj['id']}: out of room bounds")


def score_scene(scene, placements):
    try:
        validate_placements(scene, placements)
    except (ValueError, TypeError, OverflowError) as exc:
        return {"id": scene["id"], "valid": False, "error": str(exc), "combined_score": 0.0,
                "strict_success": False, "components": {"support": 0.0, "collision_free": 0.0, "relations": 0.0}}
    objects = scene["objects"]
    by_id = {o["id"]: o for o in objects}
    supported = 0
    for obj in objects:
        xyz, size = placements[obj["id"]], obj["size"]
        bottom = xyz[2]-size[2]/2
        if obj["support"] is None:
            supported += abs(bottom) <= TOL
        else:
            parent = by_id[obj["support"]]
            p, ps = placements[parent["id"]], parent["size"]
            fits = all(abs(xyz[k]-p[k])+size[k]/2 <= ps[k]/2+TOL for k in (0, 1))
            supported += fits and abs(bottom-p[2]-ps[2]/2) <= TOL
    collisions = []
    pair_count = len(objects)*(len(objects)-1)//2
    for i, a in enumerate(objects):
        for b in objects[i+1:]:
            pa, pb = placements[a["id"]], placements[b["id"]]
            if all((a["size"][k]+b["size"][k])/2-abs(pa[k]-pb[k]) > TOL for k in range(3)):
                collisions.append([a["id"], b["id"]])
    satisfied = 0
    relation_results = []
    for relation in scene["relations"]:
        a, b = by_id[relation["a"]], by_id[relation["b"]]
        pa, pb = placements[a["id"]], placements[b["id"]]
        if relation["type"] == "left_of":
            passed = pa[0]+a["size"][0]/2+relation["gap"] <= pb[0]-b["size"][0]/2+TOL
        elif relation["type"] == "near":
            passed = math.hypot(pa[0]-pb[0], pa[1]-pb[1]) <= relation["max_distance"]+TOL
        else:
            raise ValueError("unknown benchmark relation")
        satisfied += passed
        relation_results.append({**relation, "satisfied": passed})
    support = supported/len(objects)
    collision_free = 1-len(collisions)/max(pair_count, 1)
    relations = satisfied/max(len(scene["relations"]), 1)
    feasibility = 0.55*support+0.45*collision_free
    return {"id": scene["id"], "valid": True,
            "combined_score": feasibility*(0.6+0.4*relations),
            "strict_success": supported == len(objects) and not collisions and satisfied == len(scene["relations"]),
            "components": {"support": support, "collision_free": collision_free, "relations": relations},
            "collisions": collisions, "relation_results": relation_results}


def evaluate_split(program_path, split="dev"):
    start = time.monotonic()
    scenes = get_scenes(split)
    fingerprint = hashlib.sha256(json.dumps(scenes, sort_keys=True).encode()).hexdigest()
    try:
        outputs = run_candidate(program_path, scenes)
        results = [score_scene(scene, output) for scene, output in zip(scenes, outputs)]
        return {"combined_score": sum(r["combined_score"] for r in results)/len(results),
                "strict_success_rate": sum(r["strict_success"] for r in results)/len(results),
                "valid_scene_rate": sum(r["valid"] for r in results)/len(results),
                "components": {key: sum(r["components"][key] for r in results)/len(results)
                               for key in ("support", "collision_free", "relations")},
                "scenes": results, "split": split, "benchmark_version": VERSION,
                "scene_sha256": fingerprint, "evaluation_seconds": time.monotonic()-start,
                "evaluation_backend": "analytic_aabb", "blender_verified": False}
    except Exception as exc:
        return {"combined_score": 0.0, "strict_success_rate": 0.0, "valid_scene_rate": 0.0,
                "components": {key: 0.0 for key in ("support", "collision_free", "relations")},
                "error": f"{type(exc).__name__}: {exc}", "split": split,
                "benchmark_version": VERSION, "scene_sha256": fingerprint,
                "evaluation_seconds": time.monotonic()-start, "evaluation_backend": "analytic_aabb",
                "blender_verified": False}


def evaluate(program_path):
    return evaluate_split(program_path, "dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    args = parser.parse_args()
    print(json.dumps(evaluate_split(args.program, args.split), indent=2))
