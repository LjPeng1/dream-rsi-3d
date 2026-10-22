"""Geometry contracts and candidate boundary tests; no LLM/API/GPU required."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

TASK = Path(__file__).resolve().parents[1] / "tasks" / "blender_layout"
sys.path.insert(0, str(TASK))
spec = importlib.util.spec_from_file_location("layout_evaluator_test", TASK / "evaluator.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def simple_scene():
    return {"id": "unit", "room": [5, 5, 5], "objects": [
        {"id": "base", "size": [2, 2, 1], "support": None},
        {"id": "item", "size": [0.5, 0.5, 0.5], "support": "base"}],
        "relations": [{"type": "near", "a": "base", "b": "item", "max_distance": 1}]}


def test_supported_contact_is_not_collision():
    result = evaluator.score_scene(simple_scene(), {"base": [2, 2, 0.5], "item": [2, 2, 1.25]})
    assert result["combined_score"] == 1.0
    assert result["strict_success"]


@pytest.mark.parametrize("bad", [
    {"base": [2, 2, 0.5]},
    {"base": [2, 2, 0.5], "item": [float("nan"), 2, 1.25]},
    {"base": [2, 2, 0.5], "item": [True, 2, 1.25]},
    {"base": [2, 2, 0.5], "item": [6, 2, 1.25]},
    {"base": [2, 2, 0.5], "item": {"position": [2, 2, 1.25], "size": [0, 0, 0]}},
])
def test_invalid_or_cheating_output_rejected(bad):
    result = evaluator.score_scene(simple_scene(), bad)
    assert not result["valid"]
    assert result["combined_score"] == 0


def test_penetration_and_floating_reduce_score():
    result = evaluator.score_scene(simple_scene(), {"base": [2, 2, 0.5], "item": [2, 2, 0.5]})
    assert result["collisions"] == [["base", "item"]]
    assert result["components"]["collision_free"] == 0
    assert result["components"]["support"] == 0.5
    assert not result["strict_success"]


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        json.loads('{"item":[1,1,1],"item":[2,2,2]}', object_pairs_hook=evaluator._unique_pairs)


def test_candidate_cannot_change_host_sizes(tmp_path):
    program = tmp_path / "mutate.py"
    program.write_text('def solve(scene):\n    for o in scene["objects"]: o["size"] = [0,0,0]\n    return {o["id"]:[0,0,0] for o in scene["objects"]}\n')
    scene = simple_scene()
    before = copy.deepcopy(scene)
    result = evaluator.run_candidate(program, [scene])
    assert scene == before
    assert not evaluator.score_scene(scene, result[0])["valid"]


def test_timeout_bounds_candidate_runtime(tmp_path):
    program = tmp_path / "forever.py"
    program.write_text("def solve(scene):\n    while True: pass\n")
    with pytest.raises(subprocess.TimeoutExpired):
        evaluator.run_candidate(program, [simple_scene()], timeout=0.5)


def test_dev_default_and_fixed_separate_holdout():
    result = evaluator.evaluate(TASK / "init_program.py")
    heldout = evaluator.evaluate_split(TASK / "init_program.py", "heldout")
    repeat = evaluator.evaluate(TASK / "init_program.py")
    assert result["split"] == "dev"
    assert len(result["scenes"]) == 6 and len(heldout["scenes"]) == 8
    assert result["scene_sha256"] != heldout["scene_sha256"]
    assert result["combined_score"] == repeat["combined_score"]
    assert 0 < result["combined_score"] < 1
    assert result["valid_scene_rate"] == 1
    assert result["blender_verified"] is False


def test_frozen_instances_have_constructive_feasible_layouts():
    # Evaluator/generator sanity check only: this reference is never provided as
    # an evolved candidate and is not evidence of learned improvement.
    for split in ("dev", "heldout"):
        for scene in evaluator.get_scenes(split):
            objects = {obj["id"]: obj for obj in scene["objects"]}
            chain = [r for r in scene["relations"] if r["type"] == "left_of"]
            next_id = {r["a"]: r["b"] for r in chain}
            first = (set(next_id)-set(next_id.values())).pop()
            order = [first]
            while order[-1] in next_id:
                order.append(next_id[order[-1]])
            placements = {}
            cursor = 0.2
            for name in order:
                obj = objects[name]
                placements[name] = [cursor+obj["size"][0]/2, scene["room"][1]/2, obj["size"][2]/2]
                cursor += obj["size"][0]+0.12
                children = [o for o in objects.values() if o["support"] == name]
                child_cursor = placements[name][0]-obj["size"][0]/2
                for child in children:
                    placements[child["id"]] = [child_cursor+child["size"][0]/2,
                                               placements[name][1], obj["size"][2]+child["size"][2]/2]
                    child_cursor += child["size"][0]
            result = evaluator.score_scene(scene, placements)
            assert result["strict_success"], (scene["id"], result)
            assert result["combined_score"] == 1.0


def test_export_preserves_paths_with_spaces(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("layout_export_test", TASK / "export.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    output = tmp_path / "results with spaces"
    seen = []
    real_run = subprocess.run
    def fake_run(command, **kwargs):
        if "--background" not in command:
            return real_run(command, **kwargs)
        seen.append(command)
        payload = json.loads(Path(command[-1]).read_text())
        assert Path(payload["output"]) == output.resolve()
        (output / "layout.glb").write_bytes(b"test fixture")
        (output / "layout.png").write_bytes(b"test fixture")
        (output / "reimport_check.json").write_text('{"passed":true}')
        return subprocess.CompletedProcess(command, 0, "fixture, not real Blender", "")
    monkeypatch.setattr(exporter.subprocess, "run", fake_run)
    result = exporter.export_layout(TASK / "init_program.py", output, tmp_path / "Blender Path" / "blender.exe")
    assert len(seen) == 1
    assert seen[0][-1] == str(output.resolve() / "layout_spec.json")
    assert Path(result["glb"]).parent == output.resolve()
