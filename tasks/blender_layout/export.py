"""Export analytic placements through real Blender and check a GLB round trip.

Normal Python invocation evaluates a candidate and starts Blender. Blender calls
the same file with --worker; only that branch imports bpy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def export_layout(program, output, blender, split="dev", scene_index=0):
    from evaluator import run_candidate, score_scene, validate_placements
    from scenes import get_scenes
    scenes = get_scenes(split)
    if not 0 <= scene_index < len(scenes):
        raise ValueError("scene_index is outside the selected split")
    scene = scenes[scene_index]
    placements = run_candidate(program, [scene])[0]
    validate_placements(scene, placements)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output / "layout_spec.json"
    spec = {"scene": scene, "placements": placements, "output": str(output),
            "analytic_evaluation": score_scene(scene, placements),
            "program": str(Path(program).resolve())}
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    command = [str(Path(blender).resolve()), "--background", "--factory-startup",
               "--python", str(Path(__file__).resolve()), "--", "--worker", str(spec_path)]
    proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=180)
    (output / "blender.log").write_text(proc.stdout+"\n"+proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"Blender exited {proc.returncode}; see {output / 'blender.log'}")
    report_path = output / "reimport_check.json"
    if not report_path.exists():
        raise RuntimeError("Blender did not produce a round-trip verification report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise RuntimeError(f"GLB round-trip verification failed: {report}")
    for name in ("layout.glb", "layout.png"):
        if not (output / name).is_file() or (output / name).stat().st_size == 0:
            raise RuntimeError(f"Missing or empty Blender artifact: {name}")
    return {"output": str(output), "glb": str(output / "layout.glb"),
            "png": str(output / "layout.png"), "reimport_check": report,
            "analytic_evaluation": spec["analytic_evaluation"]}


def blender_worker(spec_path):
    import bpy
    from mathutils import Vector
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    scene_spec, placements = spec["scene"], spec["placements"]
    output = Path(spec["output"])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    palette = [(0.18, 0.46, 0.70, 1), (0.89, 0.42, 0.20, 1),
               (0.22, 0.63, 0.42, 1), (0.61, 0.32, 0.68, 1), (0.90, 0.68, 0.20, 1)]
    meshes = []
    for i, obj in enumerate(scene_spec["objects"]):
        bpy.ops.mesh.primitive_cube_add(size=1, location=placements[obj["id"]])
        mesh = bpy.context.object
        mesh.name = obj["id"]
        mesh.dimensions = obj["size"]
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        material = bpy.data.materials.new(f"material_{obj['id']}")
        material.diffuse_color = palette[i % len(palette)]
        material.use_nodes = True
        material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = material.diffuse_color
        material.node_tree.nodes.get("Principled BSDF").inputs["Roughness"].default_value = 0.65
        mesh.data.materials.append(material)
        meshes.append(mesh)
    # Export only task objects. The render floor/camera/lights are not benchmark
    # objects and therefore are deliberately excluded from the GLB.
    bpy.ops.object.select_all(action="DESELECT")
    for mesh in meshes:
        mesh.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(output / "layout.glb"), export_format="GLB", use_selection=True)

    width, depth, height = scene_spec["room"]
    bpy.ops.mesh.primitive_plane_add(size=1, location=(width/2, depth/2, -0.015))
    bpy.context.object.name = "render_floor"
    bpy.context.object.scale = (width, depth, 1)
    floor_material = bpy.data.materials.new("render_floor_material")
    floor_material.diffuse_color = (0.72, 0.75, 0.80, 1)
    bpy.context.object.data.materials.append(floor_material)
    target = Vector((width/2, depth/2, 0.5))
    bpy.ops.object.camera_add(location=(width*1.2, -depth*0.8, max(width, depth)*1.05))
    camera = bpy.context.object
    camera.rotation_euler = (target-camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(width, depth)*1.5
    bpy.context.scene.camera = camera
    bpy.ops.object.light_add(type="AREA", location=(width/2, depth/2, height+5))
    bpy.context.object.data.energy = 1500
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = max(width, depth)
    render_scene = bpy.context.scene
    render_scene.render.engine = "CYCLES"
    render_scene.cycles.device = "CPU"
    render_scene.cycles.samples = 8
    render_scene.world.color = (0.25, 0.25, 0.25)
    render_scene.render.resolution_x = 640
    render_scene.render.resolution_y = 480
    render_scene.render.resolution_percentage = 100
    render_scene.render.image_settings.file_format = "PNG"
    render_scene.render.filepath = str(output / "layout.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "layout.blend"))
    bpy.ops.render.render(write_still=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(output / "layout.glb"))
    imported = {o.name: o for o in bpy.context.scene.objects if o.type == "MESH"}
    expected = {o["id"] for o in scene_spec["objects"]}
    checks = []
    for obj in scene_spec["objects"]:
        mesh = imported.get(obj["id"])
        if mesh is None:
            checks.append({"id": obj["id"], "passed": False, "error": "missing mesh"})
            continue
        corners = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
        mins = [min(v[k] for v in corners) for k in range(3)]
        maxs = [max(v[k] for v in corners) for k in range(3)]
        size = [maxs[k]-mins[k] for k in range(3)]
        center = [(maxs[k]+mins[k])/2 for k in range(3)]
        size_error = max(abs(size[k]-obj["size"][k]) for k in range(3))
        center_error = max(abs(center[k]-placements[obj["id"]][k]) for k in range(3))
        checks.append({"id": obj["id"], "passed": max(size_error, center_error) < 2e-5,
                       "size_error": size_error, "center_error": center_error,
                       "vertices": len(mesh.data.vertices), "polygons": len(mesh.data.polygons)})
    report = {"passed": set(imported) == expected and all(c["passed"] for c in checks),
              "blender_version": bpy.app.version_string, "scene_id": scene_spec["id"],
              "expected_objects": len(expected), "imported_objects": len(imported),
              "checks": checks, "verification": "GLB re-imported; world bounding boxes checked against immutable input"}
    (output / "reimport_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError("GLB verification failed")


def main():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--program")
    parser.add_argument("--output")
    parser.add_argument("--blender")
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--scene-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.worker:
        blender_worker(args.worker)
    else:
        if not all((args.program, args.output, args.blender)):
            parser.error("--program, --output and --blender are required")
        print(json.dumps(export_layout(args.program, args.output, args.blender, args.split, args.scene_index), indent=2))


if __name__ == "__main__":
    main()
