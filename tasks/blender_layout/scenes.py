"""Frozen procedural benchmark; no assets or external services required."""
from __future__ import annotations

import random

VERSION = "blender-layout-v1"
DEV_SEEDS = (17, 41, 73, 109, 163, 211)
HELDOUT_SEEDS = (1009, 1063, 1129, 1223, 1301, 1427, 1543, 1663)


def make_scene(seed: int, label: str) -> dict:
    rng = random.Random(seed)
    count = rng.choice((3, 4, 5))
    room = [round(rng.uniform(7.0, 9.0), 3), round(rng.uniform(6.0, 8.0), 3), 4.0]
    objects = []
    for i in range(count):
        objects.append({"id": f"base_{i}", "size": [round(rng.uniform(0.8, 1.6), 3),
                       round(rng.uniform(0.8, 1.5), 3), round(rng.uniform(0.5, 1.1), 3)],
                        "support": None})
    # Reserve enough width for the entire left-to-right chain plus margins.
    # Feasibility must follow from construction rather than a lucky seed.
    room[0] = round(max(room[0], sum(obj["size"][0] for obj in objects) + 0.12*(count-1) + 0.80), 3)
    for i in range(2):
        parent = objects[i]
        size = [round(parent["size"][0] * rng.uniform(0.30, 0.48), 3),
                round(parent["size"][1] * rng.uniform(0.30, 0.48), 3),
                round(rng.uniform(0.20, 0.40), 3)]
        objects.append({"id": f"item_{i}", "size": size, "support": parent["id"]})
    # Two items share a support; a centre-only stack heuristic creates a real
    # collision. Their small combined footprints guarantee a feasible packing.
    p = objects[0]
    objects.append({"id": "item_2", "size": [round(p["size"][0]*0.30, 3),
                    round(p["size"][1]*0.30, 3), 0.22], "support": p["id"]})
    order = list(range(count))
    rng.shuffle(order)
    relations = [{"type": "left_of", "a": f"base_{order[i]}",
                  "b": f"base_{order[i+1]}", "gap": 0.12}
                 for i in range(count-1)]
    # Adjacent bases can satisfy both near and left_of by touching with a gap.
    a, b = objects[order[0]], objects[order[1]]
    relations.append({"type": "near", "a": a["id"], "b": b["id"],
                      "max_distance": round((a["size"][0]+b["size"][0])/2+0.50, 3)})
    rng.shuffle(objects)  # Object list order carries no solution information.
    return {"id": label, "room": room, "objects": objects, "relations": relations}


def get_scenes(split: str = "dev") -> list[dict]:
    if split not in ("dev", "heldout"):
        raise ValueError("split must be dev or heldout")
    seeds = DEV_SEEDS if split == "dev" else HELDOUT_SEEDS
    return [make_scene(seed, f"{split}_{i:02d}") for i, seed in enumerate(seeds)]
