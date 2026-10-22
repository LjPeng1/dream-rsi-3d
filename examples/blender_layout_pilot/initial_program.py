"""Initial general heuristic. Only the marked block is evolved."""
# EVOLVE-BLOCK-START
def solve(scene):
    import math
    objects = scene["objects"]
    bases = sorted((o for o in objects if o["support"] is None), key=lambda o: o["id"])
    cols = math.ceil(math.sqrt(len(bases)))
    rows = math.ceil(len(bases) / cols)
    width, depth, _ = scene["room"]
    result = {}
    by_id = {o["id"]: o for o in objects}
    for i, obj in enumerate(bases):
        result[obj["id"]] = [(i % cols + 0.5) * width / cols,
                             (i // cols + 0.5) * depth / rows, obj["size"][2] / 2]
    pending = [o for o in objects if o["support"] is not None]
    while pending:
        ready = [o for o in pending if o["support"] in result]
        if not ready:
            raise ValueError("cyclic or missing support")
        for obj in ready:
            p = result[obj["support"]]
            parent = by_id[obj["support"]]
            result[obj["id"]] = [p[0], p[1], p[2] + (parent["size"][2]+obj["size"][2])/2]
            pending.remove(obj)
    return result
# EVOLVE-BLOCK-END
