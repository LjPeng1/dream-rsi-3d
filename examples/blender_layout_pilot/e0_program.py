"""Initial general heuristic. Only the marked block is evolved."""
# EVOLVE-BLOCK-START
def solve(scene):
    import math
    import random

    objects = scene['objects']
    n = len(objects)
    if not n:
        return {}
    ids = {o['id']: i for i, o in enumerate(objects)}
    size = [[float(v) for v in o['size']] for o in objects]
    room = [float(v) for v in scene['room']]
    parent = [None if o['support'] is None else ids[o['support']] for o in objects]
    children = [[] for _ in objects]
    for i, p in enumerate(parent):
        if p is not None:
            children[p].append(i)
    order = []
    pending = set(range(n))
    z = [0.0] * n
    while pending:
        ready = sorted(i for i in pending if parent[i] is None or parent[i] in order)
        if not ready:
            raise ValueError('cyclic or missing support')
        for i in ready:
            p = parent[i]
            z[i] = size[i][2] / 2 if p is None else z[p] + (size[p][2] + size[i][2]) / 2
            order.append(i)
            pending.remove(i)
    descendants = [[i] for i in range(n)]
    for i in reversed(order):
        for c in children[i]:
            descendants[i].extend(descendants[c])
    eps = 1e-5
    lo = [[size[i][k] / 2 for k in range(2)] for i in range(n)]
    hi = [[room[k] - size[i][k] / 2 for k in range(2)] for i in range(n)]
    relations = scene.get('relations', [])
    left = []
    near = []
    for r in relations:
        a, b = ids[r['a']], ids[r['b']]
        if r['type'] == 'left_of':
            left.append((a, b, (size[a][0] + size[b][0]) / 2 + float(r.get('gap', 0.0))))
        elif r['type'] == 'near':
            near.append((a, b, float(r['max_distance'])))
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)
             if abs(z[a] - z[b]) < (size[a][2] + size[b][2]) / 2 - 1e-9]
    reach = [[False] * n for _ in range(n)]
    for a, b, _ in left:
        reach[a][b] = True
    for k in range(n):
        for a in range(n):
            if reach[a][k]:
                for b in range(n):
                    reach[a][b] = reach[a][b] or reach[k][b]

    def clamp(x):
        for i in range(n):
            for k in range(2):
                x[i][k] = max(lo[i][k], min(hi[i][k], x[i][k]))

    def shift(x, i, k, amount):
        for j in descendants[i]:
            x[j][k] += amount

    def slack(x, i, k, sign):
        members = descendants[i]
        if sign > 0:
            value = min(hi[j][k] - x[j][k] for j in members)
        else:
            value = min(x[j][k] - lo[j][k] for j in members)
        p = parent[i]
        if p is not None:
            cap = max(0.0, (size[p][k] - size[i][k]) / 2)
            value = min(value, cap - sign * (x[i][k] - x[p][k]))
        return max(0.0, value)

    def separate(x, a, b, k, sign, amount):
        sa = slack(x, a, k, -sign)
        sb = slack(x, b, k, sign)
        da = min(amount / 2, sa)
        db = min(amount - da, sb)
        da = min(amount - db, sa)
        residual = amount - da - db
        da += residual / 2
        db += residual / 2
        shift(x, a, k, -sign * da)
        shift(x, b, k, sign * db)

    def contain(x):
        clamp(x)
        for i in order:
            p = parent[i]
            if p is None:
                continue
            for k in range(2):
                cap = max(0.0, (size[p][k] - size[i][k]) / 2)
                inner = max(0.0, cap - eps)
                target = max(x[p][k] - inner, min(x[p][k] + inner, x[i][k]))
                shift(x, i, k, target - x[i][k])
        clamp(x)

    def assess(x):
        supported = 0
        penalty = 0.0
        for i in range(n):
            p = parent[i]
            if p is None:
                supported += 1
                continue
            violation = sum(max(0.0, abs(x[i][k] - x[p][k]) - (size[p][k] - size[i][k]) / 2) for k in range(2))
            supported += violation <= 1e-8
            penalty += violation
        collisions = 0
        for a, b in pairs:
            ox = (size[a][0] + size[b][0]) / 2 - abs(x[a][0] - x[b][0])
            oy = (size[a][1] + size[b][1]) / 2 - abs(x[a][1] - x[b][1])
            if ox > 1e-8 and oy > 1e-8:
                collisions += 1
                penalty += min(ox, oy)
        satisfied = 0
        for a, b, gap in left:
            v = max(0.0, gap - x[b][0] + x[a][0])
            satisfied += v <= 1e-8
            penalty += v
        for a, b, distance in near:
            v = max(0.0, math.hypot(x[a][0] - x[b][0], x[a][1] - x[b][1]) - distance)
            satisfied += v <= 1e-8
            penalty += v
        fraction = satisfied / max(1, len(left) + len(near)) if left or near else 1.0
        feasibility = 0.55 * supported / n + 0.45 * (1 - collisions / max(1, n * (n - 1) / 2))
        return feasibility * (0.6 + 0.4 * fraction), -penalty

    # Lower-bound propagation constructs a compact ordered configuration.
    constraints = [(a, b, d + eps) for a, b, d in left]
    for i, p in enumerate(parent):
        if p is not None:
            d = (size[i][0] - size[p][0]) / 2
            constraints.extend([(p, i, d), (i, p, d)])
    initial_x = [lo[i][0] for i in range(n)]
    for _ in range(n):
        changed = False
        for a, b, d in constraints:
            v = initial_x[a] + d
            if v > initial_x[b] + 1e-12:
                initial_x[b] = v
                changed = True
        if not changed:
            break
    seed = [[initial_x[i], room[1] / 2] for i in range(n)]
    contain(seed)
    best = [p[:] for p in seed]
    best_key = assess(best)
    rng = random.Random(73109)
    for restart in range(9):
        x = [p[:] for p in (seed if restart < 3 else best)]
        if restart:
            for i in order:
                p = parent[i]
                if p is None:
                    if restart % 3 == 2:
                        x[i][0] = rng.uniform(lo[i][0], hi[i][0])
                    x[i][1] = rng.uniform(lo[i][1], hi[i][1])
                else:
                    for k in range(2):
                        cap = max(0.0, (size[p][k] - size[i][k]) / 2)
                        x[i][k] = x[p][k] + rng.uniform(-cap, cap)
            contain(x)
        collision_order = pairs[:]
        edge_order = left[:]
        for iteration in range(400):
            for a, b, d in edge_order:
                violation = d + eps - (x[b][0] - x[a][0])
                if violation > 0:
                    separate(x, a, b, 0, 1, violation)
            for a, b, distance in near:
                dx, dy = x[b][0] - x[a][0], x[b][1] - x[a][1]
                norm = math.hypot(dx, dy)
                target = max(0.0, distance - eps)
                if norm > target and norm > 0:
                    factor = (norm - target) / (2 * norm)
                    for k, delta in enumerate((dx, dy)):
                        shift(x, a, k, delta * factor)
                        shift(x, b, k, -delta * factor)
            contain(x)
            for a, b in collision_order:
                overlaps = [(size[a][k] + size[b][k]) / 2 - abs(x[b][k] - x[a][k]) for k in range(2)]
                if min(overlaps) <= 0:
                    continue
                options = []
                for k in range(2):
                    for sign in (-1, 1):
                        if k == 0 and ((sign < 0 and reach[a][b]) or (sign > 0 and reach[b][a])):
                            continue
                        amount = (size[a][k] + size[b][k]) / 2 + eps - sign * (x[b][k] - x[a][k])
                        available = slack(x, a, k, -sign) + slack(x, b, k, sign)
                        cost = amount + 4 * max(0.0, amount - available)
                        if restart and iteration < 80:
                            cost *= rng.uniform(0.85, 1.15)
                        options.append((cost, k, sign, amount))
                if options:
                    _, k, sign, amount = min(options)
                    separate(x, a, b, k, sign, amount)
            contain(x)
            if iteration % 5 == 0 or iteration == 399:
                key = assess(x)
                if key > best_key:
                    best_key = key
                    best = [p[:] for p in x]
                if best_key[0] >= 1.0 - 1e-12 and best_key[1] >= -1e-8:
                    return {o['id']: [float(best[i][0]), float(best[i][1]), float(z[i])] for i, o in enumerate(objects)}
            if iteration % 40 == 39:
                rng.shuffle(collision_order)
                rng.shuffle(edge_order)
    return {o['id']: [float(best[i][0]), float(best[i][1]), float(z[i])] for i, o in enumerate(objects)}
# EVOLVE-BLOCK-END
