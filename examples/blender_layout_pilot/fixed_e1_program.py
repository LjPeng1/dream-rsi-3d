"""Initial general heuristic. Only the marked block is evolved."""
# EVOLVE-BLOCK-START
def solve(scene):
    import math

    objects = scene['objects']
    n = len(objects)
    if not n:
        return {}
    ids = [o['id'] for o in objects]
    index = {name: i for i, name in enumerate(ids)}
    size = [tuple(map(float, o['size'])) for o in objects]
    room = tuple(map(float, scene['room']))
    parent = [None if o['support'] is None else index[o['support']] for o in objects]
    z = [None] * n
    pending = set(range(n))
    while pending:
        ready = sorted(i for i in pending if parent[i] is None or z[parent[i]] is not None)
        if not ready:
            raise ValueError('cyclic or missing support')
        for i in ready:
            p = parent[i]
            z[i] = size[i][2]/2 if p is None else z[p] + (size[p][2]+size[i][2])/2
            pending.remove(i)

    left, near = [], []
    for r in scene.get('relations', []):
        a, b = index[r['a']], index[r['b']]
        if r['type'] == 'left_of':
            left.append((a, b, (size[a][0]+size[b][0])/2 + float(r.get('gap', 0))))
        elif r['type'] == 'near':
            near.append((a, b, float(r['max_distance'])))
    pairs = [(a, b) for a in range(n) for b in range(a+1, n)
             if abs(z[a]-z[b]) < (size[a][2]+size[b][2])/2-1e-9]
    options = {(a, b): [(k, u, v, (size[a][k]+size[b][k])/2)
                        for k in range(2) for u, v in ((a, b), (b, a))]
               for a, b in pairs}
    anchor = n
    m = n+1
    neginf = -float('inf')

    def copy_state(state):
        return [[row[:] for row in matrix] for matrix in state]

    def add(state, k, u, v, distance):
        if distance > state[k][u][v] + 1e-12:
            state[k][u][v] = distance
            return True
        return False

    def close(matrix):
        # matrix[u][v] is a lower bound on coordinate[v]-coordinate[u].
        for k in range(m):
            rowk = matrix[k]
            for u in range(m):
                row = matrix[u]
                base = row[k]
                if base == neginf:
                    continue
                for v in range(m):
                    candidate = base + rowk[v]
                    if candidate > row[v]:
                        row[v] = candidate
            if any(matrix[i][i] > 1e-9 for i in range(m)):
                return False
        for i in range(m):
            matrix[i][i] = 0.0
        return True

    basic = [[[0.0 if u == v else neginf for v in range(m)]
              for u in range(m)] for _ in range(2)]
    for k in range(2):
        for i in range(n):
            add(basic, k, anchor, i, size[i][k]/2)
            add(basic, k, i, anchor, size[i][k]/2-room[k])
            p = parent[i]
            if p is not None:
                d = (size[i][k]-size[p][k])/2
                add(basic, k, p, i, d)
                add(basic, k, i, p, d)

    def contract(state, disks):
        for _ in range(12):
            if not all(close(matrix) for matrix in state):
                return None
            changed = False
            for a, b, radius in disks:
                if radius < 0:
                    return None
                minimum = [max(0.0, state[k][a][b], state[k][b][a]) for k in range(2)]
                if math.hypot(*minimum) > radius+1e-9:
                    return None
                for k in range(2):
                    cap = math.sqrt(max(0.0, radius*radius-minimum[1-k]**2))
                    changed = add(state, k, a, b, -cap) or changed
                    changed = add(state, k, b, a, -cap) or changed
            for pair in pairs:
                opts = options[pair]
                if any(state[k][u][v] >= d-1e-10 for k, u, v, d in opts):
                    continue
                possible = [e for e in opts if state[e[0]][e[2]][e[1]] + e[3] <= 1e-9]
                if not possible:
                    return None
                if len(possible) == 1:
                    changed = add(state, *possible[0]) or changed
            if not changed:
                return state
        if not all(close(matrix) for matrix in state):
            return None
        return state

    def clamp(x):
        return [[max(size[i][k]/2, min(room[k]-size[i][k]/2, x[i][k]))
                 if size[i][k] <= room[k] else room[k]/2
                 for k in range(2)] for i in range(n)]

    def assess(x):
        supported = 0
        residual = 0.0
        for i, p in enumerate(parent):
            if p is None:
                supported += 1
            else:
                err = max(abs(x[i][k]-x[p][k])-(size[p][k]-size[i][k])/2 for k in range(2))
                supported += err <= 1e-8
                residual += max(0.0, err)
        hits = []
        for a, b in pairs:
            penetration = min((size[a][k]+size[b][k])/2-abs(x[a][k]-x[b][k]) for k in range(2))
            if penetration > 1e-8:
                hits.append((penetration, a, b))
                residual += penetration
        good = 0
        for a, b, d in left:
            err = max(0.0, d-x[b][0]+x[a][0])
            good += err <= 1e-8
            residual += err
        bad_near = []
        for a, b, radius in near:
            err = max(0.0, math.hypot(x[a][0]-x[b][0], x[a][1]-x[b][1])-radius)
            good += err <= 1e-8
            residual += err
            if err > 1e-8:
                bad_near.append((err, a, b))
        count = len(left)+len(near)
        fraction = good/count if count else 1.0
        feasibility = .55*supported/n + .45*(1-len(hits)/max(1, n*(n-1)/2))
        return (feasibility*(.6+.4*fraction), -residual), hits, bad_near, supported == n and not hits and good == count

    best = clamp([[room[0]/2, room[1]/2] for _ in objects])
    best_key = assess(best)[0]

    def retain(x):
        nonlocal best, best_key
        x = clamp(x)
        result = assess(x)
        if result[0] > best_key:
            best_key = result[0]
            best = [v[:] for v in x]
        return x, result

    def output():
        return {name: [float(best[i][0]), float(best[i][1]), float(z[i])]
                for i, name in enumerate(ids)}

    stages = [(True, True, 260), (True, False, 70), (False, True, 60), (False, False, 50)]
    for use_left, use_near, budget in stages:
        initial = copy_state(basic)
        if use_left:
            for a, b, d in left:
                add(initial, 0, a, b, d)
        disks = near if use_near else []
        stack = [(initial, 0)]
        nodes = 0
        while stack and nodes < budget:
            state, depth = stack.pop()
            nodes += 1
            state = contract(state, disks)
            if state is None:
                continue
            low = [[state[k][anchor][i] for k in range(2)] for i in range(n)]
            high = [[-state[k][i][anchor] for k in range(2)] for i in range(n)]
            local = None
            for tx, ty in ((.5, .5), (0., .5), (1., .5), (.5, 0.), (.5, 1.),
                           (0., 0.), (1., 1.), (0., 1.), (1., 0.)):
                x = [[(1-t)*low[i][k]+t*high[i][k] for k, t in enumerate((tx, ty))]
                     for i in range(n)]
                x, result = retain(x)
                if result[3]:
                    return output()
                if local is None or result[0] > local[1][0]:
                    local = (x, result)
            x, result = local
            if depth >= 64:
                continue
            hits, bad_near = result[1], result[2]
            branches = []
            if hits:
                choices = []
                for penetration, a, b in hits:
                    opts = [e for e in options[a, b]
                            if state[e[0]][e[2]][e[1]]+e[3] <= 1e-9
                            and state[e[0]][e[1]][e[2]] < e[3]-1e-10]
                    if opts:
                        choices.append((len(opts), -penetration, a, b, opts))
                if choices:
                    opts = min(choices, key=lambda item: item[:4])[4]
                    opts.sort(key=lambda e: (max(0.0, e[3]-x[e[2]][e[0]]+x[e[1]][e[0]]), e))
                    branches = opts
            if not branches and use_near:
                # Split relative displacement, preserving common translation freedom.
                candidates = []
                for err, a, b in bad_near:
                    if a == b:
                        continue
                    for k in range(2):
                        lower, upper = state[k][a][b], -state[k][b][a]
                        if upper-lower > 1e-7:
                            candidates.append((upper-lower, err, k, a, b, lower, upper))
                if candidates:
                    _, _, k, a, b, lower, upper = max(candidates)
                    mid = (lower+upper)/2
                    branches = [(k, b, a, -mid), (k, a, b, mid)]
                    if x[b][k]-x[a][k] > mid:
                        branches.reverse()
            for edge in reversed(branches):
                child = copy_state(state)
                if add(child, *edge):
                    stack.append((child, depth+1))
    return output()
# EVOLVE-BLOCK-END
