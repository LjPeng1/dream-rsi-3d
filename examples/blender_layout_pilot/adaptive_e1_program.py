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
    size = [tuple(float(v) for v in o['size']) for o in objects]
    room = tuple(float(v) for v in scene['room'])
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
    for rel in scene.get('relations', []):
        a, b = index[rel['a']], index[rel['b']]
        if rel['type'] == 'left_of':
            left.append((a, b, (size[a][0]+size[b][0])/2 + float(rel.get('gap', 0))))
        elif rel['type'] == 'near':
            near.append((a, b, float(rel['max_distance'])))
    pairs = [(a, b) for a in range(n) for b in range(a+1, n)
             if min(z[a]+size[a][2]/2, z[b]+size[b][2]/2)
             > max(z[a]-size[a][2]/2, z[b]-size[b][2]/2) + 1e-9]
    anchor, count = n, n+1
    neginf = -float('inf')
    tol = 1e-9

    def matrix(axis):
        m = [[neginf]*count for _ in range(count)]
        for i in range(count):
            m[i][i] = 0.0
        for i in range(n):
            m[anchor][i] = size[i][axis]/2
            m[i][anchor] = -(room[axis]-size[i][axis]/2)
            p = parent[i]
            if p is not None:
                d = (size[i][axis]-size[p][axis])/2
                m[p][i] = max(m[p][i], d)
                m[i][p] = max(m[i][p], d)
        return m

    def close(m):
        for k in range(count):
            for i in range(count):
                offset = m[i][k]
                if offset == neginf:
                    continue
                row, source = m[i], m[k]
                for j in range(count):
                    value = offset + source[j]
                    if value > row[j]:
                        row[j] = value
            if any(m[i][i] > tol for i in range(count)):
                return False
        return True

    def propagate(ms, disks):
        for _ in range(12):
            if not close(ms[0]) or not close(ms[1]):
                return False
            changed = False
            for a, b, radius in disks:
                if radius < 0:
                    return False
                minimum = [max(0.0, ms[k][a][b], ms[k][b][a]) for k in (0, 1)]
                if math.hypot(*minimum) > radius + tol:
                    return False
                for k in (0, 1):
                    cap = math.sqrt(max(0.0, radius*radius-minimum[1-k]*minimum[1-k]))
                    for u, v in ((a, b), (b, a)):
                        if -cap > ms[k][u][v] + 1e-11:
                            ms[k][u][v] = -cap
                            changed = True
            if not changed:
                return True
        return close(ms[0]) and close(ms[1])

    def extract(ms):
        return [[(ms[k][anchor][i]-ms[k][i][anchor])/2 for k in (0, 1)] for i in range(n)]

    def clamp(x):
        return [[min(max(v[k], size[i][k]/2), room[k]-size[i][k]/2)
                 if size[i][k] <= room[k] else room[k]/2
                 for k in (0, 1)] for i, v in enumerate(x)]

    def assess(x):
        supported, penalty = 0, 0.0
        for i, p in enumerate(parent):
            if p is None:
                supported += 1
            else:
                error = max(abs(x[i][k]-x[p][k])-(size[p][k]-size[i][k])/2 for k in (0, 1))
                supported += error <= tol
                penalty += max(0.0, error)
        hits = []
        for a, b in pairs:
            overlap = [(size[a][k]+size[b][k])/2-abs(x[a][k]-x[b][k]) for k in (0, 1)]
            if min(overlap) > tol:
                hits.append((min(overlap), a, b))
                penalty += min(overlap)
        good = 0
        for a, b, distance in left:
            error = distance-x[b][0]+x[a][0]
            good += error <= tol
            penalty += max(0.0, error)
        for a, b, radius in near:
            error = math.hypot(x[a][0]-x[b][0], x[a][1]-x[b][1])-radius
            good += error <= tol
            penalty += max(0.0, error)
        nr = len(left)+len(near)
        fraction = good/nr if nr else 1.0
        feasibility = .55*supported/n + .45*(1-len(hits)/max(1, n*(n-1)/2))
        return (feasibility*(.6+.4*fraction), -penalty), hits, supported == n and not hits and good == nr

    best = clamp([[room[0]/2, room[1]/2] for _ in objects])
    best_key = assess(best)[0]

    def retain(x):
        nonlocal best, best_key
        candidate = clamp(x)
        key, _, complete = assess(candidate)
        if key > best_key:
            best_key, best = key, [p[:] for p in candidate]
        return complete

    def answer():
        return {name: [float(best[i][0]), float(best[i][1]), float(z[i])] for i, name in enumerate(ids)}

    def mix_schedules(ms, seed, disks):
        # All pool vertices satisfy the closed linear difference system.
        early = [[ms[k][anchor][i] for k in (0, 1)] for i in range(n)]
        late = [[-ms[k][i][anchor] for k in (0, 1)] for i in range(n)]
        pool = []
        for sx in (0, 1):
            for sy in (0, 1):
                pool.append([[(early, late)[sx][i][0], (early, late)[sy][i][1]] for i in range(n)])
        # Pinning q to t gives exact updated earliest/latest bounds in a
        # closed difference system, without another cubic closure.
        involved = sorted({i for a, b, _ in disks for i in (a, b)})
        for q in involved[:16]:
            for k in (0, 1):
                m = ms[k]
                for t in (early[q][k], late[q][k]):
                    vertex = [v[:] for v in seed]
                    for i in range(n):
                        low = max(m[anchor][i], t + m[q][i])
                        high = min(-m[i][anchor], t - m[i][q])
                        vertex[i][k] = (low+high)/2
                    pool.append(vertex)

        def energy(x):
            return sum(max(0.0, math.hypot(x[a][0]-x[b][0], x[a][1]-x[b][1])-r)**2
                       for a, b, r in disks)

        x = [v[:] for v in seed]
        for _ in range(48):
            gradient = [[0.0, 0.0] for _ in range(n)]
            for a, b, radius in disks:
                dx, dy = x[a][0]-x[b][0], x[a][1]-x[b][1]
                length = math.hypot(dx, dy)
                if length > radius and length > 0:
                    factor = 2*(length-radius)/length
                    for k, delta in enumerate((dx, dy)):
                        gradient[a][k] += factor*delta
                        gradient[b][k] -= factor*delta
            vertex = min(pool, key=lambda v: sum(gradient[i][k]*v[i][k] for i in range(n) for k in (0, 1)))
            direction = [[vertex[i][k]-x[i][k] for k in (0, 1)] for i in range(n)]
            slope = sum(gradient[i][k]*direction[i][k] for i in range(n) for k in (0, 1))
            if slope >= -1e-20:
                break
            def at(t):
                return [[x[i][k]+t*direction[i][k] for k in (0, 1)] for i in range(n)]
            low, high = 0.0, 1.0
            for _ in range(28):
                t1, t2 = (2*low+high)/3, (low+2*high)/3
                if energy(at(t1)) <= energy(at(t2)):
                    high = t2
                else:
                    low = t1
            candidates = [x, vertex, at((low+high)/2)]
            updated = min(candidates, key=energy)
            if energy(updated) >= energy(x):
                break
            x = [v[:] for v in updated]
            if retain(x):
                return x
        return x

    for hard_left, hard_near, budget in ((True, True, 180), (True, False, 70),
                                         (False, True, 50), (False, False, 40)):
        ms = [matrix(0), matrix(1)]
        if hard_left:
            for a, b, distance in left:
                ms[0][a][b] = max(ms[0][a][b], distance)
        disks = near if hard_near else []
        stack = [(ms, 0)]
        nodes, mixing_calls = 0, 0
        while stack and nodes < budget:
            current, depth = stack.pop()
            nodes += 1
            if not propagate(current, disks):
                continue
            x = extract(current)
            if retain(x):
                return answer()
            _, hits, _ = assess(x)
            if not hits and disks and mixing_calls < 10:
                mixing_calls += 1
                x = mix_schedules(current, x, disks)
                if retain(x):
                    return answer()
                _, hits, _ = assess(x)
            alternatives = []
            if hits:
                chosen = None
                for overlap, a, b in hits:
                    options = []
                    for k in (0, 1):
                        distance = (size[a][k]+size[b][k])/2
                        for u, v in ((a, b), (b, a)):
                            if current[k][v][u]+distance <= tol:
                                cost = max(0.0, distance-x[v][k]+x[u][k])
                                options.append((cost, k, u, v, distance))
                    priority = (len(options), -overlap, a, b)
                    if chosen is None or priority < chosen[0]:
                        chosen = (priority, options)
                alternatives = chosen[1]
            elif disks:
                violated = [(math.hypot(x[a][0]-x[b][0], x[a][1]-x[b][1])-r, a, b)
                            for a, b, r in disks]
                error, a, b = max(violated, default=(0.0, 0, 0))
                if error > tol:
                    widths = [-current[k][b][a]-current[k][a][b] for k in (0, 1)]
                    k = max(range(2), key=lambda axis: widths[axis])
                    if widths[k] > 1e-8:
                        middle = (current[k][a][b]-current[k][b][a])/2
                        alternatives = [(0.0, k, a, b, middle), (0.0, k, b, a, -middle)]
            if depth >= 4*n*n + 40:
                continue
            alternatives.sort(reverse=True)
            for _, k, a, b, distance in alternatives:
                if distance <= current[k][a][b] + 1e-12:
                    continue
                child = [[row[:] for row in m] for m in current]
                child[k][a][b] = distance
                stack.append((child, depth+1))
    return answer()
# EVOLVE-BLOCK-END
