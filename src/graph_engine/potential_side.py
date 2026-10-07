"""Exact, residual-directed nested quotient potentials.

Every output is an original-node Fraction potential with unit port drop.
No reference resistance enters construction. The partition solve is classical
star-mesh elimination, not a new graph solver theorem.
"""
from collections import defaultdict
from fractions import Fraction as F
from numbers import Integral
from time import perf_counter


def quotient_solve(graph, a, b, check=lambda: None):
    """Exact Dirichlet solve on a small positive-conductance quotient."""
    g = {u: dict(ns) for u, ns in graph.items()}
    forest, operations, max_degree = [], 0, 0
    while len(g) > 2:
        check()
        u = min((v for v in g if v not in (a, b)), key=lambda v: (len(g[v]), v))
        ns = list(g[u].items())
        d = sum((c for _, c in ns), F(0))
        assert d > 0
        max_degree = max(max_degree, len(ns))
        forest.append((u, d, ns))
        for i, (v, cv) in enumerate(ns):
            del g[v][u]
            for w, cw in ns[i+1:]:
                check()
                c = g[v].get(w, F(0)) + cv*cw/d
                g[v][w] = c
                g[w][v] = c
                operations += 1
        del g[u]
    phi = {a: F(1), b: F(0)}
    for u, d, ns in reversed(forest):
        check()
        phi[u] = sum((c*phi[v] for v, c in ns), F(0))/d
    return phi, {"multiply_adds": operations, "max_elimination_degree": max_degree,
                 "eliminations": len(forest)}


def residual_quotient_potential(rows, adj, a, b, dist, check=lambda: None,
                                splits=0, cap=64, checkpoint=None, score_mode="diagonal"):
    """Split max r_u²/d_u singleton, update aggregation, reoptimize exactly.

    Last remaining member of an existing class is not split: it is already a
    singleton. The maintained partition is genuinely nested. Scan, selections,
    incremental aggregation and every elimination are charged here.

    Inputs are the validated edge rows, adjacency, and sink distances of one
    connected port component, as prepared by certified_cross_resistance.
    Return (None, receipt) when the initial partition exceeds the class cap;
    the caller can retain its original feasible potential.
    """
    if isinstance(splits, bool) or not isinstance(splits, Integral) or splits < 0:
        raise ValueError("splits must be a nonnegative integer")
    if isinstance(cap, bool) or not isinstance(cap, Integral) or cap < 2:
        raise ValueError("cap must be an integer of at least two")
    if score_mode not in ("diagonal", "schur"):
        raise ValueError("unknown singleton score")
    t0 = perf_counter()
    keys = {u: (0, u) if u in (a, b) else (1, d) for u, d in dist.items()}
    levels = sorted(set(keys.values()))
    check()
    if len(levels) > cap:
        return None, {"quotient_groups": len(levels), "quotient_fallback": True,
                      "singleton_splits": 0, "selected_nodes": []}
    ids = {key: i for i, key in enumerate(levels)}
    groups = {u: ids[key] for u, key in keys.items()}
    members = defaultdict(set)
    for u, group in groups.items():
        members[group].add(u)
    ga, gb = groups[a], groups[b]
    graph = {group: {} for group in members}
    weighted_degree = {u: F(0) for u in dist}

    def change(x, y, c):
        if x == y:
            return
        value = graph[x].get(y, F(0)) + c
        assert value >= 0
        if value:
            graph[x][y] = graph[y][x] = value
        else:
            graph[x].pop(y, None)
            graph[y].pop(x, None)

    touches = 0
    for u, v, c in rows:
        check()
        if u not in dist or u == v:
            continue
        change(groups[u], groups[v], c)
        weighted_degree[u] += c
        weighted_degree[v] += c
        touches += 1
    aggregation_seconds = perf_counter()-t0
    selection_seconds = solve_seconds = incremental_seconds = 0.
    operations = 0
    selected = []
    previous_energy = None
    for step in range(splits+1):
        ts = perf_counter()
        gp, receipt = quotient_solve(graph, ga, gb, check)
        solve_seconds += perf_counter()-ts
        operations += receipt["multiply_adds"]
        energy = F(0)
        for x, ns in graph.items():
            for y, c in ns.items():
                check()
                if x < y:
                    energy += c * (gp[x] - gp[y]) ** 2
        assert energy > 0 and (previous_energy is None or energy <= previous_energy)
        previous_energy = energy
        phi = {}
        for u in adj:
            check()
            phi[u] = gp[groups[u]] if u in dist else F(0)
        details = {"quotient_groups": len(graph), "singleton_splits": step,
                   "quotient_fallback": False,
                   "selected_nodes": list(selected), "energy": str(energy), "score_mode": score_mode,
                   "aggregation_seconds": aggregation_seconds,
                   "incremental_aggregation_seconds": incremental_seconds,
                   "selection_seconds": selection_seconds,
                   "solve_seconds": solve_seconds, "multiply_adds": operations,
                   "edge_touches": touches, "quotient_seconds": perf_counter()-t0,
                   "last_solve": receipt}
        if checkpoint is not None:
            checkpoint(step, phi, details)
        if step == splits or len(graph) == cap:
            break
        ts = perf_counter()
        free = [g for g in graph if g not in (ga, gb)]
        if score_mode == "schur":
            from ._rational_interface import solve_right
            index = {g:i for i,g in enumerate(free)}
            matrix = [[sum(graph[g].values(),F(0)) if g == h else -graph[g].get(h,F(0))
                       for h in free] for g in free]
            identity = [[F(int(i == j)) for j in range(len(free))] for i in range(len(free))]
            inverse = solve_right(matrix,identity)
            check()
        best, score = None, F(-1)
        for u in sorted(dist):
            check()
            if u in (a, b) or len(members[groups[u]]) == 1:
                continue
            r = F(0)
            for v, i, _ in adj[u]:
                check()
                r += rows[i][2] * (phi[u] - phi[v])
            stiffness = weighted_degree[u]
            if score_mode == "schur":
                vector = defaultdict(lambda:F(0))
                if groups[u] in index:
                    vector[index[groups[u]]] += weighted_degree[u]
                for v,i,_ in adj[u]:
                    if groups[v] in index:
                        vector[index[groups[v]]] -= rows[i][2]
                vector = {i:v for i,v in vector.items() if v}
                stiffness -= sum((vi*inverse[i][j]*vj for i,vi in vector.items()
                                   for j,vj in vector.items()),F(0))
                assert stiffness > 0
            value = r*r/stiffness
            if value > score:
                best, score = u, value
        selection_seconds += perf_counter()-ts
        if best is None or score == 0:
            break
        ts = perf_counter()
        old, new = groups[best], max(graph)+1
        graph[new] = {}
        for v, i, _ in adj[best]:
            check()
            c = rows[i][2]
            change(old, groups[v], -c)
            change(new, groups[v], c)
            touches += 1
        members[old].remove(best)
        members[new].add(best)
        groups[best] = new
        selected.append(best)
        incremental_seconds += perf_counter()-ts
    return phi, details
