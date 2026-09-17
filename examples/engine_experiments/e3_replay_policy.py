#!/usr/bin/env python3
"""E3: replay-learned pair ranking on two public dated citation graphs (SNAP cit-HepTh, cit-HepPh).
Label at cutoff T: a not-yet-co-cited, not-linked pair is co-cited by a paper from (T, T+24 months].
Policies are fitted only on episodes whose 24-month future ends at or before the test cutoff."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.replay_policy import FEATURES, ReplayPolicy, build_episode, average_precision, precision_at
from _data import snap_citations

H = 24
PLANS = [dict(test=95, train=[35, 47, 59, 71]), dict(test=107, train=[47, 59, 71, 83])]
graphs = {g: snap_citations(g) for g in ["HepTh", "HepPh"]}
eps = {}
def ep(g, T, seed=0):
    if (g, T, seed) not in eps:
        e, m, _ = graphs[g]; eps[(g, T, seed)] = build_episode(e, m, T, H, seed=seed)
    return eps[(g, T, seed)]
def stack(es): return np.concatenate([x.X for x in es]), np.concatenate([x.y for x in es])
def metrics(y, s): return dict(ap=round(average_precision(y, s), 4), p_at_1000=round(precision_at(y, s, 1000), 4))

out = {}
for g in graphs:
    other = [x for x in graphs if x != g][0]
    for plan in PLANS:
        te = ep(g, plan["test"]); Xtr, ytr = stack([ep(g, T) for T in plan["train"]])
        Xo, yo = stack([ep(other, T) for T in plan["train"]])
        r = {"base_rate": round(float(te.y.mean()), 4), "n_pairs": int(len(te.y))}
        # single features, sign chosen on the training episodes
        for k, f in enumerate(FEATURES):
            sgn = 1 if average_precision(ytr, Xtr[:, k]) >= average_precision(ytr, -Xtr[:, k]) else -1
            r[f"single:{'+' if sgn > 0 else '-'}{f}"] = metrics(te.y, sgn * te.X[:, k])
        r["existing:hole_field(+L⁺ii)"] = metrics(te.y, te.X[:, FEATURES.index("hole")])
        full = ReplayPolicy().fit(Xtr, ytr); r["replay:all"] = metrics(te.y, full.score(te.X)); r["replay:all:weights"] = full.weights()
        for name, drop in {"no_trajectory": ["vel", "straight"], "no_resistance": ["resistance", "hole"],
                           "no_straight": ["straight"], "no_vel": ["vel"], "topology_only": ["vel", "straight", "recency", "age_gap"]}.items():
            p = ReplayPolicy([f for f in FEATURES if f not in drop]).fit(Xtr, ytr); r[f"replay:{name}"] = metrics(te.y, p.score(te.X))
        r["transfer:other_graph_only"] = metrics(te.y, ReplayPolicy().fit(Xo, yo).score(te.X))
        r["transfer:both_graphs"] = metrics(te.y, ReplayPolicy().fit(np.concatenate([Xtr, Xo]), np.concatenate([ytr, yo])).score(te.X))
        # nonlinear check
        from sklearn.ensemble import HistGradientBoostingClassifier
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(Xtr, ytr)
        r["replay:gradient_boosting"] = metrics(te.y, gb.predict_proba(te.X)[:, 1])

        # recursive replay: chunks = (cutoff, community). uniform vs worst-first choice of the next chunks
        chunks = [(T, c) for T in plan["train"] for c in np.unique(ep(g, T).community) if c >= 0]
        def data(sel):
            X, y = [], []
            for T, c in sel:
                e_ = ep(g, T); m = e_.community == c
                idx = np.flatnonzero(m)[:6000]              # a chunk is a bounded replay, not the whole community
                X.append(e_.X[idx]); y.append(e_.y[idx])
            return np.concatenate(X), np.concatenate(y)
        curves = {"uniform": [], "worst_first": [], "best_first": []}
        for seed in range(8):
            rng = np.random.default_rng(seed); start = [chunks[i] for i in rng.choice(len(chunks), 3, replace=False)]
            for mode in curves:
                sel, curve = list(start), []
                for rnd in range(6):
                    X, y = data(sel)
                    if y.sum() < 5: curve.append(np.nan); pol = None
                    else: pol = ReplayPolicy().fit(X, y); curve.append(average_precision(te.y, pol.score(te.X)))
                    rest = [c for c in chunks if c not in sel]
                    if mode == "uniform" or pol is None: add = [rest[i] for i in rng.choice(len(rest), 3, replace=False)]
                    else:
                        sc = []
                        for T, c in rest:
                            Xc, yc = data([(T, c)]); a = average_precision(yc, pol.score(Xc)) / max(yc.mean(), 1e-9) if yc.sum() else np.nan
                            sc.append(a)                     # lift over the chunk's own base rate
                        order = np.argsort(np.nan_to_num(sc, nan=1e9 if mode == "worst_first" else -1e9))
                        add = [rest[i] for i in (order[:3] if mode == "worst_first" else order[::-1][:3])]
                    sel += add
                curves[mode].append(curve)
        r["recursive_replay_test_ap_by_round(3,6,..18 chunks)"] = {m: [round(float(x), 4) for x in np.nanmean(c, 0)] for m, c in curves.items()}
        r["recursive_replay_sd_last_round"] = {m: round(float(np.nanstd(np.array(c)[:, -1])), 4) for m, c in curves.items()}
        out[f"{g}@{plan['test']}"] = r
        print(g, plan["test"]); [print("  ", k, v) for k, v in r.items()]
        json.dump(out, open(Path(__file__).parent / "e3_results.json", "w"), indent=1, ensure_ascii=False)
