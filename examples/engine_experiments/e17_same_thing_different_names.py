#!/usr/bin/env python3
"""E17: the same mechanism under different names, found by computation. Eleven named textbook phenomena from six fields, each a
one-state equilibrium model F(x, μ) = 0 in dimensionless form. The signature (limit points, tangency order, β, γ, oddness) is
computed blind to the name; models are grouped by signature distance < 0.1. Ground truth = the normal form (Thom / Arnold):
fold (saddle-node), pitchfork (symmetric), transcritical or none. A field's name for a fold is not a mechanism; the fold is."""
import json
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.mechanism_signature import signature, distance
C = {  # name: (field, F(x, μ), x-range, μ-bracket, normal form)
    "electrostatic pull-in":            ("microsystems",  lambda u, m: m - u * (1 - u) ** 2,            (0.02, 0.60), (-1, 1),     "fold"),
    "thermal explosion (Semenov)":      ("combustion",    lambda t, p: p * np.exp(t) - t,               (0.05, 2.50), (1e-9, 5),  "fold"),
    "snap-through (shallow arch)":      ("structures",    lambda u, P: P - u * (u - 1) * (u - 2),       (0.02, 0.95), (-3, 3),    "fold"),
    "harvested population collapse":    ("ecology",       lambda x, h: x * (1 - x) - h,                 (0.05, 0.95), (-1, 1),    "fold"),
    "jump-to-contact (AFM cantilever)": ("surface physics", lambda z, d: (d - z) - 1 / z ** 2,          (0.6, 3.0),   (0, 10),    "fold"),
    "Euler buckling (column)":          ("structures",    lambda x, m: x * (m - 1 - x * x),             (-0.8, 0.8),  (-3, 5),    "pitchfork"),
    "ferromagnetic ordering (Landau)":  ("magnetism",     lambda s, T: np.tanh(s / T) - s if T > 0 else np.sign(s) - s, (-0.9, 0.9), (0.05, 3), "pitchfork"),
    "laser threshold":                  ("optics",        lambda n, G: (G - 1) * n - n * n,             (0.05, 2.0),  (-1, 5),    "transcritical"),
    "cusp (Zeeman machine axis)":       ("generic",       lambda x, m: m - x ** 3,                      (-0.8, 0.8),  (-2, 2),    "none"),
    "Hooke spring":                     ("mechanics",     lambda x, m: m - 2 * x,                       (0.02, 0.9),  (-5, 5),    "none"),
    "Michaelis–Menten saturation":      ("biochemistry",  lambda x, m: m - x / (1 - x),                 (0.02, 0.9),  (-1, 50),   "none"),
}
names = list(C); S = {n: signature(C[n][1], C[n][2], C[n][3]) for n in names}
D = np.array([[distance(S[a], S[b]) for b in names] for a in names])
groups, seen = [], set()
for a in names:
    if a in seen: continue
    g = [b for b in names if D[names.index(a), names.index(b)] < 0.1]; seen |= set(g); groups.append(g)
out = {"signatures": {n: {"field": C[n][0], "normal_form": C[n][4], "n_limit": S[n].n_limit, "order": None if np.isnan(S[n].order) else round(S[n].order, 2),
                          "gamma": None if np.isnan(S[n].gamma) else round(S[n].gamma, 2), "odd": S[n].odd} for n in names},
       "groups_by_computed_signature": [{"members": g, "fields": sorted({C[n][0] for n in g}), "normal_forms": sorted({C[n][4] for n in g})} for g in groups]}
pure = all(len(g["normal_forms"]) == 1 for g in out["groups_by_computed_signature"] if not (g["normal_forms"] == ["none"] or "transcritical" in g["normal_forms"]))
out["every_group_is_one_normal_form"] = pure
json.dump(out, open(Path(__file__).parent / "e17_results.json", "w"), indent=1, ensure_ascii=False)
for g in out["groups_by_computed_signature"]: print(g["normal_forms"], g["fields"], "←", g["members"])
print("pure:", pure)
