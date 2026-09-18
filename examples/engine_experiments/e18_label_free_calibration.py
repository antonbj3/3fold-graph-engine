#!/usr/bin/env python3
"""E18: how much of the calibration gain needs labels? Same 320 labelled sentences and 8 measured lenses as e7.
  raw pooled                 sum of log-odds
  batch-calibrated (no labels)  per lens, subtract the mean log-odds over the unlabeled batch (Zhou et al. 2023 Batch Calibration)
  batch-calibrated per form      the same, but the mean is taken per sentence FORM (form known, no labels)
  rule-labelled (no ground truth)  per-form Platt on the symbolic rule's answers (e14)
  ground-truth labelled            per-form Platt on true labels (e7b)
Held-out by quantity pair (20 splits) where a fit is involved."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.lens_pooling import logit, StratifiedPlatt
from graph_engine.polarity_rules import asserted_sign
import importlib, io, contextlib
with contextlib.redirect_stdout(io.StringIO()): e14 = importlib.import_module('e14_two_real_channels')
P = np.load(Path(__file__).parent / "e7_probabilities.npy").astype(float); L = logit(P)
truth = np.tile(np.repeat([True, False], 8), 20); tmpl = np.tile(np.arange(8), 40); pair = np.repeat(np.arange(20), 16)
acc = lambda z, m=slice(None): float(((z > 0) == truth[m]).mean())
out = {"raw_pooled": round(acc(L.sum(1)), 3), "batch_calibrated": round(acc((L - L.mean(0)).sum(1)), 3)}
Lf = L.copy()
for k in range(8): Lf[tmpl == k] -= L[tmpl == k].mean(0)
out["batch_calibrated_per_form"] = round(acc(Lf.sum(1)), 3)
rule = np.array([asserted_sign(t, x, y) for t, x, y, _, _ in e14.items]); a1, a2 = [], []
for seed in range(20):
    rng = np.random.default_rng(seed); tr = np.isin(pair, rng.choice(20, 10, replace=False)); te = ~tr; ans = rule != 0
    c_rule = StratifiedPlatt().fit(P[tr & ans], rule[tr & ans] > 0, tmpl[tr & ans]); c_true = StratifiedPlatt().fit(P[tr], truth[tr], tmpl[tr])
    a1.append(acc(c_rule.transform(P[te], tmpl[te]).sum(1), te)); a2.append(acc(c_true.transform(P[te], tmpl[te]).sum(1), te))
out["rule_labelled_per_form"] = round(float(np.mean(a1)), 3); out["ground_truth_per_form"] = round(float(np.mean(a2)), 3)
json.dump(out, open(Path(__file__).parent / "e18_results.json", "w"), indent=1); [print(k, v) for k, v in out.items()]
