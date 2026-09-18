#!/usr/bin/env python3
"""E20: read the model's geometry before the collapse to a token. e7 showed that larger models answer from their prior
(0.995 when the text agrees with physics, 0.71 when it contradicts). Question: is that prior in the representation or only
in the output head? For each sentence (same 320 as e7, lens 0) take the hidden state of the last prompt token at every
layer, fit a linear probe on the RULE's labels (polarity_rules, no human labels) with held-out quantity pairs (20 splits),
and compare with the token output on the same splits — overall, on sentences that contradict physics, per model size.
Also: the probe trained on 10 pairs is tested on the other 10 pairs, so quantity-specific knowledge cannot carry it.
Run: JUDGE_MODEL=<dir> JUDGE_TAG=_1p5b E7_DEVICE=cuda python3 e20_probe_before_collapse.py"""
import io, contextlib, importlib, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
with contextlib.redirect_stdout(io.StringIO()):
    e7 = importlib.import_module("e7_typed_judge_measured")
from graph_engine.polarity_rules import asserted_sign
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

items, tok, model, DEV, TAG = e7.items, e7.tok, e7.model, e7.DEV, e7.TAG
truth = np.array([s > 0 for _, _, _, s, _, _ in items]); agree = np.array([a for *_, a, _ in items])
pair = np.repeat(np.arange(20), 16); rule = np.array([asserted_sign(t, x, y) for t, x, y, *_ in items]); assert (rule != 0).all()
prompts = [e7.prompt(t, e7.QUESTIONS[0], x, y, 0) for t, x, y, *_ in items]

@torch.no_grad()
def hidden(prompts, bs=16):
    H = None
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(DEV)
        out = model(**enc, output_hidden_states=True)
        hs = torch.stack([h[:, -1, :].float() for h in out.hidden_states], 1).cpu().numpy()      # batch × layers × d
        H = hs if H is None else np.concatenate([H, hs])
    return H
H = hidden(prompts); L = H.shape[1]
p_out = e7.p_increase(prompts, [0] * len(prompts))
splits = [np.isin(pair, np.random.default_rng(s).choice(20, 10, replace=False)) for s in range(20)]

def probe_acc(layer):
    a, c = [], []
    for tr in splits:
        te = ~tr
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(H[tr, layer], rule[tr] > 0)
        pred = clf.predict(H[te, layer]); a.append((pred == truth[te]).mean()); c.append((pred == truth[te])[~agree[te]].mean())
    return float(np.mean(a)), float(np.std(a)), float(np.mean(c))
per_layer = [probe_acc(l) for l in range(L)]
best = int(np.argmax([p[0] for p in per_layer]))

def nested():
    """Layer chosen on the training pairs only (inner split over pairs), then tested on the held-out pairs."""
    a, c = [], []
    for tr in splits:
        te = ~tr; trp = np.unique(pair[tr]); inner_rng = np.random.default_rng(int(trp.sum()))
        inner = np.isin(pair, inner_rng.choice(trp, 5, replace=False)); fit, val = tr & ~inner, tr & inner
        sc = [(make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(H[fit, l], rule[fit] > 0).predict(H[val, l]) == (rule[val] > 0)).mean() for l in range(L)]
        l = int(np.argmax(sc))
        pred = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(H[tr, l], rule[tr] > 0).predict(H[te, l])
        a.append((pred == truth[te]).mean()); c.append((pred == truth[te])[~agree[te]].mean())
    return {"accuracy": round(float(np.mean(a)), 3), "sd_over_splits": round(float(np.std(a)), 3), "on_contradicting_text": round(float(np.mean(c)), 3)}
probe_nested = nested()
out_acc = float(((p_out > 0.5) == truth).mean()); out_contra = float(((p_out > 0.5) == truth)[~agree].mean())
res = {"model": Path(e7.MODEL).name, "layers": L, "output_token": {"accuracy": round(out_acc, 3), "on_contradicting_text": round(out_contra, 3)},
       "probe_best_layer": best, "probe_best": {"accuracy": round(per_layer[best][0], 3), "sd_over_splits": round(per_layer[best][1], 3),
                                                "on_contradicting_text": round(per_layer[best][2], 3)},
       "probe_nested_layer_choice": probe_nested, "last_layer": {"accuracy": round(per_layer[-1][0], 3), "on_contradicting_text": round(per_layer[-1][2], 3)},
       "probe_by_layer": [round(p[0], 3) for p in per_layer], "probe_contradicting_by_layer": [round(p[2], 3) for p in per_layer],
       "note": "probe fitted on the rule's labels over 10 quantity pairs, tested on the other 10; lens 0 only"}
json.dump(res, open(Path(__file__).parent / f"e20_results{TAG}.json", "w"), indent=1); [print(k, v) for k, v in res.items()]
