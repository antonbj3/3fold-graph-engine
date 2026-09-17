#!/usr/bin/env python3
"""E12: one flow, left to right, through EXISTING and new parts.
   text → typed_extraction (new) → pipeline.grade + ABSTAIN gate + kg_update (existing, unchanged) → c_pred per node
        → unlock_value.rank (new; falls back to the existing priority when nothing is uncertain).
SIMULATION. The judge is simulated with a known error structure taken from the e7 measurement of a local model:
answers are deterministic per (paper, field, lens); an option-order bias of ±0.8 log-odds; a shared misreading per
(paper, field) with probability q that flips every lens; independent lens noise. TEST knowledge graph =
pipeline._synthetic_kg(); per held node: 3 independent confirmers, 2 same-lineage copies, 1 refuter, 3 off-topic papers.
Arms: oracle (the true claim, confidence 0.9 — what the pipeline's own mock uses) · one lens, raw confidence ·
four lenses of ONE option order · four BALANCED lenses with N_eff tempering · the same with the pooled
log-odds isotonically calibrated on 400 labelled fields · two such judges with independent misreadings, log-odds added (typed_extraction.agree).
Reported per arm, over 200 seeds: abstain rate, HARMFUL updates (a confirmer / refutation / hole-fill written to the
graph that the true claim would not have produced), correct updates, mean |c_pred − oracle c_pred|."""
import copy, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from graph_engine.paper_graph import pipeline as pp
from graph_engine.typed_extraction import fields_of, extract, agree, to_pipeline_claim, fit_pooled_calibrator
from sklearn.linear_model import LogisticRegression
from graph_engine.lens_pooling import balanced_lenses, error_correlation, logit

STRENGTH, ORDER_BIAS, NOISE = 1.4, 0.8, 0.9
def make_judge(rng, truth, q_misread):
    flip = {f: rng.random() < q_misread for f in truth}; noise = {}
    def judge(text, field, lens):
        key = (field, tuple(sorted(lens.items())))
        if key not in noise: noise[key] = NOISE * rng.standard_normal()
        v = truth[field] ^ flip[field]
        lo = (STRENGTH if v else -STRENGTH) + (ORDER_BIAS if lens["option_order"] == 0 else -ORDER_BIAS) + noise[key]
        return 1 / (1 + np.exp(-lo))
    return judge

def papers_for(kg, rng):
    out = []
    for nid, node in kg.items():
        f = fields_of(node.claim); yes = {x: True for x in f}
        kinds = [("confirm", f"lab{k}") for k in range(3)] + [("confirm", "lab0"), ("confirm", "lab1")]
        if node.kind == "HELD-CLAIM": kinds.append(("refute", "labR"))
        kinds += [("offtopic", f"labX{k}") for k in range(3)]
        for j, (kind, lin) in enumerate(kinds):
            t = dict(yes)
            if kind == "refute":
                for x in f:
                    if x[0] == "polarity": t[x] = False
            if kind == "offtopic":
                t = {x: False for x in f}
            out.append((f"{nid}:{kind}{j}", nid, lin, t))
    return out

def run(seed, q):
    rng = np.random.default_rng(seed); kg0 = pp._synthetic_kg(); plist = papers_for(kg0, rng)
    bal = balanced_lenses({"option_order": 2, "phrasing_a": 2, "phrasing_b": 2})            # 4 lenses, two of each order
    one = [dict(l, option_order=0) for l in bal]
    # lens error correlation from a labelled calibration set of 300 fields (as one would measure it)
    def corr(lenses):
        Y, L = [], []
        for _ in range(300):
            tr = {("atom", "c"): bool(rng.integers(2))}; jd = make_judge(rng, tr, q)
            L.append([logit(np.array([jd("", ("atom", "c"), l)]))[0] for l in lenses]); Y.append(tr[("atom", "c")])
        return error_correlation(np.array(L), np.array(Y))
    Cb = corr(bal)
    def platt(lenses, C):                                   # calibrate the POOLED log-odds on 400 labelled fields
        X, Y = [], []
        for _ in range(400):
            tr = {("atom", "c"): bool(rng.integers(2))}; ex = extract("", [("atom", "c")], make_judge(rng, tr, q), lenses, lens_corr=C)
            X.append(ex["log_odds"][("atom", "c")]); Y.append(tr[("atom", "c")])
        return fit_pooled_calibrator(np.array(X), np.array(Y))
    cal = platt(bal, Cb)
    arms = {"oracle": None, "one_lens_raw": ([bal[0]], None, None, False), "four_lenses_one_order": (one, None, None, False),
            "four_balanced_tempered": (bal, Cb, None, False), "balanced_calibrated": (bal, Cb, cal, False),
            "two_judges_balanced_calibrated": (bal, Cb, cal, True)}
    res = {}
    judges = {pid: make_judge(rng, t, q) for pid, _, _, t in plist}; judges2 = {pid: make_judge(rng, t, q) for pid, _, _, t in plist}
    oracle_ops, oracle_c = {}, {}
    for arm, cfg in arms.items():
        kg = copy.deepcopy(kg0); n_abst = harm = good = 0
        for pid, nid, lin, t in plist:
            node = kg[nid]; f = fields_of(node.claim); paper = pp.Paper(id=pid, title=pid)
            if cfg is None:
                ex = {"mechanism": [k for (kind, k) in f if kind == "atom" and t[(kind, k)]], "polarity": {k: (1 if t[(kind, k)] else -1) for (kind, k) in f if kind == "polarity"},
                      "extraction_confidence": 0.9}
            else:
                ex = extract("", f, judges[pid], cfg[0], lens_corr=cfg[1], pooled_calibrator=cfg[2])
                if cfg[3]: ex = agree(ex, extract("", f, judges2[pid], cfg[0], lens_corr=cfg[1], pooled_calibrator=cfg[2]))
            pn = to_pipeline_claim(ex, paper, lin, template=node.claim)
            if not pn.claim.mechanism: pn.claim.polarity = {}
            g = pp.grade(pn, node); up = pp.kg_update(node, g, pn, pp.Candidate(pid, nid, node.kind, 1.0))
            if cfg is None: oracle_ops[pid] = up.op
            else:
                n_abst += up.op == "NOOP-HUMAN-REVIEW" and oracle_ops[pid] != "NOOP-HUMAN-REVIEW"
                writes = up.op in ("ADD-CONFIRMER", "FLAG-REFUTATION", "CLOSE-HOLE")
                harm += writes and up.op != oracle_ops[pid]; good += writes and up.op == oracle_ops[pid]
        if cfg is None: oracle_c = {k: v.c_pred for k, v in kg.items()}
        res[arm] = dict(abstain=n_abst / len(plist), harmful=harm, correct=good,
                        c_pred_err=float(np.mean([abs(kg[k].c_pred - oracle_c[k]) for k in kg])))
    res["n_papers"] = len(plist); res["oracle_writes"] = sum(o in ("ADD-CONFIRMER", "FLAG-REFUTATION", "CLOSE-HOLE") for o in oracle_ops.values())
    return res

if __name__ == "__main__":
    out = {}
    for q in (0.0, 0.1, 0.2):
        R = [run(s, q) for s in range(200)]
        agg = {a: {m: round(float(np.mean([r[a][m] for r in R])), 3) for m in R[0][a]} for a in R[0] if isinstance(R[0][a], dict)}
        agg["papers_per_run"] = R[0]["n_papers"]; agg["oracle_writes_per_run"] = float(np.mean([r["oracle_writes"] for r in R]))
        out[f"misread_rate={q}"] = agg; print(f"misread_rate={q}  papers={agg['papers_per_run']}  oracle writes={agg['oracle_writes_per_run']:.1f}")
        for a in agg:
            if isinstance(agg[a], dict): print("   ", f"{a:32s}", agg[a])
    json.dump(out, open(Path(__file__).parent / "e12_results.json", "w"), indent=1)
