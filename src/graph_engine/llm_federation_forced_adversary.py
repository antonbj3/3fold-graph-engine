#!/usr/bin/env python3
"""d_llm_federation_forced_adversary.py -- OODA follow-up forced by the borderline C1 result (federation-small vs
single-big margin=0.012, under the pre-registered 0.05 threshold): a marginal-CI comparison of two independently-
bootstrapped AUCs is UNDERPOWERED here because both conditions share the same 90 claims (shared item-difficulty
noise). A PAIRED bootstrap (resample claim indices ONCE per draw, evaluate BOTH conditions on that same resample)
cancels the shared noise and is the properly-powered test -- exactly the "force the adversary to its strongest
fair form" step, not a rescue of a lazy negative. Also runs the degenerate-answer machine check (is a flat 0.500
accuracy real acquiescence-bias or a parsing artifact?) and a category-level (capitals vs elements) breakdown for
generality. Appends all of this to d_llm_federation_evidence.json under "forced_adversary_followups" -- read-only
over the existing raw data, no new GPU calls.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from llm_federation_analysis import build_row_scores, auc_raw, parse_bool_first

RAW = os.path.join(HERE, "d_llm_federation_raw.json")
EVIDENCE = os.path.join(HERE, "d_llm_federation_evidence.json")

def paired_bootstrap_diff(sA, sB, label, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(label == 1)[0]; neg_idx = np.where(label == 0)[0]
    diffs = []
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn]); lab = np.concatenate([np.ones(len(bp)), np.zeros(len(bn))])
        diffs.append(auc_raw(sA[idx], lab) - auc_raw(sB[idx], lab))
    diffs = np.array(diffs)
    point = float(auc_raw(sA, label) - auc_raw(sB, label))
    return dict(point=round(point, 4), ci95=[round(float(np.percentile(diffs, 2.5)), 4),
                round(float(np.percentile(diffs, 97.5)), 4)], p_diff_gt_0=round(float(np.mean(diffs > 0)), 4),
                p_diff_ge_prereg_margin_0p05=round(float(np.mean(diffs >= 0.05)), 4))

def main():
    raw = json.load(open(RAW))
    followups = {}

    # ---- 1. paired bootstrap: the properly-powered version of every federation-vs-X comparison ----
    fed05 = build_row_scores(raw["qwen05b"]["federation"], "federation")
    single05 = build_row_scores(raw["qwen05b"]["single"], "single")
    single15 = build_row_scores(raw["qwen15b"]["single"], "single")
    label = np.array([1 if r["label_fake"] else 0 for r in fed05])
    order_check = ([r["item"] for r in fed05] == [r["item"] for r in single05] == [r["item"] for r in single15])
    sA = np.array([r["s_federation"] for r in fed05])
    sB = np.array([r["s_m1"] for r in single05])
    sC = np.array([r["s_m1"] for r in single15])
    followups["paired_bootstrap_claim_order_verified_identical"] = bool(order_check)
    followups["paired_federation_small_vs_single_BIG_matched"] = paired_bootstrap_diff(sA, sC, label, seed=7)
    followups["paired_federation_small_vs_single_SMALL_matched"] = paired_bootstrap_diff(sA, sB, label, seed=7)

    per_model_gap = {}
    for tag in ("smollm135m", "qwen05b", "qwen15b"):
        fed = build_row_scores(raw[tag]["federation"], "federation")
        single = build_row_scores(raw[tag]["single"], "single")
        lab = np.array([1 if r["label_fake"] else 0 for r in fed])
        per_model_gap[tag] = paired_bootstrap_diff(np.array([r["s_federation"] for r in fed]),
                                                     np.array([r["s_m1"] for r in single]), lab, seed=11)
    followups["paired_federation_vs_single_by_model_monotone_check"] = per_model_gap

    # ---- 2. degenerate-answer machine check (acquiescence bias vs parsing artifact) ----
    degen = {}
    for tag in ("smollm135m", "qwen05b", "qwen15b", "qwen06b"):
        cap = raw.get(tag, {}).get("capability")
        if not cap:
            continue
        tf = cap["tf_judgment"]
        bools = [parse_bool_first(r["raw"]) for r in tf]
        parsed = [b for b in bools if b is not None]
        degen[tag] = dict(frac_True=round(float(np.mean(parsed)), 3) if parsed else None,
                           n_parsed=len(parsed), n_total=len(tf))
    followups["degenerate_always_true_check"] = degen

    # ---- 3. category breakdown (generality across capitals vs elements) ----
    cat_breakdown = {}
    for tag in ("smollm135m", "qwen05b", "qwen15b"):
        fed = build_row_scores(raw[tag]["federation"], "federation")
        single = build_row_scores(raw[tag]["single"], "single")
        cat_breakdown[tag] = {}
        for cat in ("capitals", "elements"):
            fed_c = [r for r in fed if r["category"] == cat]
            single_c = [r for r in single if r["category"] == cat]
            lab = np.array([1 if r["label_fake"] else 0 for r in fed_c])
            aF = auc_raw([r["s_federation"] for r in fed_c], lab)
            aS = auc_raw([r["s_m1"] for r in single_c], lab)
            cat_breakdown[tag][cat] = dict(n=len(fed_c), auc_federation=round(aF, 3), auc_single=round(aS, 3),
                                            gap=round(aF - aS, 3))
    followups["category_breakdown_generality_check"] = cat_breakdown

    ev = json.load(open(EVIDENCE))
    ev["forced_adversary_followups"] = followups
    with open(EVIDENCE, "w") as f:
        json.dump(ev, f, indent=2, default=str)
    print(json.dumps(followups, indent=2))
    print(f"\nappended -> {EVIDENCE}")

if __name__ == "__main__":
    main()
