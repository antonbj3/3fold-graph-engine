#!/usr/bin/env python3
"""d_llm_federation_analysis.py -- CPU-only analysis pass over d_llm_federation_raw.json. Computes per-view and
fused (federation) scores, AUC (Mann-Whitney form, identical to an agent worktree's l_llm_coherent_fake.py for direct
comparability) with bootstrap CI, capability-floor accuracy, empirical member-correlation (checks the
"decorrelated" claim isn't just asserted), and the C1/C2/C3 verdicts per d_llm_federation_PREREG.md.
"""
import os, sys, json, re, unicodedata
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "d_llm_federation_raw.json")
OUT = os.path.join(HERE, "d_llm_federation_evidence.json")
sys.path.insert(0, HERE)
from llm_federation_data import build_claims

RNG = np.random.default_rng(0)
N_BOOT = 2000

# ---------------------------------------------------------------- text normalization / parsing
def norm_text(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"^(the|city of|republic of)\s+", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def contains_entity(raw_text, entity):
    """Robust (not first-line-exact) match: does the normalized raw text CONTAIN the normalized entity as a
    whole-word phrase? Small models (esp. 135M) rarely front-load a bare answer -- see pilot transcript, e.g.
    'The answer to the question "Which country has Ank...' -- so first-token/first-line matching would
    undercount every model's true competence, not just the small ones. Applied UNIFORMLY to all models/views."""
    nt, ne = norm_text(raw_text), norm_text(entity)
    if not ne:
        return False
    return re.search(r"(?<!\w)" + re.escape(ne) + r"(?!\w)", nt) is not None

def parse_bool_first(s):
    """First True/False token wins (order-sensitive: small models often hedge, e.g. 'True.... not Istanbul' --
    scoring the FIRST assertion matches how a majority-vote / greedy consumer would actually read the output)."""
    n = s.strip().lower()
    m = re.search(r"\b(true|false)\b", n)
    return (m.group(1) == "true") if m else None

# ---------------------------------------------------------------- AUC + bootstrap
def auc_raw(score, label):
    score = np.asarray(score, float); label = np.asarray(label, float)
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # vectorized Mann-Whitney AUC (ties = 0.5), identical formula to an agent worktree's l_llm_coherent_fake.py auc
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)))

def auc_bootstrap_ci(score, label, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    score = np.asarray(score, float); label = np.asarray(label, float)
    pos_idx = np.where(label == 1)[0]; neg_idx = np.where(label == 0)[0]
    point = auc_raw(score, label)
    if point is None or len(pos_idx) < 2 or len(neg_idx) < 2:
        return point, (None, None)
    boots = []
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        a = auc_raw(np.concatenate([score[bp], score[bn]]), np.concatenate([np.ones(len(bp)), np.zeros(len(bn))]))
        if a is not None:
            boots.append(a)
    boots = np.array(boots)
    return point, (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

# ---------------------------------------------------------------- self-test: void-floor sweep on the AUC machinery
def selftest_void_floor():
    """Machine cross-check on the analysis code itself (not the models): a completely uninformative random score
    must yield AUC~0.50 with a CI that CONTAINS 0.50 across repeated draws, and a perfectly separating synthetic
    score must yield AUC=1.00 exactly. If either fails, the AUC/bootstrap code has a bug -- do not trust any
    downstream number until this passes."""
    rng = np.random.default_rng(42)
    label = np.array([1] * 45 + [0] * 45)
    # void floor: pure noise
    fails = 0
    trials = 30
    for t in range(trials):
        rand_score = rng.uniform(size=90)
        pt, (lo, hi) = auc_bootstrap_ci(rand_score, label, n_boot=400, seed=t)
        if not (lo <= 0.50 <= hi):
            fails += 1
    # perfect separation: score = label + tiny noise in the right direction, must give AUC=1.0 exactly (no noise)
    perfect_score = label.astype(float) * 10.0
    perfect_auc = auc_raw(perfect_score, label)
    # exact chance: score independent of label by construction (all-equal score) -> AUC=0.5 exactly (all ties)
    tie_auc = auc_raw(np.ones(90), label)
    ok = (fails <= 2) and (perfect_auc == 1.0) and (abs(tie_auc - 0.5) < 1e-9)  # allow <=2/30 CI misses at 95% nominal
    return dict(void_floor_ci_misses=f"{fails}/{trials}", perfect_separation_auc=perfect_auc,
                all_tied_auc=tie_auc, PASS=ok)

# ---------------------------------------------------------------- view scoring (raw samples -> per-claim "fake score", higher=more-fake)
def score_m1(samples):
    bools = [parse_bool_first(s) for s in samples]
    parsed = [b for b in bools if b is not None]
    if not parsed:
        return None, 0.0
    frac_true = float(np.mean(parsed))
    return 1.0 - frac_true, len(parsed) / len(samples)  # score = "rejection fraction"; higher = judged more false = more fake

def score_crosscheck(samples, target_entity):
    matches = [contains_entity(s, target_entity) for s in samples]
    if not samples:
        return None, 0.0
    consistency = float(np.mean(matches))
    parse_rate = float(np.mean([len(s.strip()) > 0 for s in samples]))
    return 1.0 - consistency, parse_rate  # higher = LESS self-consistent = more fake

def build_row_scores(raw_rows, condition):
    """condition: 'federation' (m1,m2,m3 populated) or 'single' (m1 only, K=15)."""
    out = []
    for r in raw_rows:
        v = r["views"]
        s1, p1 = score_m1(v["m1"])
        row = dict(category=r["category"], item=r["item"], label_fake=r["label_fake"], claim_text=r["claim_text"],
                   s_m1=s1, parse_m1=p1)
        if condition == "federation":
            s2, p2 = score_crosscheck(v["m2"], r["item"])          # M2: does decorr-crosscheck point back to the item (country/element)?
            s3, p3 = score_crosscheck(v["m3"], r["claim_answer"])  # M3: does independent recall match the CLAIMED entity?
            row.update(s_m2=s2, parse_m2=p2, s_m3=s3, parse_m3=p3)
            valid = [x for x in (s1, s2, s3) if x is not None]
            row["s_federation"] = float(np.mean(valid)) if valid else None
        out.append(row)
    return out

# ---------------------------------------------------------------- capability floor
def capability_accuracy(cap_raw):
    fwd = cap_raw["fwd_recall"]; tf = cap_raw["tf_judgment"]
    fwd_correct = [contains_entity(r["raw"], r["ground_truth"]) for r in fwd]
    fwd_parse = [len(r["raw"].strip()) > 0 for r in fwd]
    tf_bools = [parse_bool_first(r["raw"]) for r in tf]
    tf_parse = [b is not None for b in tf_bools]
    tf_correct = []
    for r, b in zip(tf, tf_bools):
        if b is None:
            continue
        want_true = not r["label_fake"]  # true claim -> correct verdict is True; fake claim -> correct verdict is False
        tf_correct.append(b == want_true)
    return dict(
        fwd_recall_accuracy=round(float(np.mean(fwd_correct)), 3), fwd_recall_n=len(fwd),
        fwd_recall_parseable=round(float(np.mean(fwd_parse)), 3),
        tf_judgment_accuracy=round(float(np.mean(tf_correct)), 3) if tf_correct else None,
        tf_judgment_n_parseable=len(tf_correct), tf_judgment_n_total=len(tf),
        tf_judgment_parse_rate=round(float(np.mean(tf_parse)), 3),
    )

# ---------------------------------------------------------------- member correlation (empirical decorrelation check)
def member_correlation(scored_federation_rows):
    m1 = np.array([r["s_m1"] for r in scored_federation_rows if None not in (r["s_m1"], r["s_m2"], r["s_m3"])])
    m2 = np.array([r["s_m2"] for r in scored_federation_rows if None not in (r["s_m1"], r["s_m2"], r["s_m3"])])
    m3 = np.array([r["s_m3"] for r in scored_federation_rows if None not in (r["s_m1"], r["s_m2"], r["s_m3"])])
    if len(m1) < 3:
        return None
    def r(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    return dict(n=len(m1), corr_m1_m2=r(m1, m2), corr_m1_m3=r(m1, m3), corr_m2_m3=r(m2, m3))

# ---------------------------------------------------------------- held-out logistic fusion (secondary robustness check)
def logistic_fusion_heldout(scored_federation_rows, seed=0):
    rows = [r for r in scored_federation_rows if None not in (r["s_m1"], r["s_m2"], r["s_m3"])]
    if len(rows) < 20:
        return None
    X = np.array([[r["s_m1"], r["s_m2"], r["s_m3"]] for r in rows])
    y = np.array([1.0 if r["label_fake"] else 0.0 for r in rows])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    half = len(rows) // 2
    train, test = idx[:half], idx[half:]
    # standardize on TRAIN only, tiny hand-rolled logistic regression (no sklearn dependency assumed)
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-8
    Xtr, Xte = (X[train] - mu) / sd, (X[test] - mu) / sd
    w = np.zeros(3); b = 0.0
    lr = 0.3
    for _ in range(2000):
        z = Xtr @ w + b
        p = 1 / (1 + np.exp(-z))
        grad_w = Xtr.T @ (p - y[train]) / len(train)
        grad_b = np.mean(p - y[train])
        w -= lr * grad_w; b -= lr * grad_b
    z_test = Xte @ w + b
    p_test = 1 / (1 + np.exp(-z_test))
    a = auc_raw(p_test, y[test])
    return dict(auc_heldout_test=round(a, 3) if a is not None else None, n_train=len(train), n_test=len(test),
                weights_m1_m2_m3=[round(float(x), 3) for x in w])

# ---------------------------------------------------------------- main
def main():
    if not os.path.exists(RAW):
        print(f"NO RAW DATA YET at {RAW}"); return
    raw = json.load(open(RAW))
    claims = build_claims()
    n_fake_expected = sum(1 for c in claims if c["label_fake"])
    print(f"pre-registered dataset: n_claims={len(claims)} n_fake={n_fake_expected} n_true={len(claims)-n_fake_expected}")

    selftest = selftest_void_floor()
    print(f"\n=== SELF-TEST: AUC/bootstrap machinery (void-floor + perfect-separation) ===\n{selftest}")
    if not selftest["PASS"]:
        print("!!! SELF-TEST FAILED -- downstream AUC numbers are NOT trustworthy until this is fixed. Aborting analysis.")
        return

    evidence = dict(selftest=selftest, models={})

    for tag, mres in raw.items():
        print(f"\n{'='*70}\n{tag}\n{'='*70}")
        model_ev = {}
        if "federation" in mres:
            fed_rows = build_row_scores(mres["federation"], "federation")
            label = np.array([1 if r["label_fake"] else 0 for r in fed_rows])
            auc_fed, ci_fed = auc_bootstrap_ci([r["s_federation"] for r in fed_rows], label)
            auc_m1, ci_m1 = auc_bootstrap_ci([r["s_m1"] for r in fed_rows], label)
            auc_m2, ci_m2 = auc_bootstrap_ci([r["s_m2"] for r in fed_rows], label)
            auc_m3, ci_m3 = auc_bootstrap_ci([r["s_m3"] for r in fed_rows], label)
            corr = member_correlation(fed_rows)
            heldout = logistic_fusion_heldout(fed_rows)
            model_ev["federation_K5x3"] = dict(
                auc=round(auc_fed, 3) if auc_fed is not None else None, ci95=ci_fed,
                member_auc=dict(m1_direct=(round(auc_m1, 3) if auc_m1 else None, ci_m1),
                                 m2_decorr_crosscheck=(round(auc_m2, 3) if auc_m2 else None, ci_m2),
                                 m3_independent_recall=(round(auc_m3, 3) if auc_m3 else None, ci_m3)),
                member_correlation=corr, heldout_logistic_fusion=heldout,
                parse_rates=dict(m1=round(float(np.mean([r["parse_m1"] for r in fed_rows])), 3),
                                  m2=round(float(np.mean([r["parse_m2"] for r in fed_rows])), 3),
                                  m3=round(float(np.mean([r["parse_m3"] for r in fed_rows])), 3)))
            print(f"  FEDERATION (3 views x K=5=15 total): AUC={model_ev['federation_K5x3']['auc']} CI95={ci_fed}")
            print(f"    members: M1(direct)={round(auc_m1,3) if auc_m1 else None} CI{ci_m1}  "
                  f"M2(decorr)={round(auc_m2,3) if auc_m2 else None} CI{ci_m2}  "
                  f"M3(recall)={round(auc_m3,3) if auc_m3 else None} CI{ci_m3}")
            print(f"    member correlation: {corr}")
            print(f"    held-out logistic fusion (secondary): {heldout}")
        if "single" in mres:
            single_rows = build_row_scores(mres["single"], "single")
            label = np.array([1 if r["label_fake"] else 0 for r in single_rows])
            auc_s, ci_s = auc_bootstrap_ci([r["s_m1"] for r in single_rows], label)
            model_ev["single_M1_K15"] = dict(auc=round(auc_s, 3) if auc_s is not None else None, ci95=ci_s,
                                              parse_rate=round(float(np.mean([r["parse_m1"] for r in single_rows])), 3))
            print(f"  SINGLE-MEMBER compute-matched (M1 x K=15): AUC={model_ev['single_M1_K15']['auc']} CI95={ci_s}")
        if "capability" in mres:
            cap = capability_accuracy(mres["capability"])
            model_ev["capability_floor"] = cap
            print(f"  CAPABILITY FLOOR: fwd_recall_acc={cap['fwd_recall_accuracy']} (parse {cap['fwd_recall_parseable']})  "
                  f"TF_judgment_acc={cap['tf_judgment_accuracy']} (parsed {cap['tf_judgment_n_parseable']}/{cap['tf_judgment_n_total']})")
        evidence["models"][tag] = model_ev

    # ---------------- C1: federation-small vs single-small vs single-big, compute-matched ----------------
    print(f"\n{'='*70}\nVERDICTS\n{'='*70}")
    verdicts = {}
    try:
        auc_A = evidence["models"]["qwen05b"]["federation_K5x3"]["auc"]
        ci_A = evidence["models"]["qwen05b"]["federation_K5x3"]["ci95"]
        auc_B = evidence["models"]["qwen05b"]["single_M1_K15"]["auc"]
        auc_C = evidence["models"]["qwen15b"]["single_M1_K15"]["auc"]
        auc_D = evidence["models"]["qwen15b"]["federation_K5x3"]["auc"]
        margin_B = round(auc_A - auc_B, 3)
        margin_C = round(auc_A - auc_C, 3)
        ci_excludes_half = (ci_A[0] is not None and ci_A[0] > 0.50)
        c1_pass = (margin_B >= 0.05) and (margin_C >= 0.05) and ci_excludes_half
        verdicts["C1_variance_cert"] = dict(
            auc_federation_small=auc_A, ci95=ci_A, auc_single_small_matched=auc_B, auc_single_big_matched=auc_C,
            auc_federation_big_ceiling=auc_D, margin_vs_single_small=margin_B, margin_vs_single_big=margin_C,
            ci_excludes_void_floor=ci_excludes_half,
            verdict="CONFIRMED" if c1_pass else "HONEST-NEGATIVE",
            crux="" if c1_pass else f"margin_vs_single_small={margin_B} margin_vs_single_big={margin_C} (need >=0.05 both) ci_excludes_0.50={ci_excludes_half}")
        print(f"C1 (variance/cert): {verdicts['C1_variance_cert']['verdict']}  "
              f"AUC federation-small={auc_A} CI{ci_A}  single-small={auc_B}  single-big={auc_C}  "
              f"[ceiling federation-big={auc_D}]  margins: vs-single-small={margin_B} vs-single-big={margin_C}")
    except KeyError as e:
        print(f"C1: could not compute, missing {e}")

    # ---------------- C2: capability floor ----------------
    try:
        floors = {k: evidence["models"][k]["capability_floor"] for k in ("smollm135m", "qwen05b", "qwen06b")}
        def worst_gap(k_small, others, metric):
            v = floors[k_small][metric]
            if v is None:
                return None
            gaps = [floors[o][metric] - v for o in others if floors[o][metric] is not None]
            return min(gaps) if gaps else None
        gap_fwd = worst_gap("smollm135m", ["qwen05b", "qwen06b"], "fwd_recall_accuracy")
        gap_tf = worst_gap("smollm135m", ["qwen05b", "qwen06b"], "tf_judgment_accuracy")
        c2_pass = (gap_fwd is not None and gap_fwd >= 0.10) or (gap_tf is not None and gap_tf >= 0.10)
        verdicts["C2_capability_floor"] = dict(floors=floors, min_gap_fwd_recall=gap_fwd, min_gap_tf_judgment=gap_tf,
                                                verdict="CONFIRMED" if c2_pass else "HONEST-NEGATIVE")
        print(f"C2 (capability floor): {verdicts['C2_capability_floor']['verdict']}  "
              f"135M vs min(0.5B,0.6B) gap: fwd_recall={gap_fwd} tf_judgment={gap_tf}")
    except KeyError as e:
        print(f"C2: could not compute, missing {e}")

    # ---------------- C3: federation on the floored model (A2 vs B2) ----------------
    try:
        auc_A2 = evidence["models"]["smollm135m"]["federation_K5x3"]["auc"]
        ci_A2 = evidence["models"]["smollm135m"]["federation_K5x3"]["ci95"]
        auc_B2 = evidence["models"]["smollm135m"]["single_M1_K15"]["auc"]
        gap_tiny = round(auc_A2 - auc_B2, 3) if (auc_A2 is not None and auc_B2 is not None) else None
        floored = verdicts.get("C2_capability_floor", {}).get("verdict") == "CONFIRMED"
        collapse = (gap_tiny is not None and gap_tiny < 0.05) or (ci_A2[0] is not None and ci_A2[0] <= 0.50)
        if floored:
            c3_result = "C3-CONFIRMED (gap collapses on floored base, as predicted)" if collapse else \
                        "C3-REFUTED-STRONGER-CLAIM (federation lifts AUC even on the floored base)"
        else:
            c3_result = "N/A (C2 did not establish a floor to test against)"
        verdicts["C3_bias_variance_separation"] = dict(auc_federation_tiny=auc_A2, ci95=ci_A2,
                                                         auc_single_tiny_matched=auc_B2, gap=gap_tiny, result=c3_result)
        print(f"C3 (bias/variance separation): {c3_result}  AUC federation-135M={auc_A2} CI{ci_A2}  single-135M={auc_B2}  gap={gap_tiny}")
    except KeyError as e:
        print(f"C3: could not compute, missing {e}")

    evidence["verdicts"] = verdicts
    with open(OUT, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"\nwritten -> {OUT}")

if __name__ == "__main__":
    main()
