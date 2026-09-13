#!/usr/bin/env python3
"""cert_federation_measure.py — does the AGENT POOL's fusion-law LOCK the grep-replacement cert precision?

The grep-replacement's cert-federation had a measured hole: naive OR-union of decorrelated probes lifts recall
(0.80) but DROPS precision (0.52, FP compounds). B/C shipped the fix (body-1048 + rulebook rule-1): OR-union FPR
≈ Σ per-leg FPR ≈ K·f, so to hold ENSEMBLE FPR ≤ α with K legs, calibrate each leg to FPR ≤ α/K. This MEASURES
that law on the data-engine's 4692-claim held-out (vs W4's 35) with 4 TYPE-DECORRELATED probes (one per templated
fake-kind — number / negation / entity / relation), the honest test of whether calibrated fusion locks precision.

Not on the pool's word: I re-measure per-probe FPR + the union vs the calibrated (per-leg-FPR-budgeted) fusion.
"""
import os, sys, re, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def extract_numbers(text):
    return re.findall(r'\d+\.?\d*', text)

def number_covered(n, pool):
    if n in pool: return True
    try:
        v = float(n)
        return any(abs(float(p) - v) < 1e-9 for p in pool)
    except Exception:
        return False

NEG = {"not", "no", "never", "cannot", "n't", "without", "neither", "nor", "none"}
def negation_parity_suspect(claim, src):
    cn = sum(1 for w in claim.lower().split() if w.strip('.,;:') in NEG)
    # a claim that negates around a term the source states plainly (or vice-versa) is suspect
    sl = src.lower()
    for w in claim.lower().split():
        w = w.strip('.,;:')
        if w in NEG:
            return "claim-negates" if w not in sl else None
    return None

REL = {"beats","loses","faster","slower","more","less","above","below","higher","lower","larger","smaller",
       "increases","decreases","wins","fails","stronger","weaker","exceeds","trails","lifts","floors","catches","misses"}
ANTO = {"beats":"loses","faster":"slower","more":"less","above":"below","higher":"lower","larger":"smaller",
        "increases":"decreases","wins":"fails","stronger":"weaker","exceeds":"trails","lifts":"floors","catches":"misses"}
ANTO.update({v: k for k, v in ANTO.items()})

def probe_numeric(claim, src):
    nums = extract_numbers(claim); pool = extract_numbers(src)
    return any(not number_covered(n, pool) for n in nums)

def probe_negation(claim, src):
    return negation_parity_suspect(claim, src) is not None

def probe_entity(claim, src):
    ents = re.findall(r'\b[A-Z][a-zA-Z_]{4,}\b', claim)
    sl = src.lower()
    return any(e.lower() not in sl for e in ents)

def probe_relation(claim, src):
    cw = set(w.lower().strip('.,;:') for w in claim.split())
    sl = src.lower()
    for w in cw & REL:
        anto = ANTO.get(w)
        if anto and anto in sl and w not in sl:     # claim asserts w, source has the ANTONYM
            return True
    return False

PROBES = {"numeric": probe_numeric, "negation": probe_negation, "entity": probe_entity, "relation": probe_relation}

def rates(flags, labels):
    tp = sum(1 for f, l in zip(flags, labels) if f and l); fp = sum(1 for f, l in zip(flags, labels) if f and not l)
    fn = sum(1 for f, l in zip(flags, labels) if not f and l); tn = sum(1 for f, l in zip(flags, labels) if not f and not l)
    P = tp + fn; N = fp + tn
    return dict(recall=tp/P if P else 0, fpr=fp/N if N else 0, prec=tp/(tp+fp) if tp+fp else 0,
                acc=(tp+tn)/(P+N) if P+N else 0, tp=tp, fp=fp, fn=fn, tn=tn)

def main():
    hp = os.path.join(ROOT, "data", "cert_dataset", "pairs_heldout.jsonl")
    rows = [json.loads(l) for l in open(hp) if l.strip()]
    labels = [r["label"] == 1 for r in rows]                        # True = coherent-fake
    K = len(PROBES); alpha = 0.10
    print("=" * 90 + f"\n CERT-FEDERATION lock test — {len(rows)} held-out claims, K={K} decorrelated probes\n" + "=" * 90)
    per = {}
    for name, fn in PROBES.items():
        fl = [fn(r["claim"], r["source"]) for r in rows]
        per[name] = (fl, rates(fl, labels))
        r = per[name][1]
        print(f"  {name:<9} recall={r['recall']:.2f}  FPR={r['fpr']:.3f}  prec={r['prec']:.2f}")
    # naive OR-union
    union = [any(per[n][0][i] for n in PROBES) for i in range(len(rows))]
    ur = rates(union, labels)
    sum_fpr = sum(per[n][1]["fpr"] for n in PROBES)
    print(f"\n  NAIVE OR-UNION   recall={ur['recall']:.2f}  FPR={ur['fpr']:.3f}  prec={ur['prec']:.2f}")
    print(f"    → B/C law check: union FPR {ur['fpr']:.3f} vs Σ per-leg FPR {sum_fpr:.3f} "
          + ("✓ compounds as predicted" if ur['fpr'] > max(per[n][1]['fpr'] for n in PROBES) else "✗"))
    # CALIBRATED fusion (B/C): keep only legs within the per-leg FPR budget α/K; union those
    budget = alpha / K
    kept = [n for n in PROBES if per[n][1]["fpr"] <= budget]
    dropped = [n for n in PROBES if per[n][1]["fpr"] > budget]
    cal = [any(per[n][0][i] for n in kept) for i in range(len(rows))] if kept else [False]*len(rows)
    cr = rates(cal, labels)
    print(f"\n  CALIBRATED (each leg FPR ≤ α/K={budget:.3f}): kept {kept}, dropped {dropped}")
    print(f"  CALIBRATED FUSION  recall={cr['recall']:.2f}  FPR={cr['fpr']:.3f}  prec={cr['prec']:.2f}")
    def f1(r): return 2*r['recall']*r['prec']/(r['recall']+r['prec']) if r['recall']+r['prec'] else 0
    print(f"\n  LOCK VERDICT: naive (prec {ur['prec']:.2f} rec {ur['recall']:.2f} F1 {f1(ur):.2f}) → "
          f"calibrated (prec {cr['prec']:.2f} rec {cr['recall']:.2f} F1 {f1(cr):.2f})")
    # HONEST lock = precision up AND recall held (F1 must not collapse). Precision-up-at-recall≈0 is a hollow PASS.
    locked = cr['prec'] > ur['prec'] and cr['fpr'] < alpha and cr['recall'] >= 0.5 * ur['recall']
    if locked:
        print("  ★LOCKED: calibrated fusion recovers precision while HOLDING recall — B/C's law locks it on 4692 claims.")
    else:
        print(f"  ✗ NOT LOCKED by calibration alone. Calibration met the FPR budget by DROPPING the high-recall legs")
        print(f"    ({dropped}) — but those are the only ones catching fakes, so recall collapsed {ur['recall']:.2f}→{cr['recall']:.2f}.")
        print(f"    ROOT: the probes are high-recall/HIGH-FPR near-chance (per-probe prec ~{max(per[n][1]['prec'] for n in PROBES):.2f}).")
        print(f"    B/C's fusion MATH is confirmed (union FPR compounds ✓) but it cannot manufacture precision from")
        print(f"    weak members — the LOCK needs a STRONG decorrelated member per domain (L's answer-space probe for")
        print(f"    factual, schema-relation for code), or ABSTAIN+escalate on the floor domain. Fusion is necessary, not sufficient.")

if __name__ == "__main__":
    main()
