#!/usr/bin/env python3
"""HOLD-OUT EXAM — the patent-as-falsifiable-exam proof: predict a WITHHELD claim, never fit it (an agent worktree).

A recovered design is only TRUSTWORTHY if the model PREDICTS data it never saw. Any model can fit its training claims by
adjusting parameters (that is the fit-trap). The proof is: recover x from a TRAIN subset of claims, then PREDICT a
WITHHELD claim with the FIXED forward and check it lands within σ. A wrong-physics model fits the train claims but MISSES
the withheld one — the exam falsifies it. This validates the service `predict_holdout` and demonstrates the discipline.

GATES (null/control each):
 (G0) CORRECT MODEL PASSES — the true forward, recovering from train claims, predicts the withheld claim within σ.
 (G1) WRONG-PHYSICS MODEL FAILS — a model with the wrong functional form for the withheld observable fits the SAME train
      claims equally well (low train residual) but its withheld prediction is off by ≫σ → the exam FALSIFIES it.
 (G2) σ-AWARE — the predicted withheld claim carries a propagated σ; the true observable lies inside ±k·σ_pred for the
      correct model and OUTSIDE it for the wrong one (the exam's decision is σ-calibrated, not an arbitrary threshold).
 (G3) MORE TRAIN ⇒ SHARPER EXAM — over-determining the recovery (more train claims) shrinks σ_pred, making the exam a
      tighter falsifier (the held-out prediction interval narrows).

Run: python3 m.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from graph_engine.inverse_design.persona_design_gi_service import predict_holdout, inverse

bnds = ([0.2, 0.2], [3.0, 3.0])
X_TRUE = np.array([1.4, 0.9])

# TRUE forward: 4 observables of (w,t). The 4th (index 3) is a PRODUCT — the discriminating claim.
def forward_true(x):
    w, t = x
    return np.array([w + t, 2 * w + t, 0.5 * w + 1.5 * t, w * t])

# WRONG-physics model: identical on observables 0,1,2 but a wrong FORM for the withheld observable 3 (sum, not product).
def forward_wrong(x):
    w, t = x
    return np.array([w + t, 2 * w + t, 0.5 * w + 1.5 * t, w + t])


def main():
    print("=" * 98)
    print("HOLD-OUT EXAM — predict a WITHHELD claim (the patent-as-falsifiable-exam proof)")
    print("=" * 98)
    rng = np.random.default_rng(0)
    obs = forward_true(X_TRUE) * (1 + 0.01 * rng.standard_normal(4))   # 4 measured claims, 1% noise
    train, test = [0, 1, 2], [3]

    rT = predict_holdout(forward_true, obs, [1.0, 1.0], bnds, train, test, noise_rel=0.02)
    rW = predict_holdout(forward_wrong, obs, [1.0, 1.0], bnds, train, test, noise_rel=0.02)

    g0 = rT["passed"] and rT["rel_err"][0] < 0.03
    print(f"\n(G0) CORRECT MODEL PASSES — true forward recovers (w,t)=({rT['x_hat'][0]:.2f},{rT['x_hat'][1]:.2f}), predicts withheld claim "
          f"{rT['pred'][0]:.3f} vs observed {rT['obs_test'][0]:.3f} (err {100*rT['rel_err'][0]:.1f}%) → PASS={rT['passed']}: {'PASS' if g0 else 'FAIL'}")

    # both fit the TRAIN claims equally well (low train residual) — fitting proves nothing
    res_T = inverse(forward_true, obs, [1.0, 1.0], bnds, use=train)[2]
    res_W = inverse(forward_wrong, obs, [1.0, 1.0], bnds, use=train)[2]
    g1 = (not rW["passed"]) and rW["rel_err"][0] > 0.10 and res_W < 1e-2
    print(f"\n(G1) WRONG-PHYSICS MODEL FAILS — wrong model fits train equally (resid {res_W:.1e} vs true {res_T:.1e}) but its withheld "
          f"prediction {rW['pred'][0]:.3f} is off by {100*rW['rel_err'][0]:.0f}% → PASS={rW['passed']} (FALSIFIED): {'PASS' if g1 else 'FAIL'}")

    # G2: σ-calibrated decision — true observable inside ±k·σ_pred for correct, outside for wrong
    inside_T = rT["rel_err"][0] < 3 * (rT["sigma_pred"][0] / abs(rT["obs_test"][0]) + 0.02)
    outside_W = rW["rel_err"][0] > 3 * (rW["sigma_pred"][0] / abs(rW["obs_test"][0]) + 0.02)
    g2 = inside_T and outside_W
    print(f"\n(G2) σ-AWARE — σ_pred true {rT['sigma_pred'][0]:.3f} (obs inside ±3σ: {inside_T}) vs wrong {rW['sigma_pred'][0]:.3f} "
          f"(obs outside ±3σ: {outside_W}) — σ-calibrated, not an arbitrary cutoff: {'PASS' if g2 else 'FAIL'}")

    # G3: more train claims → sharper exam (smaller σ_pred). Compare 2-train vs full over-determined train.
    sig_few = predict_holdout(forward_true, obs, [1.0, 1.0], bnds, [0, 1], test, noise_rel=0.02)["sigma_pred"][0]
    sig_many = rT["sigma_pred"][0]
    g3 = sig_many <= sig_few + 1e-9
    print(f"\n(G3) MORE TRAIN ⇒ SHARPER EXAM — σ_pred with 2 train claims {sig_few:.4f} vs 3 (over-determined) {sig_many:.4f} "
          f"(tighter falsifier): {'PASS' if g3 else 'FAIL'}")

    allok = g0 and g1 and g2 and g3
    print("\n" + "=" * 98)
    if allok:
        print("VERDICT: predict_holdout() is the falsifiable-exam proof — the correct model recovers from the train claims and predicts")
        print(f"  the WITHHELD claim to {100*rT['rel_err'][0]:.1f}% (PASS), while a wrong-physics model that fits the SAME train claims just as well misses")
        print(f"  the withheld claim by {100*rW['rel_err'][0]:.0f}% (FALSIFIED) — fitting proves nothing, predicting withheld data proves everything. The")
        print(f"  decision is σ-calibrated and sharpens with over-determination. The verticals' core proof tool: never fit the held-out claim, predict it.")
    else:
        print(f"VERDICT: NOT all pass — G0 {g0} G1 {g1} G2 {g2} G3 {g3}. Fix at SOURCE.")
    print("=" * 98)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
