#!/usr/bin/env python3
"""coupling_admission_precheck.py — predict whether a coupled-inverse / composition experiment
carries information, before building it.

Measured result: a coupling lifts sigma_min ONLY if the added channel delivers non-negligible
Fisher information IN THE DEPLOYMENT REGIME, orthogonal to what the base already covers. That is
the admission theorem (Schur innovation) applied to the coupling decision.

Use it BEFORE building a composition: measure each channel's discriminative power in the actual
regime (pass rate on correct vs wrong cases, or posterior contribution), feed it in, get
ADMIT/REJECT plus the expected sigma_min lift.

  from code:
    from coupling_admission_precheck import precheck
    precheck(base_fisher, {'leg_A': fisher_A_in_regime, 'leg_B': fisher_B_in_regime}, tau=0.01)
  or via CLI with a JSON {base:[[...]], channels:{name:[[...]]}, tau:0.01}:
    python3 coupling_admission_precheck.py spec.json
"""
import numpy as np, json, sys


def schur_innovation(F_base, F_channel):
    """The channel's Fisher information projected onto the base's WEAKEST direction = the marginal
    lift it can give. High => the channel covers what the base misses (admit). ~0 => the channel is
    out of regime or redundant (reject)."""
    w, V = np.linalg.eigh(F_base)
    weak_dir = V[:, 0]                      # the base's least-identified direction
    return float(weak_dir @ F_channel @ weak_dir)


def discriminative_to_fisher(pass_rate_pos, pass_rate_neg, direction=None, dim=1):
    """Coarse Fisher proxy from a leg's discriminative power in the regime: the gap between the
    pass rate on correct and wrong cases scales the information the leg actually carries. Chance
    level (pos ~ neg) => ~0 Fisher. `direction` (unit vector) places the leg's information in the
    target space; default = its own axis."""
    # COORDINATE CHOICE: raw (delta p)^2 is the wrong coordinate for a Bernoulli rate -- it
    # under-weights near-edge discrimination (p -> 0/1) where the variance shrinks. The
 # variance-stabilising (native) coordinate is arcsin(sqrt(p)) (constant Fisher information), so
 # the effect measure is Cohen's h^2: bounded, no edge divergence.
    import math
    h = 2*math.asin(math.sqrt(min(max(pass_rate_pos,0.0),1.0))) - 2*math.asin(math.sqrt(min(max(pass_rate_neg,0.0),1.0)))
    d = h ** 2
    if direction is None:
        F = np.zeros((dim, dim)); F[0, 0] = d; return F
    v = np.asarray(direction, float); v = v / (np.linalg.norm(v) + 1e-12)
    return d * np.outer(v, v)


def precheck(F_base, channels, tau=0.01):
    """channels: {name: Fisher matrix in regime}. Returns per channel {innovation, admit} plus the
    expected lift."""
    F_base = np.asarray(F_base, float)
    smin0 = float(np.linalg.eigvalsh(F_base)[0])
    out = {"sigma_min_base": smin0, "tau": tau, "channels": {}, "admitted": []}
    F_acc = F_base.copy()
    # greedy admission: admit channels in innovation order (largest first)
    ranked = sorted(channels.items(), key=lambda kv: -schur_innovation(F_base, np.asarray(kv[1], float)))
    for name, Fc in ranked:
        Fc = np.asarray(Fc, float)
        innov = schur_innovation(F_acc, Fc)
        admit = innov > tau
        smin_before = float(np.linalg.eigvalsh(F_acc)[0])
        if admit:
            F_acc = F_acc + Fc
        smin_after = float(np.linalg.eigvalsh(F_acc)[0])
        out["channels"][name] = {"schur_innovation": round(innov, 6), "admit": admit,
                                 "sigma_min_lift_if_added": round(smin_after - smin_before, 6)}
        if admit:
            out["admitted"].append(name)
    out["sigma_min_final"] = float(np.linalg.eigvalsh(F_acc)[0])
    out["verdict"] = ("COUPLING CARRIES (>=1 channel admitted)" if out["admitted"]
                      else "COUPLING DOES NOT CARRY -- every channel is below threshold in regime "
                           "(do not build; fix the regime first)")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        spec = json.load(open(sys.argv[1]))
        r = precheck(spec["base"], spec["channels"], spec.get("tau", 0.01))
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        # selftest: in-regime channel admitted, out-of-regime channel rejected.
        # The channel must cover the base's ACTUAL weak direction (not an arbitrary axis), otherwise
        # the test only measures that the channel misses where the base is weak.
        rng = np.random.default_rng(1)
        A = rng.standard_normal((3, 4)); F_base = A.T @ A
        w, V = np.linalg.eigh(F_base); weak = V[:, 0]     # the base's true weakest direction
        ch = {"in_regime": discriminative_to_fisher(0.95, 0.20, weak),   # discriminates, covers the weak direction
              "out_of_regime": discriminative_to_fisher(0.60, 0.57, weak)}  # chance level on the same direction
        r = precheck(F_base, ch, tau=0.01)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        assert "in_regime" in r["admitted"] and "out_of_regime" not in r["admitted"], "selftest FAIL"
        print("\nSELFTEST PASS: in-regime channel admitted, out-of-regime (chance level) rejected.")
