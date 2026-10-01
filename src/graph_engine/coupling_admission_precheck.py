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


def source_family_admission(reports, inherited_support=0, threshold=2, *,
                            verification_registry=None, require_registry=False,
                            target_id=None, require_target_binding=False):
    """Coupling annotations can diagnose a cascade but never supply evidence.

    With this rule the final admitted set is invariant under adding/removing any
    coupling edges, and is bounded by direct verified independent-family support.
    """
    try:
        from .claim_federation import source_family_components
    except ImportError:
        from claim_federation import source_family_components
    if (isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 2
            or isinstance(inherited_support, bool) or not isinstance(inherited_support, int)
            or inherited_support < 0):
        return dict(admit=False, reason_codes=["PROVENANCE_MISSING"], reason="invalid support threshold")
    if require_registry or verification_registry is not None:
        if not isinstance(verification_registry, dict):
            return dict(admit=False, reason_codes=["PROVENANCE_MISSING"], reason="external verification registry required")
        resolved = []
        for report in reports:
            sid = report.get("source_id") if isinstance(report, dict) else None
            if not isinstance(sid, str) or sid not in verification_registry:
                return dict(admit=False, reason_codes=["PROVENANCE_MISSING"], reason="unregistered source ID")
            entry = verification_registry[sid]
            if require_target_binding and (not isinstance(target_id, str) or not target_id
                    or not isinstance(entry, dict) or not isinstance(entry.get("supports"), list)
                    or target_id not in entry["supports"]):
                return dict(admit=False, reason_codes=["PROVENANCE_MISSING"],
                            reason="registered source does not support the bound target")
            if isinstance(entry, dict) and (entry.get("derives_from") or entry.get("parents")):
                return dict(admit=False, reason_codes=["PROVENANCE_MISSING"],
                            reason="registry must bind resolved source families; unresolved lineage is not evidence")
            resolved.append(entry)
        reports = resolved
    support = source_family_components(reports)
    if not support["valid"]:
        return dict(admit=False, support=support, reason_codes=support["reason_codes"], reason="family provenance missing")
    direct = support["count"]
    code = ("COUPLING_CASCADE" if direct < threshold <= direct + inherited_support else
            "SOURCE_FAMILY_DUPLICATION" if direct < threshold <= support["verified_reports"] else
            "INSUFFICIENT_SOURCE_FAMILIES")
    return dict(admit=direct >= threshold, direct_families=direct, inherited_support=inherited_support,
                support=support, reason_codes=[] if direct >= threshold else [code],
                reason="direct family support" if direct >= threshold else code)


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


def precheck(F_base, channels, tau=0.01, *, source_reports=None,
             inherited_support=None, require_source_families=False, verification_registry=None,
             channel_targets=None):
    """channels: {name: Fisher matrix in regime}. Returns per channel {innovation, admit} plus the
    expected lift."""
    F_base = np.asarray(F_base, float)
    smin0 = float(np.linalg.eigvalsh(F_base)[0])
    out = {"sigma_min_base": smin0, "tau": tau, "channels": {}, "admitted": []}
    F_acc = F_base.copy()
    support = {}
    if require_source_families or source_reports is not None:
        for name in channels:
            reports = (source_reports or {}).get(name)
            support[name] = (source_family_admission(reports, (inherited_support or {}).get(name, 0),
                            verification_registry=verification_registry, require_registry=require_source_families,
                            target_id=(channel_targets or {}).get(name), require_target_binding=require_source_families)
                             if isinstance(reports, list) else
                             dict(admit=False, reason_codes=["PROVENANCE_MISSING"]))
        out["source_family_support"] = support
    # greedy admission: admit channels in innovation order (largest first)
    ranked = sorted(channels.items(), key=lambda kv: -schur_innovation(F_base, np.asarray(kv[1], float)))
    for name, Fc in ranked:
        Fc = np.asarray(Fc, float)
        innov = schur_innovation(F_acc, Fc)
        admit = innov > tau and (not support or support[name]["admit"])
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
        r = precheck(spec["base"], spec["channels"], spec.get("tau", 0.01),
                     source_reports=spec.get("source_reports"), inherited_support=spec.get("inherited_support"),
                     require_source_families=spec.get("require_source_families", False),
                     verification_registry=spec.get("verification_registry"), channel_targets=spec.get("channel_targets"))
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
