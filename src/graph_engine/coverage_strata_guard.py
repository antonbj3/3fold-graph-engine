"""coverage_strata_guard — enforce the ADVERSARIAL-MINIMUM booking on the COVERAGE-crediting axis of a coherent-fake
admission gate (an agent worktree cert-tooling; numpy-free, zero cross-worktree import — composes via contract like
neff_channel_diversity / motion_validity_cert).

★WHY (the measured, documented-but-ungated residual). The video_admission_cert credits a fake-dimension as
`effective_covered` iff a leg emits `score_effective_ci_lo >= EFF_COVER_MAG` (0.60). That crediting reads a SINGLE
scalar per leg — nothing forces it to be the dim's ADVERSARIAL-MINIMUM sub-mode. So a leg FAVOURABLE on one fake
sub-mode (content-identity on a rigid recycle: ci_lo ~0.87) but BLIND on the dim's adversarial minimum (novel
content-swap: ci_lo ~0.21) can book its favourable scalar and LIFT the dim to "covered" — a fake living in the
adversarial-minimum sub-mode then walks straight through the "covered" dim. This is the coverage meta-mirage
(systemic). The ANALOGOUS hole is already closed on the FAMILY-CORRELATION axis (`family_corr_strata`: require an
`adversarial_minimum` stratum, book the WORST stratum, else basis=measured-unvalidated). This module is the SAME recipe
realised on the COVERAGE axis, so the pool's "book on the adversarial minimum" discipline becomes ENFORCED, not merely
documented. This module is a standalone reference guard and drop-in to wire
into `_fake_coverage`, and it OVER-DETERMINES the coverage-strata guard with an independent implementation.

The recipe (axis-invariant with the family_corr strata, one law two axes):
  - a per-submode calibration dict `{submode_name: score_effective_ci_lo}` MUST contain the key `adversarial_minimum`
    (the confusion-mode sub-mode where the coherent fake lives). If absent -> the calibration is INCOMPLETE ->
    basis="measured-unvalidated" -> NOT credited (nominal, cannot be trusted; mirrors lines 213-216).
  - otherwise book the WORST sub-mode. For a ci_lo (higher = more detection, LOW is the failure) the worst = MIN over
    sub-modes (the conservative booking; mirrors the `max(strata.values)` for a correlation where HIGH is the
    failure). A dim is credited iff the WORST-submode booked ci_lo >= EFF_COVER_MAG.

This closes the mirage BY CONSTRUCTION: a leg blind on the adversarial minimum books a low worst-submode ci_lo and does
NOT lift the dim, while a leg genuinely strong everywhere (worst-submode still >= bar) is still credited.
"""

import math

EFF_COVER_MAG = 0.60          # video_admission_cert.EFF_COVER_MAG (pinned; the contract-liveness sentinel alarms if it moves)
ADV_MIN_KEY = "adversarial_minimum"
MIN_N_FLOOR = 30              # ★sample-floor: below this a per-stratum ci_lo on a proportion/AUC is not a trustworthy
# lower confidence bound (classic normal-approx / CLT threshold). The target is 100
# matched same-class pairs; a documented n_same=20 is "too thin to book anything under
# CI-lo". Callers SHOULD raise this toward 100 for a hard adversarial minimum, and (when
# they hold raw counts) ALSO run the significance_checked_min_power -- n>=floor is a crude
# necessary proxy for power, not a substitute for it.


def book_coverage_magnitude(submode_ci_lo, adv_min_key=ADV_MIN_KEY, submode_n=None, min_n_floor=MIN_N_FLOOR):
    """Book a leg's coverage ci_lo on the adversarial minimum.

    submode_ci_lo: dict {submode_name: score_effective_ci_lo (float in [0,1])}. MUST contain `adversarial_minimum`.
    submode_n: optional dict {submode_name: n_effective_samples behind that ci_lo}. ★SAMPLE-FLOOR / FAIL-CLOSED
: a ci_lo is a lower CONFIDENCE bound
        only if it was computed from an adequately-powered stratum. A starved stratum (e.g.  same-class n=18/20) can
        yield an UNTRUSTWORTHY ci_lo -- a wide/unstable bootstrap 2.5th percentile can land above the bar by chance, or a
        caller may pass a non-widening estimate -- and, because booking takes the MIN over sub-modes, an untrustworthy
        value that participates in (or sets) the booked minimum can spuriously credit the dim. When submode_n is provided,
        EVERY sub-mode must have a finite n >= min_n_floor; any starved OR MISSING n -> ABSTAIN (a coverage guard must
        default pessimistic, never credit on unverifiable power). This ENFORCES the documented "n too thin -> no booking"
        precondition BY CONSTRUCTION instead of relying on the caller's ci_lo to have widened. When submode_n is None the
        behaviour is unchanged (backward-compatible; the ci_lo's own widening is the only protection).
    Returns dict:
      booked_ci_lo: float | None  -- the WORST-submode ci_lo (None when the calibration is incomplete/starved)
      basis: "adversarial-minimum-validated" | "measured-unvalidated[:reason]"
      worst_submode: name of the worst sub-mode (None when incomplete)
      complete: bool -- adversarial_minimum stratum present AND (if submode_n given) every stratum adequately powered
    """
    if not submode_ci_lo or adv_min_key not in submode_ci_lo:
        # incomplete calibration -> cannot certify the adversarial minimum -> NOT trustworthy (nominal).
        return {"booked_ci_lo": None, "basis": "measured-unvalidated", "worst_submode": None, "complete": False}
    # ★KEY-COLLISION GUARD / FAIL-CLOSED (failure-envelope re-sweep follow-up): keys are stringified below, so a
    # mixed-type submode dict (e.g. int 5 and str '5', or 5.0 and '5.0') COLLIDES -- one submode SILENTLY OVERWRITES the
    # other, dropping a stratum from the MIN booking so an unfavourable low ci_lo (0.10) can vanish and the booked minimum
    # (0.72) is spuriously favourable. A collision = ambiguous/lossy input -> abstain rather than book a partial minimum.
    if len({str(k) for k in submode_ci_lo}) != len(submode_ci_lo):
        return {"booked_ci_lo": None, "basis": "measured-unvalidated:stringified-key-collision", "worst_submode": None,
                "complete": False}
    vals = {str(k): float(v) for k, v in submode_ci_lo.items()}
    # ★NULL-SAFETY / FAIL-CLOSED (§4c): a NaN submode ci_lo makes `min(vals, key=vals.get)` SKIP
    # the NaN (all NaN comparisons are False) and book the FAVOURABLE submode -> the exact meta-mirage this guard exists
    # to prevent (a leg blind on its adversarial minimum credited on an easy sub-mode). A degenerate submode = no
    # trustworthy adversarial-minimum measurement -> ABSTAIN (measured-unvalidated), never book a favourable fallback.
    if not all(math.isfinite(v) for v in vals.values()):
        return {"booked_ci_lo": None, "basis": "measured-unvalidated:non-finite-submode", "worst_submode": None,
                "complete": False}
    # ★RANGE-VALIDATION / FAIL-CLOSED (failure-envelope wave): a coverage score_effective ci_lo is a
    # confidence-bounded rate in [0,1]. A supplied ci_lo > 1 (or < 0) is IMPOSSIBLE -> corrupted input; because booking
    # takes MIN, an out-of-range-HIGH value never sets the worst submode and the credit gate `booked_ci_lo >= bar`
    # FAILS OPEN (e.g. adversarial_minimum=1.5 credits the dim). Refuse rather than credit on an invalid magnitude.
    if not all(0.0 <= v <= 1.0 for v in vals.values()):
        return {"booked_ci_lo": None, "basis": "measured-unvalidated:ci_lo-out-of-[0,1]", "worst_submode": None,
                "complete": False}
    # ★SAMPLE-FLOOR / FAIL-CLOSED (real-control-starved finding): when the caller supplies per-stratum n, refuse to
    # book on ANY stratum whose n is missing, non-finite, or below the floor -- its ci_lo cannot be trusted as a bound.
    if submode_n is not None:
        starved = []
        for name in vals:
            nv = submode_n.get(name)
            if nv is None or not math.isfinite(float(nv)) or float(nv) < min_n_floor:
                starved.append(name)
        if starved:
            return {"booked_ci_lo": None, "basis": "measured-unvalidated:starved-stratum", "worst_submode": None,
                    "complete": False, "starved_strata": sorted(starved), "min_n_floor": min_n_floor}
    worst_submode = min(vals, key=vals.get)      # MIN ci_lo = worst detection = the adversarial booking
    return {"booked_ci_lo": vals[worst_submode], "basis": "adversarial-minimum-validated",
            "worst_submode": worst_submode, "complete": True}


def coverage_effective_credited(submode_ci_lo, eff_cover_mag=EFF_COVER_MAG, adv_min_key=ADV_MIN_KEY,
                                submode_n=None, min_n_floor=MIN_N_FLOOR):
    """True iff the dim is EFFECTIVELY covered under the adversarial-minimum strata guard.

    Credited iff the calibration is COMPLETE (has an adversarial_minimum stratum AND, when submode_n is given, every
    stratum is adequately powered) AND the WORST-submode booked ci_lo clears the bar. Incomplete calibration
    (favourable-submode-only) or a starved stratum is NEVER credited -> the meta-mirage and the starved-CI-lo hole are
    both closed.
    """
    b = book_coverage_magnitude(submode_ci_lo, adv_min_key=adv_min_key, submode_n=submode_n, min_n_floor=min_n_floor)
    return {"credited": bool(b["complete"] and b["booked_ci_lo"] is not None and b["booked_ci_lo"] >= eff_cover_mag),
            **b}


def guard_leg_covers(submode_ci_lo, covers, eff_cover_mag=EFF_COVER_MAG, adv_min_key=ADV_MIN_KEY,
                     submode_n=None, min_n_floor=MIN_N_FLOOR):
    """Coverage decision for a leg declaring `covers=[dim,...]` with a per-submode calibration.
    Returns the set of dims this leg may credit as effective_covered under the strata guard (empty if not credited)."""
    c = coverage_effective_credited(submode_ci_lo, eff_cover_mag=eff_cover_mag, adv_min_key=adv_min_key,
                                    submode_n=submode_n, min_n_floor=min_n_floor)
    return set(covers) if c["credited"] else set(), c


def _selftest():
    """Pre-registered gates (see the owning evidence cell). Returns (all_pass, results)."""
    # G1: favourable-submode ONLY (no adversarial_minimum) -> NOT credited (the meta-mirage)
    g1 = coverage_effective_credited({"rigid_recycle": 0.87})
    # G2: adversarial_minimum present but LOW -> worst-submode booking -> NOT credited
    g2 = coverage_effective_credited({"rigid_recycle": 0.87, "adversarial_minimum": 0.21})
    # G3: genuinely strong everywhere -> worst-submode still clears -> credited (null control, no over-rejection)
    g3 = coverage_effective_credited({"easy": 0.95, "adversarial_minimum": 0.72})
    # G4: SAMPLE-FLOOR -- a strong-looking ci_lo whose adversarial_minimum stratum is STARVED (n=18) -> ABSTAIN
    # (real-control-starved: a ci_lo from n<floor is not a trustworthy bound). Without submode_n it would credit.
    g4_nofloor = coverage_effective_credited({"easy": 0.95, "adversarial_minimum": 0.72})
    g4_floor = coverage_effective_credited({"easy": 0.95, "adversarial_minimum": 0.72},
                                           submode_n={"easy": 400, "adversarial_minimum": 18})
    # G5: SAMPLE-FLOOR null control -- same ci_lo but adversarial_minimum ADEQUATELY powered (n=120) -> still credited
    g5 = coverage_effective_credited({"easy": 0.95, "adversarial_minimum": 0.72},
                                     submode_n={"easy": 400, "adversarial_minimum": 120})
    # G6: SAMPLE-FLOOR fail-closed -- a submode with MISSING n (caller omits the thin stratum's count) -> ABSTAIN
    g6 = coverage_effective_credited({"easy": 0.95, "adversarial_minimum": 0.72}, submode_n={"easy": 400})
    # G7: FAILURE-ENVELOPE-WAVE regression -- an out-of-range ci_lo (a rate must be in [0,1]) is IMPOSSIBLE.
    # Because booking takes MIN, a supra-unit adversarial_minimum=1.5 never sets the worst submode and the credit gate
    # FAILS OPEN. Must ABSTAIN on any ci_lo outside [0,1]; a valid 0.9 still credits.
    g7_high = coverage_effective_credited({"adversarial_minimum": 1.5})
    g7_neg = coverage_effective_credited({"easy": 0.95, "adversarial_minimum": -0.2})
    g7_ok = coverage_effective_credited({"adversarial_minimum": 0.9})
    res = {
        "G1_favourable_only_not_credited": (g1["credited"] is False and g1["basis"] == "measured-unvalidated"),
        "G2_advmin_worst_booked_not_credited": (g2["credited"] is False and abs(g2["booked_ci_lo"] - 0.21) < 1e-9
                                                and g2["worst_submode"] == "adversarial_minimum"),
        "G3_genuine_still_credited": (g3["credited"] is True and abs(g3["booked_ci_lo"] - 0.72) < 1e-9),
        "G4_sample_floor_starved_advmin_abstains": (g4_nofloor["credited"] is True and g4_floor["credited"] is False
                                                    and g4_floor["basis"] == "measured-unvalidated:starved-stratum"
                                                    and g4_floor.get("starved_strata") == ["adversarial_minimum"]),
        "G5_sample_floor_adequately_powered_still_credited": (g5["credited"] is True
                                                              and abs(g5["booked_ci_lo"] - 0.72) < 1e-9),
        "G6_sample_floor_missing_n_fail_closed": (g6["credited"] is False
                                                  and g6["basis"] == "measured-unvalidated:starved-stratum"),
        "G7_ci_lo_out_of_range_fail_closed": (g7_high["credited"] is False and g7_neg["credited"] is False
                                              and g7_high["basis"] == "measured-unvalidated:ci_lo-out-of-[0,1]"
                                              and g7_ok["credited"] is True),
    }
    return all(res.values()), {"results": res, "g1": g1, "g2": g2, "g3": g3,
                               "g4_floor": g4_floor, "g5": g5, "g6": g6, "g7_high": g7_high}


if __name__ == "__main__":
    ok, detail = _selftest()
    import json as _json
    print(_json.dumps(detail, indent=2))
    print("coverage_strata_guard selftest:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
