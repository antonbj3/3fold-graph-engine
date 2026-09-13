"""
gauge_kind_cert -- callable distillation of H's session gauge/cert program.

Turns the session's demonstration nodes into ONE reusable, noise-aware instrument. Given a per-sample
design Fisher F (and a noise level), for EACH least-identified direction it decides the KIND of the
null and the SAFE lifting action, and it reports the authenticity-certification COST via the coupling law.

Consolidates:
  - capstone 3-axis KIND signature   (rank-null vs conditioning-null)          [commit dc912d30b]
  - SAFE fence (data-only curvature vs noise floor, NOT prior fluctuation)     [commit c3cd0ca69, ties C body-1043]
  - coupling law: authenticity-cert-COST = identifiability-NULL-DIMENSION      [commit e4351157a, confirms B]

KIND / action logic (all from F + noise + sample budget N -- no prior-contaminated fluctuation reader):
  q = per-sample Rayleigh curvature d^T F d along a direction (the info the DATA provides).
  tau(sigma) = sigma^2 * trace(F)/dim  (noise-aware floor, C's noise-aware tol).
  N*q  is the info after N samples.
    N*q  >  tau            -> IDENTIFIABLE      (resolved by the current design)
    q  ~  0 (< rank_tol)   -> RANK null         -> ABSTAIN; lift ONLY by a NEW decorrelated CHANNEL
    0 < q, N*q <= tau      -> CONDITIONING null -> liftable by MORE DATA (N_req = tau/q) or a wider LEVER-ARM
  authenticity_cert_cost = # directions a coherent fake can hide in = # UNRESOLVED directions
  (RANK + still-below-floor CONDITIONING) = the noise-aware identifiability null dimension. Each needs a
  distinct decorrelated, null-ALIGNED external channel (misaligned channels certify nothing).

Pure numpy. Self-contained. `python -m graph_engine.gauge_kind_cert` runs the selftest.
"""
from __future__ import annotations
import numpy as np

__all__ = ["null_kind_cert", "authenticity_cert_cost", "KIND_IDENTIFIABLE", "KIND_RANK",
           "KIND_CONDITIONING", "KIND_UNCERTAIN"]

KIND_IDENTIFIABLE = "IDENTIFIABLE"
KIND_RANK = "RANK_NULL"                 # structural: only a NEW channel lifts it
KIND_CONDITIONING = "CONDITIONING_NULL" # ill-conditioned: more data / wider lever-arm lifts it
KIND_UNCERTAIN = "KIND_UNCERTAIN"       # curvature indistinguishable from 0 given estimation noise --
                                        # cannot tell RANK from CONDITIONING -> DISAMBIGUATE before committing
                                        # (B's third action: the binary gate misses this band near the floor)


def _noise_floor(F: np.ndarray, noise_sigma: float) -> float:
    """tau = sigma^2 * mean-eigenvalue = sigma^2 * trace(F)/dim (noise-aware tol, C body-1043)."""
    dim = F.shape[0]
    return float(noise_sigma) ** 2 * float(np.trace(F)) / dim


def null_kind_cert(F: np.ndarray, noise_sigma: float, N: int = 1,
                   rank_tol: float | None = None, q_noise: float = 0.0,
                   z_kind: float = 3.0) -> dict:
    """Classify every eigen-direction of a per-sample design Fisher F.

    Parameters
    ----------
    F: (d,d) symmetric PSD per-sample Fisher / evidence Gramian of the observation.
    noise_sigma: measurement noise scale (same units that make F the per-sample info).
    N: number of samples in the current design (info along a direction is N * eigenvalue).
    rank_tol: curvature below this (per-sample) counts as a STRUCTURAL zero (RANK null).
               Default: max(eig) * 1e-9.
    q_noise: estimation std on the per-sample curvature q (how noisily F itself is measured).
              Default 0.0 -> the classifier is the original SHARP binary gate (backward compatible).
              When > 0, unresolved directions whose curvature is NOT statistically separable from 0
              (q <= z_kind * q_noise) are labelled KIND_UNCERTAIN -> DISAMBIGUATE (B's third action):
              you cannot tell a structural RANK null from a genuine tiny CONDITIONING null, so probing
              the KIND is cheaper-in-expectation than committing the wrong expensive resource.
    z_kind: how many q_noise sigmas above 0 the curvature must sit to be called CONDITIONING (default 3).

    Returns dict with per-direction verdicts (weakest first), summary counts, and the deployment action.
    Uses ONLY the data curvature vs the noise floor -- never a prior-contaminated fluctuation reader.

    SCOPE (C body-1054/1055, real eth3d): consume the discrete KIND VERDICT, which is noise-robust for
    directions FAR from the floor tau. Do NOT use the raw per-direction curvature as a noise-robust
    CONTINUOUS ranking across items -- under noise the fine order scrambles (Spearman degrades) while the
    coarse above/below-tau decision holds; misclassifications concentrate in the |q-tau|<~noise band, which
    is exactly the KIND_UNCERTAIN region (pass q_noise to flag it). The binary/discrete abstain is
    irreducible; a continuous floor-corrected score does not preserve ranking under noise.
    """
    F = np.asarray(F, float)
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be square")
    # (self-audit, repository-wide vacuous-truth-over-empty-collection sweep, mirrors the self-census):
    # a 0x0 F passes BOTH the square-shape check above AND the np.all(np.isfinite(F)) VACUOUSLY (np.all of an
    # empty array is True by convention) -- it then crashed downstream in _noise_floor's trace(F)/dim with an
    # opaque ZeroDivisionError, the exact same "wrong site, unclear message" class already fixed for non-finite F,
    # now reproduced for the EMPTY case that check didn't cover. Guard dim==0 explicitly.
    if F.shape[0] == 0:
        raise ValueError("F is empty (0x0) -- no directions to classify")
    # self-audit (repository-wide NaN-fail-open sweep): a non-finite F previously reached the n_req
    # computation deep in the per-direction loop and crashed with an OPAQUE "cannot convert float NaN to integer"
    # (ValueError, but at the WRONG site -- no caller-actionable message, and every comparison ABOVE it (q_tot>tau,
    # q<rank_tol, q<=q_decide) silently evaluates False on NaN, so a NaN curvature could ALSO fall through to the
    # CONDITIONING branch on paths that never hit int -- e.g. if a caller wraps this in try/except ValueError and
    # only catches the crash, other NaN directions in the SAME F would silently mis-classify as CONDITIONING with a
    # garbage/NaN n_req, never surfaced). FAIL FAST AND CLEARLY at the source instead: a degenerate Fisher cannot be
    # classified at all -- this is a caller-input-quality problem, not a directional-kind decision.
    if not np.all(np.isfinite(F)):
        raise ValueError("F contains non-finite (NaN/inf) entries -- cannot classify a degenerate Fisher; "
                         "fix the upstream Jacobian/observation before calling null_kind_cert")
    # (self-audit, prompted by the new consume-side fail-open class -- a verdict decision that guards one
    # input's score but not another input that also feeds its own threshold): guarded F for exactly this
    # OverflowError-via-int(ceil(tau/q)) mechanism, but noise_sigma feeds the SAME tau computation (_noise_floor)
    # and was left unguarded. Reproduced live: noise_sigma=inf -> tau=inf -> most directions fall to the
    # CONDITIONING branch's `n_req = int(np.ceil(tau/q))` -> int(inf) -> uncaught OverflowError, the exact
    # opaque-crash mechanism 's own comment describes, just reached via the other input. Guard noise_sigma
    # the same way, at the same source, for consistency with 's own established convention.
    if not np.isfinite(noise_sigma):
        raise ValueError("noise_sigma is non-finite (NaN/inf) -- cannot compute a trustworthy noise floor; "
                         "fix the upstream noise estimate before calling null_kind_cert")
    w, V = np.linalg.eigh((F + F.T) / 2.0)          # ascending eigenvalues = per-sample curvatures q
    dim = F.shape[0]
    tau = _noise_floor(F, noise_sigma)
    if rank_tol is None:
        rank_tol = max(float(w[-1]), 1e-300) * 1e-9
    q_decide = z_kind * float(q_noise)              # curvature must clear this to be confidently CONDITIONING

    directions = []
    for i in range(dim):
        q = float(max(w[i], 0.0))                    # per-sample curvature along eigenvector i
        q_tot = N * q                                # info after N samples
        if q_tot > tau:
            kind, action, n_req = KIND_IDENTIFIABLE, "resolved -- no action", None
        elif q < rank_tol:
            kind, action, n_req = (KIND_RANK,
                                   "ABSTAIN: add a NEW decorrelated channel (data/lever-arm will NOT lift)",
                                   None)
        elif q <= q_decide:                          # not separable from 0 given estimation noise
            kind, action, n_req = (KIND_UNCERTAIN,
                                   "DISAMBIGUATE before committing a channel vs samples: sample-bump probe "
                                   "(add samples, sigma_min rises=>conditioning, flat=>rank) is valid ONLY "
                                   "under AVERAGING noise; under SYSTEMATIC noise (estimate plateaus above "
                                   "the 1/sqrt(N) envelope) it converges to a BIASED value and mis-reads a "
                                   "rank null as conditioning -- then disambiguate via a DECORRELATED channel "
                                   "(independent bias). [C body-1050 / systematic-bias-blindspot]",
                                   None)
        else:
            n_req = int(np.ceil(tau / q))
            kind, action = (KIND_CONDITIONING,
                            f"liftable: ~{n_req} samples (or a wider lever-arm) crosses the noise floor")
        directions.append({"index": i, "curvature_persample": q, "info_at_N": q_tot,
                            "kind": kind, "action": action, "n_req_to_resolve": n_req,
                            "eigvec": V[:, i]})

    unresolved = [d for d in directions if d["kind"] != KIND_IDENTIFIABLE]
    rank_dirs = [d for d in directions if d["kind"] == KIND_RANK]
    cond_dirs = [d for d in directions if d["kind"] == KIND_CONDITIONING]
    uncertain_dirs = [d for d in directions if d["kind"] == KIND_UNCERTAIN]
    # dominant_kind: the single top-level KIND a consumer (e.g. G's full_sim_ready_gate) reads to bind a fix —
    # IDENTIFIABLE if nothing is unresolved, else the null KIND with the most directions. (Added on B's
    # gate:56 bug: the gate read kc['dominant_kind']/['kind'] which this cert did not emit → always NON_IDENTIFIABLE.
    # Fix at the SOURCE: emit the key the consumer naturally reads; additive, backward-compatible.)
    # dominant_kind uses SEVERITY PRECEDENCE (rank > conditioning > uncertain), NOT max-count — so it MATCHES
    # G's full_sim_ready_gate: count-based read (IDENTIFIABLE if none unresolved, else RANK_NULL if any rank,
    # else CONDITIONING_NULL, else KIND_UNCERTAIN). A consumer reading dominant_kind gets the SAME verdict as the
    # gate on a MIXED null (e.g. 1 rank + 2 conditioning → RANK_NULL, the most severe). (max-count would have
    # disagreed; agents'-word-never-final caught it on the H⊕G fix composition.)
    if not unresolved:
        dominant_kind = KIND_IDENTIFIABLE
    elif rank_dirs:
        dominant_kind = KIND_RANK
    elif cond_dirs:
        dominant_kind = KIND_CONDITIONING
    else:
        dominant_kind = KIND_UNCERTAIN
    return {
        "noise_floor_tau": tau, "N": N, "dim": dim,
        "directions": directions,
        "n_identifiable": dim - len(unresolved),
        "n_unresolved": len(unresolved),
        "n_rank_null": len(rank_dirs),
        "n_conditioning_null": len(cond_dirs),
        "n_kind_uncertain": len(uncertain_dirs),
        "dominant_kind": dominant_kind, "kind": dominant_kind,   # top-level verdict for gate consumers (B gate:56)
        # coupling law: a coherent fake can hide in ANY unresolved direction (uncertain ones included)
        "authenticity_cert_cost": len(unresolved),
        "deployment": _advise(rank_dirs, cond_dirs, uncertain_dirs),
    }


def authenticity_cert_cost(F: np.ndarray, noise_sigma: float, N: int = 1) -> int:
    """Coupling law (H commit e4351157a, confirms B): the minimum number of distinct decorrelated,
    null-ALIGNED external channels needed to certify authenticity = the noise-aware identifiability
    null dimension = # directions a coherent fake can hide in. (Tight LOWER bound: misaligned channels
    certify nothing.)"""
    return int(null_kind_cert(F, noise_sigma, N)["authenticity_cert_cost"])


def _advise(rank_dirs, cond_dirs, uncertain_dirs=()) -> str:
    if not rank_dirs and not cond_dirs and not uncertain_dirs:
        return "fully identifiable at this design+noise; authenticity certifiable (null empty)"
    parts = []
    if uncertain_dirs:
        parts.append(f"{len(uncertain_dirs)} KIND-UNCERTAIN null(s): DISAMBIGUATE (probe the KIND) before "
                     "committing -- curvature not separable from 0 given q_noise (B's third action)")
    if cond_dirs:
        parts.append(f"{len(cond_dirs)} conditioning null(s): add data / widen lever-arm")
    if rank_dirs:
        parts.append(f"{len(rank_dirs)} RANK null(s): add {len(rank_dirs)} NEW decorrelated channel(s) "
                     "-- data & lever-arm cannot lift these (this is the authenticity floor)")
    return "; ".join(parts)


# --------------------------- selftest ---------------------------
def _selftest() -> bool:
    rng = np.random.default_rng(0)
    passed = 0
    total = 0

    def check(name, cond):
        nonlocal passed, total
        total += 1
        passed += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1) RANK null: rank-deficient Fisher (metamer) -> a RANK_NULL direction, not liftable by N
    S = np.linalg.qr(rng.standard_normal((5, 5)))[0][:, :3]
    Fm = S @ S.T                                   # 5x5 rank 3 -> 2 structural nulls
    r = null_kind_cert(Fm, noise_sigma=1e-3, N=10_000)
    check("rank-deficient Fisher yields RANK nulls", r["n_rank_null"] == 2)
    check("RANK nulls are NOT resolved even at huge N",
          all(d["kind"] == KIND_RANK for d in r["directions"] if d["curvature_persample"] < 1e-12))
    check("authenticity_cert_cost == null dim (coupling law)",
          r["authenticity_cert_cost"] == 2 == authenticity_cert_cost(Fm, 1e-3, 10_000))
    # dominant_kind (gate-consumer key, B gate:56 bug): rank-deficient → RANK; full-rank → IDENTIFIABLE
    check("dominant_kind = RANK for a rank-deficient Fisher (+ 'kind' alias)",
          r["dominant_kind"] == KIND_RANK and r["kind"] == KIND_RANK)
    check("dominant_kind = IDENTIFIABLE for a well-conditioned Fisher",
          null_kind_cert(np.diag([4.0, 3.0, 2.0]), noise_sigma=0.01)["dominant_kind"] == KIND_IDENTIFIABLE)

    # 2) CONDITIONING null: full-rank but ill-conditioned -> liftable by data, N_req finite
    Fc = np.diag([1.0, 1.0, 1e-4])                 # soft 3rd direction, nonzero curvature
    rc = null_kind_cert(Fc, noise_sigma=0.3, N=1)  # tau = 0.09*trace/3
    soft = rc["directions"][0]
    check("ill-conditioned direction classified CONDITIONING", soft["kind"] == KIND_CONDITIONING)
    check("CONDITIONING null reports a finite N_req", isinstance(soft["n_req_to_resolve"], int))
    check("CONDITIONING null becomes IDENTIFIABLE at large N",
          null_kind_cert(Fc, noise_sigma=0.3, N=soft["n_req_to_resolve"] * 10)["directions"][0]["kind"]
          == KIND_IDENTIFIABLE)

    # 3) fully identifiable: well-conditioned, low noise -> null empty, cost 0
    Fi = np.diag([1.0, 2.0, 3.0])
    ri = null_kind_cert(Fi, noise_sigma=1e-3, N=100)
    check("well-conditioned low-noise -> null empty", ri["n_unresolved"] == 0)
    check("authenticity certifiable when null empty", ri["authenticity_cert_cost"] == 0)

    # 4) noise raises the floor -> a weak conditioning null must ABSTAIN (C body-1043 noise-aware tol)
    hi_noise = null_kind_cert(Fc, noise_sigma=1.0, N=1)
    check("higher noise leaves the weak direction unresolved (noise-aware floor)",
          hi_noise["n_unresolved"] >= 1)

    # 5) KIND_UNCERTAIN (B's third action) -- backward compatible + opt-in via q_noise
    # a genuine tiny CONDITIONING null (q=5e-5); with q_noise=0 it reads CONDITIONING (old behavior)
    Fu = np.diag([1.0, 1.0, 5e-5])
    r_sharp = null_kind_cert(Fu, noise_sigma=0.3, N=1)                    # q_noise default 0
    check("q_noise=0 preserves the old binary gate (no KIND_UNCERTAIN)",
          r_sharp.get("n_kind_uncertain", 0) == 0 and r_sharp["directions"][0]["kind"] == KIND_CONDITIONING)
    # with estimation noise q_noise=1e-4, that q=5e-5 is NOT separable from 0 (< 3*q_noise) -> UNCERTAIN
    r_unc = null_kind_cert(Fu, noise_sigma=0.3, N=1, q_noise=1e-4)
    check("weak curvature within estimation noise -> KIND_UNCERTAIN (disambiguate)",
          r_unc["directions"][0]["kind"] == KIND_UNCERTAIN and r_unc["n_kind_uncertain"] == 1)
    # a curvature well above the estimation noise stays CONDITIONING even with q_noise on
    Fc2 = np.diag([1.0, 1.0, 2e-3])
    r_conf = null_kind_cert(Fc2, noise_sigma=0.3, N=1, q_noise=1e-4)
    check("curvature >> 3*q_noise stays CONDITIONING", r_conf["directions"][0]["kind"] == KIND_CONDITIONING)
    # an exact structural zero stays RANK regardless of q_noise
    Fr = np.diag([1.0, 1.0, 0.0])
    r_rank = null_kind_cert(Fr, noise_sigma=1e-3, N=1, q_noise=1e-4)
    check("exact structural zero stays RANK under q_noise", r_rank["n_rank_null"] == 1)

    # 6) API guards
    try:
        null_kind_cert(np.zeros((2, 3)), 1.0); ok = False
    except ValueError:
        ok = True
    check("non-square F rejected", ok)

    # 7) (self-audit, repository-wide NaN-fail-open sweep): a NaN/inf entry in F must FAIL FAST with a clear
    # message at the top of the function, not crash deep inside the loop with an opaque int(nan) ValueError, and
    # must NOT silently fall through any comparison to a garbage CONDITIONING/n_req classification.
    for bad_F, name in [(np.diag([1.0, 1.0, float("nan")]), "NaN"), (np.diag([1.0, 1.0, float("inf")]), "inf")]:
        try:
            null_kind_cert(bad_F, 0.3, N=1)
            nonfinite_ok = False
        except ValueError as e:
            nonfinite_ok = "non-finite" in str(e)
        check(f"non-finite F ({name}) fails fast with a clear message", nonfinite_ok)

    # 8) (self-audit, vacuous-truth-over-empty-collection sweep, mirrors the earlier sweep): an EMPTY (0x0) F
    # passes np.all(np.isfinite(.)) VACUOUSLY (True on an empty array) and previously crashed downstream with an
    # opaque ZeroDivisionError -- must fail fast with a clear message instead.
    try:
        null_kind_cert(np.zeros((0, 0)), 0.3, N=1)
        empty_ok = False
    except ValueError as e:
        empty_ok = "empty" in str(e)
    except ZeroDivisionError:
        empty_ok = False
    check("empty (0x0) F fails fast with a clear message (not a ZeroDivisionError)", empty_ok)

    # 9) (self-audit, prompted by the consume-side fail-open class): a non-finite noise_sigma feeds the
    # SAME tau computation as F does, and previously crashed with the identical opaque int(inf)/int(nan)
    # OverflowError/ValueError fixed for F -- must fail fast with a clear message at the same source.
    for bad_sigma, name in [(float("nan"), "NaN"), (float("inf"), "inf")]:
        good_F = np.diag([1.0, 2.0, 3.0])
        try:
            null_kind_cert(good_F, bad_sigma, N=1)
            nonfinite_sigma_ok = False
        except ValueError as e:
            nonfinite_sigma_ok = "non-finite" in str(e)
        except OverflowError:
            nonfinite_sigma_ok = False
        check(f"non-finite noise_sigma ({name}) fails fast with a clear message", nonfinite_sigma_ok)

    print(f"gauge_kind_cert selftest: {passed}/{total}")
    return passed == total


if __name__ == "__main__":
    # --selftest is accepted explicitly (module-bank detectable selftest path); running with no args
    # keeps the original behavior (the selftest IS this module's CLI).
    import argparse
    _ap = argparse.ArgumentParser(description="gauge_kind_cert selftest runner")
    _ap.add_argument("--selftest", action="store_true", help="run the selftest (also the default action)")
    _ap.parse_args()
    ok = _selftest()
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
