"""
motion_validity_cert — DEPLOYABLE PAYOFF of C's motion-generation prior vein (bea8efcfa/5389cb219/8febe083b/25e593b46).
Agent pool-consumable callable (matches criticality_safety_cert / BMS lever / observation_budget deployable-payoff pattern).

Give it a real/generated/estimated 6-DoF trajectory; it returns the motion-generation-prior cert vector:
  - the const-acceleration (2nd-order) local-smoothness fit vs const-velocity (is the motion 2nd-order predictable?),
  - the PREDICTABILITY HORIZON H* (where const-vel prediction error reaches 50% of the motion scale),
  - a per-frame VALIDITY residual + flags against a prior-grounded threshold (implausible motion = generated-model glitch or
    perception/VO failure fires high residual).

Pure numpy + scipy; no substrate. an agent worktree = cert-tooling (allowed). Positions in metres, dt in seconds.

★OPERATING CONSTRAINT (independent-adversary discovery, evidence spine_c_provenance_gate_independent_injector_adversary_C):
the per-frame residual uses gaussian_filter1d(sigma=3) smoothing of the derivatives + a prediction horizon of h samples,
which creates a ~40-sample CONTAMINATION BAND around each anomaly. When EVALUATING detection AUC on data with many closely
spaced anomalies (spacing < ~40 samples), neighbour contamination corrupts the 'clean'-class residuals and UNDERSTATES the AUC
— this is a measurement-DENSITY artifact, NOT a detectability limit. For a fair AUC benchmark, space injected defects >= ~150
samples or score only far-clean negatives (>100 samples from any anomaly). Real per-frame detection/quarantine is unaffected.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def _resid(pos, dt, h, smooth_sigma=3.0):
    """per-frame const-acceleration prediction residual over horizon h (samples)."""
    ps = gaussian_filter1d(np.asarray(pos, float), smooth_sigma, axis=0)
    v = np.gradient(ps, axis=0) / dt
    a = np.gradient(v, axis=0) / dt
    n = len(pos); res = np.full(n, np.nan)
    idx = np.arange(0, n - h)
    pred = pos[idx] + v[idx] * (h * dt) + 0.5 * a[idx] * (h * dt) ** 2
    res[idx] = np.linalg.norm(np.asarray(pos)[idx + h] - pred, axis=1)
    return res


def motion_validity_cert(positions, dt, horizon_sec=0.05, valid_threshold_m=None, valid_quantile=99.0):
    """
    positions: (N,3) array of 3-D positions (m). dt: sample period (s).
    horizon_sec: cert prediction horizon (default 50 ms). valid_threshold_m: if given, the prior-grounded validity threshold;
    else derived as the `valid_quantile` percentile of this trajectory's own residual (self-calibrating).
    Returns a dict cert-vector.
    """
    positions = np.asarray(positions, float)
    if positions.ndim != 2 or positions.shape[1] != 3:   # explicit raise: -O strips `assert` -> a guard-assert fails OPEN
        raise ValueError("positions must be (N,3)")
    n0 = len(positions)
    # ★DOCUMENTED-BUT-UNGATED FIX (-class audit): a too-short trajectory cannot support a 2nd-order (const-accel)
    # predictability residual -- n<2 makes np.gradient RAISE, and n in {2,3} silently yields all-False invalid_flags +
    # a spurious H* (a downstream false-CLEAN). Cannot certify -> conservatively mark ALL frames invalid (a data-poisoning
    # gate must quarantine an unassessable short corpus, not admit it) and report H*=nan.
    if n0 < 4:
        return {"second_order_predictable": None,
                "err_const_pos_m": float("nan"), "err_const_vel_m": float("nan"), "err_const_accel_m": float("nan"),
                "predictability_horizon_sec": float("nan"),
                "per_frame_residual_m": np.full(n0, np.nan), "validity_threshold_m": float("nan"),
                "invalid_flags": np.ones(n0, bool), "invalid_fraction": 1.0,
                "verdict": "ABSTAIN-INSUFFICIENT (n<4: cannot assess 2nd-order predictability; quarantine-conservative)"}
    h = max(1, int(round(horizon_sec / dt)))
    # model-order comparison at the cert horizon
    ps = gaussian_filter1d(positions, 3.0, axis=0); v = np.gradient(ps, axis=0) / dt; a = np.gradient(v, axis=0) / dt
    idx = np.arange(0, len(positions) - h); true = positions[idx + h]
    e_p0 = float(np.sqrt(np.mean(np.sum((positions[idx] - true) ** 2, 1))))
    e_p1 = float(np.sqrt(np.mean(np.sum((positions[idx] + v[idx] * (h * dt) - true) ** 2, 1))))
    e_p2 = float(np.sqrt(np.mean(np.sum((positions[idx] + v[idx] * (h * dt) + 0.5 * a[idx] * (h * dt) ** 2 - true) ** 2, 1))))
    # predictability horizon H*: sweep horizons, find where const-vel err = 50% of motion scale
    Hstar = None
    hs = [max(1, int(round(t / dt))) for t in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)]
    hsec = np.array(hs) * dt; fr = []; scales = []
    for hh in hs:
        ii = np.arange(0, len(positions) - hh); tr = positions[ii + hh]
        e1 = np.sqrt(np.mean(np.sum((positions[ii] + v[ii] * (hh * dt) - tr) ** 2, 1)))
        sc = np.sqrt(np.mean(np.sum((tr - positions[ii]) ** 2, 1)))
        fr.append(e1 / (sc + 1e-12)); scales.append(sc)
    fr = np.array(fr)
    # ★DOCUMENTED-BUT-UNGATED FIX (-class audit): an excitation-starved / static trajectory (near-zero motion scale at
    # EVERY horizon -- e.g. a frozen/duplicated sensor frame) makes fr=e1/sc~0/0~0, which the fallback below read as "still
    # predictable" and returned the MAX horizon (a confident H*). scope of the consumers declares such objects out-of-scope
    # ("no measurable H*"). H* is UNDEFINED here -> return nan (reid_persistence_cert then correctly ABSTAINs on H<=0/nan).
    motion_scale = float(np.max(scales)) if scales else 0.0
    if motion_scale < 1e-9:
        Hstar = float("nan")
    else:
        for j in range(1, len(hs)):
            if fr[j] >= 0.5 and fr[j - 1] < 0.5:
                f = (0.5 - fr[j - 1]) / (fr[j] - fr[j - 1] + 1e-12); Hstar = float(hsec[j - 1] + f * (hsec[j] - hsec[j - 1])); break
        if Hstar is None:
            Hstar = float(hsec[-1]) if fr[-1] < 0.5 else float(hsec[0])
    # per-frame validity
    res = _resid(positions, dt, h)
    self_calibrated = valid_threshold_m is None
    # ★SUPPLIED-THRESHOLD FINITENESS GUARD (supplied-ref-bypass class + the np.maximum-NaN-blind sweep): the
    # SELF-calibrated path uses NaN-robust nanpercentile, but a caller-SUPPLIED valid_threshold_m bypasses it unvalidated. A
    # non-finite supplied threshold (NaN OR +inf) makes `res > thr` = False for EVERY frame -> invalid_fraction=0 -> the motion
    # reads VALID regardless of real faults = fail-open (measured: a 5m jump missed). The line-below guard flags non-finite
    # POSITIONS but not a non-finite THRESHOLD. A residual-magnitude threshold must be finite and >= 0 -> fail-loud.
    if valid_threshold_m is not None and not (np.isfinite(valid_threshold_m) and valid_threshold_m >= 0.0):
        raise ValueError("valid_threshold_m must be finite and >= 0: a non-finite supplied threshold makes `res > thr` "
                         "False for all frames -> the motion reads VALID regardless of real faults (fail-open)")
    thr = valid_threshold_m if valid_threshold_m is not None else float(np.nanpercentile(res, valid_quantile))
    flags = np.zeros(len(positions), bool); m = ~np.isnan(res); flags[m] = res[m] > thr
    flags[~np.isfinite(positions).all(axis=1)] = True   # ★non-finite position is corrupt -> INVALID (post-ship audit fix: a NaN
    # residual silently fails `> thr` (NaN>x=False), so an un-guarded NaN frame passes as valid — flag it explicitly)
    return {
        "second_order_predictable": bool(e_p2 <= e_p1 <= e_p0 * 1.02 and e_p2 < e_p0 * 0.9),  # ★+predictability MARGIN (null-safety wave, threshold CALIBRATED from the distribution not one draw, per order-fil §6): smoothed WHITE NOISE gives e_p2/e_p0~1.13-1.22 (const-accel provides NO gain), real motion max 0.646 over 120 EuRoC+TUM windows -> 0.9 cleanly separates (all real pass, noise fails); an earlier 0.5 false-abstained 3% of real EuRoC windows
        "err_const_pos_m": e_p0, "err_const_vel_m": e_p1, "err_const_accel_m": e_p2,
        "predictability_horizon_sec": Hstar,
        "per_frame_residual_m": res, "validity_threshold_m": thr,
        "invalid_flags": flags, "invalid_fraction": float(np.mean(flags)),  # over ALL frames (incl non-finite); was flags[m], which silently dropped NaN frames from the count
        # ★THRESHOLD-PROVENANCE (per agent pool threshold-provenance sweep): a self-calibrated (own-quantile) threshold
        # makes invalid_fraction ~= (100-valid_quantile)/100 BY CONSTRUCTION -- a NOMINAL ranking artifact, NOT a measured
        # fake-rejection rate. Carry the mode in the OUTPUT (was docstring-only = documented-but-ungated) so a consumer cannot
        # read a self-percentile as effective detection. EFFECTIVE requires an external/known-fake-calibrated valid_threshold_m.
        "threshold_provenance": "self-calibrated-NOMINAL" if self_calibrated else "external-EFFECTIVE",
        "nominal_invalid_rate": (100.0 - valid_quantile) / 100.0 if self_calibrated else None,
    }


def _omega(R, dt):
    """body-frame angular velocity series (N,3), padded to length N."""
    w = (R[:-1].inv() * R[1:]).as_rotvec() / dt
    return np.vstack([w, w[-1]])


def _rot_resid(R, w, a, dt, h):
    """per-frame const-angular-acceleration attitude prediction geodesic residual (rad), over horizon h samples."""
    n = len(R); res = np.full(n, np.nan)
    idx = np.arange(0, n - h)
    Rt = R[idx]
    Rpred = Rt * Rotation.from_rotvec(w[idx] * (h * dt) + 0.5 * a[idx] * (h * dt) ** 2)
    res[idx] = (Rpred.inv() * R[idx + h]).magnitude()
    return res


def orientation_validity_cert(quaternions, dt, horizon_sec=0.05, valid_threshold_rad=None, valid_quantile=99.0):
    """
    quaternions: (N,4) in scipy order [x,y,z,w] (EuRoC w,x,y,z -> reorder to x,y,z,w before calling; TUM is already x,y,z,w).
    Attitude analogue of motion_validity_cert: const-angular-accel (2nd-order) predictability of attitude on SO(3), the
    rotation predictability horizon, and per-frame attitude validity via a geodesic residual. Returns a dict cert-vector.
    """
    q = np.asarray(quaternions, float)
    if q.ndim != 2 or q.shape[1] != 4:                   # explicit raise: -O strips `assert` -> a guard-assert fails OPEN
        raise ValueError("quaternions must be (N,4) in scipy [x,y,z,w] order")
    R = Rotation.from_quat(q); n = len(R)
    # ★MISSING-ABSTAIN-GUARD FIX (self-census wave): the translation sibling motion_validity_cert quarantines
    # n<4 (a too-short sequence cannot support a 2nd-order predictability residual; n in {2,3} yields a spurious CLEAN);
    # orientation_validity_cert replicated the SAME 2nd-order computation but had NO such guard -> a 3-sample smooth
    # attitude sequence returned attitude_second_order_predictable=True + a finite H* (a false attitude CLEAN the sibling
    # ABSTAINs on). Mirror the sibling's n<4 quarantine, fail-closed (all frames invalid), before any 2nd-order/H* compute.
    if n < 4:
        return {"attitude_second_order_predictable": None,
                "err_const_att_rad": float("nan"), "err_const_omega_rad": float("nan"), "err_const_alpha_rad": float("nan"),
                "rotation_predictability_horizon_sec": float("nan"),
                "per_frame_geodesic_residual_rad": np.full(n, np.nan), "validity_threshold_rad": float("nan"),
                "attitude_invalid_flags": np.ones(n, bool), "attitude_invalid_fraction": 1.0,
                "threshold_provenance": "abstain-insufficient", "nominal_invalid_rate": None,
                "verdict": "ABSTAIN-INSUFFICIENT (n<4: cannot assess 2nd-order attitude predictability; quarantine-conservative)"}
    h = max(1, int(round(horizon_sec / dt)))
    w = _omega(R, dt)
    ws = gaussian_filter1d(w, 3.0, axis=0)                 # smoothed omega for the model derivatives
    a = np.gradient(ws, axis=0) / dt                       # angular acceleration
    idx = np.arange(0, n - h); Rt = R[idx]; Rtrue = R[idx + h]
    e_p0 = float(np.sqrt(np.mean((Rt.inv() * Rtrue).magnitude() ** 2)))
    e_p1 = float(np.sqrt(np.mean(((Rt * Rotation.from_rotvec(ws[idx] * (h * dt))).inv() * Rtrue).magnitude() ** 2)))
    e_p2 = float(np.sqrt(np.mean(((Rt * Rotation.from_rotvec(ws[idx] * (h * dt) + 0.5 * a[idx] * (h * dt) ** 2)).inv() * Rtrue).magnitude() ** 2)))
    # rotation predictability horizon H*: sweep, find where const-omega err = 50% of attitude scale
    Hstar = None
    # ★EMPTY-ARANGE GUARD (self-census wave): the fixed sweep horizons (up to 1.6s) give hh=round(t/dt) that can
    # EXCEED n at a fine dt / short sequence -> arange(0,n-hh) is empty -> from_rotvec(empty) RAISES (crash, not a verdict).
    # Keep only horizons with >=1 evaluable pair (n-hh>=1); n>=4 is already guaranteed by the abstain guard above.
    hs = [hh for hh in (max(1, int(round(t / dt))) for t in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)) if n - hh >= 1]
    if not hs:
        hs = [1]                                          # n>=4 guarantees n-1>=3 pairs at the shortest horizon
    hsec = np.array(hs) * dt; fr = []
    for hh in hs:
        ii = np.arange(0, n - hh); Ri = R[ii]; Rtr = R[ii + hh]
        e1 = np.sqrt(np.mean(((Ri * Rotation.from_rotvec(ws[ii] * (hh * dt))).inv() * Rtr).magnitude() ** 2))
        sc = np.sqrt(np.mean((Ri.inv() * Rtr).magnitude() ** 2))
        fr.append(e1 / (sc + 1e-12))
    fr = np.array(fr)
    for j in range(1, len(hs)):
        if fr[j] >= 0.5 and fr[j - 1] < 0.5:
            f = (0.5 - fr[j - 1]) / (fr[j] - fr[j - 1] + 1e-12); Hstar = float(hsec[j - 1] + f * (hsec[j] - hsec[j - 1])); break
    if Hstar is None:
        Hstar = float(hsec[-1]) if fr[-1] < 0.5 else float(hsec[0])
    res = _rot_resid(R, ws, a, dt, h)
    self_calibrated = valid_threshold_rad is None
    # ★SUPPLIED-THRESHOLD FINITENESS GUARD (sibling of motion_validity_cert above, supplied-ref-bypass class): a
    # non-finite supplied valid_threshold_rad (NaN/+inf) makes `res > thr` False for every frame -> attitude reads VALID
    # regardless = fail-open. A geodesic-residual threshold must be finite and >= 0 -> fail-loud.
    if valid_threshold_rad is not None and not (np.isfinite(valid_threshold_rad) and valid_threshold_rad >= 0.0):
        raise ValueError("valid_threshold_rad must be finite and >= 0: a non-finite supplied threshold makes `res > thr` "
                         "False for all frames -> the attitude reads VALID regardless of real faults (fail-open)")
    thr = valid_threshold_rad if valid_threshold_rad is not None else float(np.nanpercentile(res, valid_quantile))
    flags = np.zeros(n, bool); mm = ~np.isnan(res); flags[mm] = res[mm] > thr
    flags[~np.isfinite(q).all(axis=1)] = True           # ★non-finite quaternion is corrupt -> INVALID (post-ship audit fix)
    return {
        "attitude_second_order_predictable": bool(e_p2 <= e_p1 <= e_p0 * 1.02 and e_p2 < e_p0 * 0.9),  # ★+predictability margin (null-safety wave, calibrated 0.9 per translation)
        "err_const_att_rad": e_p0, "err_const_omega_rad": e_p1, "err_const_alpha_rad": e_p2,
        "rotation_predictability_horizon_sec": Hstar,
        "per_frame_geodesic_residual_rad": res, "validity_threshold_rad": thr,
        "attitude_invalid_flags": flags, "attitude_invalid_fraction": float(np.mean(flags)),  # over ALL frames (incl non-finite)
        # ★THRESHOLD-PROVENANCE (per agent pool sweep): self-calibrated own-quantile -> attitude_invalid_fraction
        # ~= (100-valid_quantile)/100 by construction = NOMINAL, not measured fake-rejection. Carry the mode in the output.
        "threshold_provenance": "self-calibrated-NOMINAL" if self_calibrated else "external-EFFECTIVE",
        "nominal_invalid_rate": (100.0 - valid_quantile) / 100.0 if self_calibrated else None,
    }


def pose_validity_cert(positions, quaternions, dt, horizon_sec=0.05,
                       valid_threshold_m=None, valid_threshold_rad=None, valid_quantile=99.0):
    """
    FULL 6-DoF pose validity: translation (motion_validity_cert) + attitude (orientation_validity_cert). Returns both
    sub-certs plus a combined `pose_valid` (2nd-order predictable on BOTH DoF families) and `invalid_flags` (a frame is
    invalid if EITHER position OR attitude fires). quaternions in scipy [x,y,z,w] order (see orientation_validity_cert).
    """
    t = motion_validity_cert(positions, dt, horizon_sec, valid_threshold_m, valid_quantile)
    r = orientation_validity_cert(quaternions, dt, horizon_sec, valid_threshold_rad, valid_quantile)
    n = min(len(t["invalid_flags"]), len(r["attitude_invalid_flags"]))
    combined = t["invalid_flags"][:n] | r["attitude_invalid_flags"][:n]
    return {
        "translation": t, "rotation": r,
        "pose_second_order_predictable": bool(t["second_order_predictable"] and r["attitude_second_order_predictable"]),
        "translation_horizon_sec": t["predictability_horizon_sec"], "rotation_horizon_sec": r["rotation_predictability_horizon_sec"],
        "invalid_flags": combined, "invalid_fraction": float(np.mean(combined)),
    }
