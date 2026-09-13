"""
gauge_detector — which parameters are unobservable, and what data would fix that?

Given a Fisher information matrix over quantities of interest, split it into the IDENTIFIABLE
subspace and the GAUGE (sigma_min-null) subspace, and recommend the decorrelated DATA CHANNEL
that lifts the gauge.

An estimate is only as good as its gauge awareness: many vision/reconstruction quantities are
certifiable only UP TO a gauge — a null direction of the observation the estimator cannot see
through, no matter how much of the SAME observation it gets. A gauge is DATA-liftable, never
dither-liftable: an estimator perturbation never lifts a shared gauge, only a decorrelated data
channel does. Typical examples: absolute depth/scale (lifted by a known length or a stereo
baseline), material x light scale and metamer spectral shape (white reference / diverse
illuminants), motion-depth scale (second view / metric anchor).

    g = certify_up_to_gauge(F, param_names) # -> {sigma_min, gauge_dim, identifiable, gauge_basis, verdict}
    rec = next_channel_to_lift(F, {name: J_channel}) # -> the data channel that most raises sigma_min

Self-contained (numpy only). Run `python gauge_detector.py` for the self-test.
"""
import numpy as np

__all__ = ["gauge_structure", "certify_up_to_gauge", "next_channel_to_lift", "name_gauge_directions", "min_channels_to_certify", "sigma_min_via_fluctuation"]

_EPS = 1e-15


def gauge_structure(F, tol_ratio=1e-6):
    """
    Split a Fisher F (p x p, symmetric PSD) into IDENTIFIABLE vs GAUGE (null) subspaces. Returns sigma_min (clipped >=0),
    gauge_dim (# eigenvalues below tol_ratio*sigma_max), the gauge basis (null eigenvectors), and the identifiable rank.
    """
    F = np.asarray(F, float)
    # "CERTIFIED (full-rank, no gauge)" (nan<tol is False -> gauge_dim 0). Guard it, matching the sibling read_envelope /
    # recon_cert_report ValueError convention -- a garbage Fisher is a caller error, not a certifiable input.
    if F.size == 0 or not np.all(np.isfinite(F)):
        raise ValueError("Fisher F must be finite and non-empty (a non-finite Fisher yields sigma_min=nan read as CERTIFIED)")
    F = 0.5 * (F + F.T) # symmetrize
    w, V = np.linalg.eigh(F)
    w = np.clip(w, 0.0, None) # Fisher eigenvalues are physically >= 0
    smax = float(w[-1]) if w.size else 0.0
    tol = tol_ratio * smax if smax > 0 else _EPS
    gauge_mask = w < tol
    return {"sigma_min": float(w[0]) if w.size else 0.0, "sigma_max": smax,
            "gauge_dim": int(gauge_mask.sum()), "identifiable_dim": int((~gauge_mask).sum()),
            "gauge_basis": V[:, gauge_mask], "eigvals": w, "tol": float(tol)}


def certify_up_to_gauge(F, param_names=None, tol_ratio=1e-6):
    """
    Deployable gauge-cert report for a reconstruction QoI Fisher. verdict: CERTIFIED (full rank) / UP_TO_GAUGE
    (gauge_dim>0, list the gauge directions in param terms) / DEGENERATE (all null).
    """
    g = gauge_structure(F, tol_ratio)
    p = int(np.asarray(F).shape[0])
    if param_names is None:
        param_names = [f"p{i}" for i in range(p)]
    gauge_dirs = name_gauge_directions(g["gauge_basis"], param_names)
    if g["gauge_dim"] == 0:
        verdict = "CERTIFIED (full-rank Fisher, no gauge)"
    elif g["identifiable_dim"] == 0:
        verdict = "DEGENERATE (no identifiable direction)"
    else:
        verdict = (f"UP_TO_GAUGE (gauge_dim={g['gauge_dim']}: certifiable up to {g['gauge_dim']} "
                   f"unobservable direction(s); lift with a decorrelated data channel)")
    return {"sigma_min": g["sigma_min"], "gauge_dim": g["gauge_dim"], "identifiable_dim": g["identifiable_dim"],
            "gauge_directions": gauge_dirs, "verdict": verdict}


def name_gauge_directions(gauge_basis, param_names):
    """Render each gauge (null) eigenvector as its dominant param combination, e.g. '+0.71*logDepth -0.71*logScale'."""
    out = []
    B = np.asarray(gauge_basis, float)
    if B.ndim == 1:
        B = B[:, None]
    for j in range(B.shape[1]):
        v = B[:, j]
        idx = np.argsort(-np.abs(v))
        terms = [f"{v[i]:+.2f}*{param_names[i]}" for i in idx if abs(v[i]) > 0.15]
        out.append(" ".join(terms) if terms else "(diffuse)")
    return out


def next_channel_to_lift(F, channel_jacobians, tol_ratio=1e-6):
    """
    Given candidate DATA channels (each a Jacobian J_c of new observations w.r.t. the same params), return them ranked by
    most lifts the gauge; if NONE lifts it, the class is externally-irreducible (needs a GT anchor).
    """
    F = np.asarray(F, float)
    base = gauge_structure(F, tol_ratio)["sigma_min"]
    lifts = []
    for name, Jc in channel_jacobians.items():
        Jc = np.asarray(Jc, float)
        if Jc.ndim == 1:
            Jc = Jc[None, :]
        F2 = F + Jc.T @ Jc
        s2 = gauge_structure(F2, tol_ratio)["sigma_min"]
        lifts.append({"channel": name, "sigma_min_after": s2, "lift": float(s2 - base)})
    lifts.sort(key=lambda d: -d["lift"])
    best = lifts[0] if lifts else None
    return {"baseline_sigma_min": base, "ranked": lifts,
            "recommend": (best["channel"] if best and best["lift"] > 1e-9 else "NONE (externally-irreducible; needs a GT anchor)")}


def min_channels_to_certify(F, channel_jacobians, tol_ratio=1e-6):
    """
    CONSTRUCTIVE: the minimal set of DATA channels that CERTIFIES a multi-gauge Fisher (breaks ALL its gauges). Greedy by
    GAUGE-DIM reduction (tie-broken by the sum of the k smallest eigenvalues), because with SEVERAL gauges sigma_min stays 0
    until the LAST gauge is broken -- so next_channel_to_lift's sigma_min metric is MYOPIC for multi-gauge acquisition
    (it reads 0 lift for every channel that leaves >=1 gauge). Returns the chosen channels, the final gauge_dim, and whether
    the twin is certified. A channel that breaks MULTIPLE gauges (e.g. a 2nd view pins depth AND motion-depth scale) is
    preferred -- acquisition is matroid COVERAGE of the gauge subspace, not one-channel-per-gauge.
    """
    F = np.asarray(F, float)
    current = F.copy(); chosen = []; remaining = dict(channel_jacobians)
    g0 = gauge_structure(current, tol_ratio); steps = 0
    while gauge_structure(current, tol_ratio)["gauge_dim"] > 0 and remaining and steps < len(channel_jacobians) + 1:
        base = gauge_structure(current, tol_ratio); k = base["gauge_dim"]
        best_name = None; best_F = None; best_key = None
        for name, Jc in remaining.items():
            Jc = np.asarray(Jc, float)
            if Jc.ndim == 1:
                Jc = Jc[None, :]
            F2 = current + Jc.T @ Jc
            g2 = gauge_structure(F2, tol_ratio)
            small_sum = float(np.sort(g2["eigvals"])[:max(k, 1)].sum()) # progress even when gauge_dim not yet dropping
            key = (base["gauge_dim"] - g2["gauge_dim"], small_sum) # 1st: reduce gauge_dim; 2nd: raise small eigs
            if best_key is None or key > best_key:
                best_key = key; best_name = name; best_F = F2
        if best_key is None or (best_key[0] <= 0 and best_key[1] <= gauge_structure(current, tol_ratio)["eigvals"][:max(k,1)].sum() + 1e-12):
            break # no channel makes progress -> externally-irreducible
        chosen.append(best_name); current = best_F; del remaining[best_name]; steps += 1
    gf = gauge_structure(current, tol_ratio)
    return {"chosen": chosen, "initial_gauge_dim": g0["gauge_dim"], "final_gauge_dim": gf["gauge_dim"],
            "certified": bool(gf["gauge_dim"] == 0), "final_sigma_min": gf["sigma_min"]}


def sigma_min_via_fluctuation(observe_fn, theta0, eps=1e-4, tol_ratio=1e-6):
    """
    MODEL-FREE sigma_min + gauge structure from a BLACK-BOX forward-observation function via ACTIVE DITHER (the FDT /
    fluctuation instrument). observe_fn(theta) -> observation vector; we finite-difference it around theta0 with NO analytic
    Jacobian, build the empirical Fisher J^T J, and read sigma_min + the gauge (null) directions. A GAUGE is a parameter
    direction the observations do NOT respond to (a free fluctuation direction: perturb along it, the observation is
    unchanged). This is the 'one sigma_min-via-fluctuation instrument' for a model-free recoverable-vs-hallucinated cert:
    feed a reconstruction + its forward model as a black box, get what is identifiable vs gauge without deriving anything.
    """
    theta0 = np.asarray(theta0, float); p = theta0.size
    obs0 = np.asarray(observe_fn(theta0), float).ravel()
    J = np.zeros((obs0.size, p))
    for i in range(p):
        step = eps * max(abs(float(theta0[i])), 1.0)
        d = np.zeros(p); d[i] = step
        plus = np.asarray(observe_fn(theta0 + d), float).ravel()
        minus = np.asarray(observe_fn(theta0 - d), float).ravel()
        J[:, i] = (plus - minus) / (2.0 * step)
    F = J.T @ J
    g = gauge_structure(F, tol_ratio)
    g["response_per_direction"] = np.sqrt(np.clip(np.linalg.eigvalsh(F), 0, None)) # per-mode fluctuation response
    return g


def _selftest():
    ok = True
    Nx, Nt = 5, 4; Dp = 1 + Nx + Nt
    rows = []
    for x in range(Nx):
        for t in range(Nt):
            r = np.zeros(Dp); r[0] = 1; r[1 + x] = 1; r[1 + Nx + t] = 1; rows.append(r)
    F = np.array(rows).T @ np.array(rows)
    g = certify_up_to_gauge(F)
    ok &= g["gauge_dim"] == 2 and "UP_TO_GAUGE" in g["verdict"]
    print(f"  material x light: gauge_dim={g['gauge_dim']} (expect 2), sigma_min={g['sigma_min']:.1e}")

    rng = np.random.default_rng(0)
    J1 = rng.standard_normal((8, 4)); scale = np.array([1., 1., -1., -1.]); scale /= np.linalg.norm(scale)
    J1 = J1 - np.outer(J1 @ scale, scale) # remove the scale direction -> J1 blind to it (a gauge)
    Fm = J1.T @ J1
    gm = gauge_structure(Fm)
    ok &= gm["gauge_dim"] >= 1
    J2 = scale[None, :] * 2.0 # a data channel that SEES the scale direction
    rec = next_channel_to_lift(Fm, {"2nd_view": J2, "more_frames": J1[:1]})
    ok &= rec["recommend"] == "2nd_view"
    print(f"  motion-depth: gauge_dim={gm['gauge_dim']}, lift recommend={rec['recommend']} (expect 2nd_view)")

    # 3) full-rank Fisher -> CERTIFIED
    Ff = np.eye(4) * 3.0
    gf = certify_up_to_gauge(Ff)
    ok &= gf["gauge_dim"] == 0 and "CERTIFIED" in gf["verdict"]
    print(f"  full-rank: {gf['verdict'][:40]}")

    # 4) no channel lifts -> externally-irreducible
    rec2 = next_channel_to_lift(Fm, {"blind_a": J1[:1], "blind_b": J1[1:2]})
    ok &= "NONE" in rec2["recommend"]
    print(f"  no-lifting-channel: recommend={rec2['recommend'][:30]}")

    # 5) gauge direction naming
    names = name_gauge_directions(gm["gauge_basis"], ["dx", "dy", "dz", "ds"])
    ok &= len(names) == gm["gauge_dim"]
    print(f"  gauge directions: {names}")

    # 6) min_channels_to_certify: a 3-gauge Fisher certified by 2 channels (one breaks 2 gauges) -- constructive/minimal
    F3 = np.diag([0., 0., 0., 1., 1., 1.]) # gauges = params 0,1,2
    chans = {"known_length": np.eye(6)[0], "white_ref": np.eye(6)[1], "known_light": np.eye(6)[2],
             "second_view": np.array([[1., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.]])} # rank-2: breaks gauges 0 AND 2
    mc = min_channels_to_certify(F3, chans)
    ok &= mc["certified"] and len(mc["chosen"]) == 2 and "second_view" in mc["chosen"]
    print(f"  min_channels_to_certify: {mc['chosen']} -> certified={mc['certified']} (expect 2 incl second_view)")

    # 7) sigma_min_via_fluctuation: MODEL-FREE read of the material x light gauge (observe = log(g)+log(a)+log(e), no analytic Fisher)
    Nx7, Nt7 = 4, 3
    def observe(theta): # black-box forward model: log-image over (x,t)
        lg = theta[0]; la = theta[1:1 + Nx7]; le = theta[1 + Nx7:]
        return (lg + la[:, None] + le[None, :]).ravel()
    th7 = np.concatenate([[0.2], rng.standard_normal(Nx7), rng.standard_normal(Nt7)])
    gf = sigma_min_via_fluctuation(observe, th7)
    ok &= gf["gauge_dim"] == 2 and gf["sigma_min"] < 1e-6 # recovers the 2-dim material x light gauge MODEL-FREE
    print(f"  sigma_min_via_fluctuation (model-free): gauge_dim={gf['gauge_dim']} (expect 2), sigma_min={gf['sigma_min']:.1e}")
    print("gauge_detector selftest:", "ALL PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
