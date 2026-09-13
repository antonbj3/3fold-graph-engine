"""video_admission_identifiability_pregate — an agent worktree: the DEPLOY-TIME realization of 's depth-gauge pre-gate,
decided from REAL feature tracks alone (no known motion type — that is what this detects), for
video_admission_cert (see video_admission_cert.py in this package) to wire in
as a SHARED veto_leg (name="depth_recoverability_pregate").


Theory anchor: scripts/physics_exp/g1125_video_admission_identifiability_pregate_approach_ADMIT_zoom_rotation_ABSTAIN.py
(PYTHONPATH=srcpython) proves via the JOINT (3D points ⊕ motion) design-Jacobian NULLITY that
APPROACH (parallax) => depth recoverable up to overall scale (nullity 1) = ADMIT; ZOOM and PURE-ROTATION => depth is
a GAUGE (nullity N+1 / N) = ABSTAIN.  uses the KNOWN motion type in its reprojection model; this module cannot
assume that — it must discriminate the gauge from tracks alone.

★THE OBSERVABLE DISCRIMINANT (deploy-time realization of 's nullity split) — the classic degenerate-motion
H-vs-E model-selection from multi-view geometry (Torr & Murray 1997; Hartley & Zisserman ch.11; the basis of
ORB-SLAM's H/F initialization test):
  - ZOOM: R=I, t=0, only focal changes. A pure isotropic image-plane scaling about the principal point is EXACTLY a
    homography (H = diag(s,s,1) in centered coords) for ANY scene, planar or not.
  - PURE ROTATION: t=0. The "infinite homography" H_inf = K R K^-1 EXACTLY explains pure-rotation correspondences
    for ANY 3D scene structure, planar or not (a classical MVG fact, not a heuristic).
  - APPROACH (translation, parallax on a non-planar scene): no single homography explains parallax from a
    non-planar structure => H residuals GROW with baseline, past what the assumed pixel-noise floor can explain.
So the test is: does a single global homography H (RANSAC-fit) explain ALL correspondences at the assumed pixel-
noise level? This is a formal reduced-chi-square GOODNESS-OF-FIT test for the H model: the forward-transfer squared
error e_H^2=||x'-Hx||^2 is a genuine 2-dim residual (x,y free params -> x',y' determined), so under H0 ("H explains
the motion, given only pixel noise") e_H^2/sigma^2 ~ chi-square(2), median ≈ 1.386. parallax_score = the ratio of
the OBSERVED median(e_H^2) to that expected null median, aggregated (median) across consecutive-frame pairs.
Score ≈ 1 => H suffices => depth is a gauge => ABSTAIN. Score >> 1 => H is REJECTED by the data => parallax present
=> ADMIT.

★WHY NOT RAW GRIC / H-vs-E RESIDUAL RATIO (a documented dead end, kept here so the fix isn't silently lost): the
first design compared H's residual to the ESSENTIAL matrix E's Sampson residual via Torr's GRIC. This FAILED to
separate PURE ROTATION from real parallax: under t=0, the epipolar constraint x1^T F x0=0 is satisfied EXACTLY by
ANY F of the form [e']_x H for an arbitrary epipole e' (since x1=Hx0 exactly and [e']_x x1 ⊥ x1 trivially) — a
classical MVG degenerate-F fact. So E's residual sits at the noise floor for rotation too, not just genuine
parallax, and GRIC's d/k complexity-penalty terms (which favor H by only ~n*log(4) while E's inherently weaker
1-dim constraint gives it an O(n) raw-residual head start) were not enough to overcome that structurally-optimistic
E fit. Measured: GRIC parallax_score gave approach≈0.8-1.3, zoom≈0.7-1.4, rotation≈0.6-1.1 — NO reliable
separation (see the g-run in the report). The H-alone GOF test above sidesteps this entirely: it never needs E to
fit at all, so it cannot inherit E's rotation-degenerate optimism. E is still computed here as a SECONDARY,
non-load-bearing diagnostic (reported per pair) for audit/transparency, never gating the verdict.

★HONEST SCOPE / CONFOUND (state, do not hide): a genuinely PLANAR scene under APPROACH also fits H exactly (no
parallax is observable from a plane regardless of translation) => this pre-gate ABSTAINs on planar-scene-under-
translation too. That is ARGUABLY CORRECT, not a bug: relative depth truly is unrecoverable (a gauge) from a planar
scene under monocular translation alone (the classical planar-degenerate SfM case) — the pre-gate cannot and does
not claim to distinguish "no parallax because zoom/rotation" from "no parallax because the observed scene points
happen to be coplanar." Both are honestly ABSTAIN under this observable. A caller with an independent planarity
signal (e.g. known non-planar CAD-derived scene) may override; this module does not assume it.

★HONEST SNR FLOOR (measured, force-the-negative — see selftest GATE 2): parallax_score is an SNR statistic — for a
FIXED real baseline, the achievable separation from the zoom/rotation ceiling (~1.4-2.6, stable across noise) SHRINKS
as pixel noise grows, because a fixed real-world parallax signal becomes harder to detect against a higher noise
floor (this is physics, not a bug — same story as 's tiny-baseline continuum, but along the noise axis instead
of the baseline axis). Calibrated on the canonical non-planar scene (80-200 pts, depth 3-9, K=diag(800,800),
lateral-dominant baseline |t|~1.4): verdicts are STABLE across the FULL requested 0.5->8px sweep for this baseline
(margin >=1.2x at worst, px_noise=8). A SMALLER real baseline collapses into the noise floor at a LOWER px_noise
(see report) — exactly the same continuum logic as GATE 3, now along the noise axis.

null-safe (mirrors the video_admission_cert): too few frames/correspondences, non-finite input, degenerate K, or a
scene with 0 valid pairs => cert_valid=False, verdict=ABSTAIN — NEVER a false ADMIT.

gate: PYTHONPATH=src OMP_NUM_THREADS=4 nice -10python src/graph_engine/video_admission_identifiability_pregate.py
cv2 + numpy only.
"""
from __future__ import annotations
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - environment without cv2
    cv2 = None

MIN_FRAMES = 2
MIN_CORR = 8                 # need a real sample for a stable robust (median) residual + RANSAC to be meaningful.
RANSAC_H_PX_MULT = 3.0       # H RANSAC inlier threshold = RANSAC_H_PX_MULT * px_noise (generous: outliers, not the
                              # parallax signal itself, should be rejected; the GOF test runs over ALL correspondences)
RANSAC_E_PX_MULT = 1.5       # E RANSAC inlier threshold (secondary diagnostic fit only)
RANSAC_CONF = 0.999
CHI2_2_MEDIAN = 2.0 * np.log(2.0)   # ≈1.3863: median of a chi-square(2) distribution (the H0 "H explains it" null)
# Calibrated on the synthetic non-planar-scene sweep in selftest (see report for exact measured margins): the
# zoom/rotation ceiling sits at ~1.4-2.6 (n_pts in [80,200], px_noise in [0.5,8], multi-seed); a real (non-tiny)
# approach baseline sits >=3.7 at worst (px_noise=8, weakest seed). THRESH_PARALLAX sits with margin on both sides.
THRESH_PARALLAX = 3.0


def _gric(resid2, sigma, n, d, k, r=4.0):
    """Torr's GRIC — SECONDARY diagnostic only (see module docstring for why it is not the decision variable: it
    cannot discriminate PURE ROTATION, whose essential-matrix fit is EXACTLY degenerate-optimistic)."""
    cap = 2.0 * (r - d)
    rho = np.minimum(resid2 / (sigma ** 2), cap)
    lam1 = np.log(r)
    lam2 = np.log(r * max(n, 1))
    return float(rho.sum() + lam1 * d * n + lam2 * k)


def _transfer_sq(H, pts0, pts1):
    """Forward-transfer squared error (pixels^2) for a homography — a genuine 2-dim residual (x,y free, x'=Hx
    determined). This IS the primary decision statistic's raw ingredient."""
    ones = np.ones((pts0.shape[0], 1))
    p0h = np.hstack([pts0, ones])
    fwd = (H @ p0h.T).T; fwd = fwd[:, :2] / np.where(np.abs(fwd[:, 2:3]) > 1e-12, fwd[:, 2:3], 1e-12)
    return np.sum((fwd - pts1) ** 2, axis=1)


def _sampson_sq(F, pts0, pts1):
    """Sampson distance squared (pixels^2) for a fundamental/essential matrix — SECONDARY diagnostic only."""
    ones = np.ones((pts0.shape[0], 1))
    p0h = np.hstack([pts0, ones]); p1h = np.hstack([pts1, ones])
    Fx0 = (F @ p0h.T).T
    Ftx1 = (F.T @ p1h.T).T
    num = np.einsum('ij,ij->i', p1h, (F @ p0h.T).T) ** 2
    denom = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    denom = np.where(denom > 1e-12, denom, 1e-12)
    return num / denom


def _pair_diagnostic(pts0, pts1, K, px_noise):
    """Per consecutive-frame-pair diagnostic. PRIMARY: h_gof_ratio (H goodness-of-fit vs the noise-floor null).
    SECONDARY (non-load-bearing, audit-only): GRIC(H) vs GRIC(E), if E happens to fit."""
    n = pts0.shape[0]
    diag = dict(n_corr=int(n), ok=False)
    if n < MIN_CORR:
        diag["skip_reason"] = f"insufficient_correspondences ({n}<{MIN_CORR})"
        return diag
    if np.std(pts0, axis=0).min() < 1e-6 or np.std(pts1, axis=0).min() < 1e-6:
        diag["skip_reason"] = "degenerate: correspondences collapse to a point/line"
        return diag
    try:
        H, maskH = cv2.findHomography(pts0, pts1, cv2.RANSAC, RANSAC_H_PX_MULT * px_noise, confidence=RANSAC_CONF)
    except cv2.error as e:
        diag["skip_reason"] = f"cv2_error fitting H: {e}"
        return diag
    if H is None or not np.all(np.isfinite(H)):
        diag["skip_reason"] = "RANSAC homography fit failed (None/non-finite H)"
        return diag
    resid_H = _transfer_sq(H, pts0, pts1)
    if not np.all(np.isfinite(resid_H)):
        diag["skip_reason"] = "non-finite H residuals"
        return diag
    h_gof_ratio = float(np.median(resid_H) / (CHI2_2_MEDIAN * px_noise ** 2))
    diag.update(ok=True, h_gof_ratio=h_gof_ratio,
                median_resid_H_px=float(np.sqrt(np.median(resid_H))),
                mean_resid_H_px2=float(np.mean(resid_H)),
                inlier_frac_H=float(maskH.sum() / n) if maskH is not None else float("nan"))
    # ---- secondary diagnostic: E / GRIC, best-effort, never blocks the primary decision ----
    try:
        E, maskE = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=RANSAC_CONF,
                                         threshold=RANSAC_E_PX_MULT * px_noise)
        if E is not None and E.shape == (3, 3) and np.all(np.isfinite(E)):
            Kinv = np.linalg.inv(K)
            F = Kinv.T @ E @ Kinv
            resid_E = _sampson_sq(F, pts0, pts1)
            if np.all(np.isfinite(resid_E)):
                gric_h = _gric(resid_H, px_noise, n, d=2, k=8)
                gric_e = _gric(resid_E, px_noise, n, d=3, k=5)
                diag.update(gric_H=gric_h, gric_E=gric_e, gric_parallax_secondary=(gric_h - gric_e) / n,
                            median_resid_E_px=float(np.sqrt(np.median(resid_E))),
                            inlier_frac_E=float(maskE.sum() / n) if maskE is not None else float("nan"))
    except cv2.error:
        pass  # secondary diagnostic only; E failing to fit (e.g. genuinely degenerate t=0) is not an error
    return diag


def depth_recoverability_pregate(tracks, K, dist=None, px_noise=1.5, min_corr=MIN_CORR,
                                  thresh_parallax=THRESH_PARALLAX):
    """The deploy-time depth-gauge pre-gate: ADMIT (parallax_recoverable) vs ABSTAIN (depth_gauge: no_parallax),
    decided from tracks alone.

    tracks: array-like (F, N, 2) pixel coordinates, F>=2 frames, N tracked points. np.nan marks a point not
             visible in a given frame (occlusion-tolerant). Consecutive-frame pairs (i, i+1) are matched by the
             finite mask; a track need not be visible in every frame.
    K: (3,3) camera intrinsic matrix (pixels). Must be finite and invertible.
    dist: optional cv2-format distortion coeffs; if given, tracks are undistorted (cv2.undistortPoints, P=K)
             before use.
    px_noise: assumed per-coordinate pixel noise std, sets the RANSAC inlier threshold and the GOF-test null scale.

    Returns dict(verdict, reason, parallax_score, per_pair, n_pairs_used, cert_valid).
    NULL-SAFE: malformed input / <2 frames / no pair with >=min_corr correspondences / degenerate K =>
    cert_valid=False, verdict=ABSTAIN (never a false ADMIT)."""
    def _abstain_invalid(reason, per_pair=None):
        return dict(verdict="ABSTAIN", reason=f"cert_invalid: {reason}", parallax_score=float("nan"),
                    per_pair=per_pair or [], n_pairs_used=0, cert_valid=False)

    if cv2 is None:
        return _abstain_invalid("cv2 not available in this environment")
    try:
        tr = np.asarray(tracks, dtype=float)
        Km = np.asarray(K, dtype=float)
    except Exception as e:
        return _abstain_invalid(f"non-numeric input ({e})")
    if tr.ndim != 3 or tr.shape[2] != 2 or tr.shape[0] < MIN_FRAMES or tr.shape[1] < min_corr:
        return _abstain_invalid(f"malformed tracks shape {tr.shape} (need (F>=2, N>={min_corr}, 2))")
    if Km.shape != (3, 3) or not np.all(np.isfinite(Km)):
        return _abstain_invalid(f"malformed/non-finite K shape {Km.shape}")
    if abs(np.linalg.det(Km)) < 1e-9 or Km[0, 0] <= 0 or Km[1, 1] <= 0:
        return _abstain_invalid("K is non-invertible or has non-positive focal length")
    if not np.isfinite(px_noise) or px_noise <= 0:
        return _abstain_invalid(f"invalid px_noise={px_noise}")

    F_frames = tr.shape[0]
    if dist is not None:
        try:
            distv = np.asarray(dist, dtype=float)
            flat = tr.reshape(-1, 1, 2)
            und = cv2.undistortPoints(flat.astype(np.float64), Km, distv, P=Km)
            tr = und.reshape(F_frames, -1, 2)
        except (cv2.error, ValueError, TypeError) as e:
            return _abstain_invalid(f"undistortPoints failed ({e})")

    per_pair = []
    scores = []
    for i in range(F_frames - 1):
        a, b = tr[i], tr[i + 1]
        valid = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
        pts0, pts1 = a[valid].astype(np.float64), b[valid].astype(np.float64)
        d = _pair_diagnostic(pts0, pts1, Km, px_noise)
        d["pair"] = (i, i + 1)
        per_pair.append(d)
        if d["ok"]:
            scores.append(d["h_gof_ratio"])

    if not scores:
        return _abstain_invalid("no consecutive-frame pair had >=min_corr usable correspondences", per_pair)

    parallax_score = float(np.median(scores))
    n_pairs_used = len(scores)
    if parallax_score > thresh_parallax:
        verdict = "ADMIT"
        reason = (f"parallax_recoverable: median H-goodness-of-fit ratio={parallax_score:.2f} > {thresh_parallax} "
                  f"over {n_pairs_used} pair(s) — a single homography is REJECTED at the assumed px_noise level; "
                  "depth recoverable up to overall scale (mirrors g1125 nullity=1)")
    else:
        verdict = "ABSTAIN"
        reason = (f"depth_gauge: no_parallax — median H-goodness-of-fit ratio={parallax_score:.2f} <= "
                  f"{thresh_parallax} over {n_pairs_used} pair(s); a homography (planar-equivalent warp, or the "
                  "zoom/infinite-homography degenerate case) explains the correspondences within the assumed pixel "
                  "noise — depth structure is a gauge (mirrors g1125 nullity=N+1/N). NOTE: this is also the honest "
                  "signature of a genuinely planar scene under translation (see module docstring scope) — this "
                  "pre-gate cannot and does not distinguish the two.")
    return dict(verdict=verdict, reason=reason, parallax_score=parallax_score, per_pair=per_pair,
                n_pairs_used=n_pairs_used, cert_valid=True)


def as_veto_leg(result, name="depth_recoverability_pregate"):
    """Adapt depth_recoverability_pregate's result into the SHARED veto_leg contract (see
    video_admission_cert.py: {"name","cert_valid","veto","reason"}) for direct pluggable composition — a lane's
    cert runs standalone and the caller appends veto_leg(**this) to video_admission_cert(..., veto_legs=[...]).
    veto=True only for a REAL depth-gauge finding (cert_valid=True, verdict=ABSTAIN) — a null-safety ABSTAIN
    (cert_valid=False, e.g. too few tracks) is reported as cert_valid=False/veto=False (leg couldn't run; not a
    veto), matching D's "cert_invalid is not a veto but is reported" convention exactly."""
    veto = bool(result["cert_valid"] and result["verdict"] == "ABSTAIN")
    return {"name": str(name), "cert_valid": bool(result["cert_valid"]), "veto": veto, "reason": result["reason"]}


# ───────────────────────────── synthetic non-planar-scene track generator (selftest only) ─────────────────────────
def _make_scene(n_pts=80, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(3.0, 9.0, n_pts)                       # varied depth => genuinely non-planar
    X = rng.uniform(-2.5, 2.5, n_pts) * (Z / 5.0)
    Y = rng.uniform(-2.0, 2.0, n_pts) * (Z / 5.0)
    return np.column_stack([X, Y, Z])


def _K(f=800.0, cx=320.0, cy=240.0):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def _rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def _project(pts3d, Kmat, R=None, t=None, img_w=640, img_h=480):
    R = np.eye(3) if R is None else R
    t = np.zeros(3) if t is None else t
    Xc = pts3d @ R.T + t[None, :]
    uvw = (Kmat @ Xc.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    inb = (uvw[:, 2] > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    out = np.full_like(uv, np.nan)
    out[inb] = uv[inb]
    return out


# canonical "approach" baseline direction: DELIBERATELY lateral-dominant (not pure axial dolly). A pure forward-only
# dolly is the classical "dolly-zoom" hard case (forward translation's radial-expansion flow is, for modest depth
# variation, close to a homography-explainable scaling) — genuinely harder to separate from zoom at fixed noise;
# see selftest's honest axial-dolly diagnostic. Real handheld/drone "approach" footage is rarely purely axial.
_APPROACH_T_DIR = np.array([0.5, 0.35, 0.3])


def _make_tracks(motion, n_frames, n_pts, px_noise, seed, baseline_scale=1.0, t_dir=None):
    """motion in {'approach','zoom','rotation','axial_dolly'}. Returns (tracks (F,N,2), K_used_for_gate)."""
    pts3d = _make_scene(n_pts, seed=seed)
    Kmat = _K()
    rng = np.random.default_rng(seed + 1000)
    t_dir = _APPROACH_T_DIR if t_dir is None else np.asarray(t_dir, dtype=float)
    frames = []
    for i in range(n_frames):
        s = i / max(n_frames - 1, 1)
        if motion == "approach":
            t = t_dir * s * baseline_scale
            uv = _project(pts3d, Kmat, R=np.eye(3), t=t)
        elif motion == "axial_dolly":
            t = np.array([0.0, 0.0, 0.9]) * s * baseline_scale
            uv = _project(pts3d, Kmat, R=np.eye(3), t=t)
        elif motion == "zoom":
            f_i = 800.0 * (1.0 + 0.5 * s)
            uv = _project(pts3d, _K(f=f_i), R=np.eye(3), t=np.zeros(3))
        elif motion == "rotation":
            w = np.array([0.02, 0.25, 0.01]) * s
            uv = _project(pts3d, Kmat, R=_rodrigues(w), t=np.zeros(3))
        else:
            raise ValueError(motion)
        uv = uv + rng.normal(0, px_noise, uv.shape)
        frames.append(uv)
    return np.stack(frames, axis=0), Kmat


def selftest():
    if cv2 is None:
        print("SKIP: cv2 not available"); return 1
    fails = 0
    n_pts, n_frames = 80, 5
    APPROACH_BASELINE = 2.5   # |t| ~= [1.25, 0.875, 0.75] scene-units at full baseline (depth range 3-9)

    # ---- GATE 1: canonical 3-way verdicts at a nominal noise level ----
    px = 1.5
    results = {}
    for motion in ("approach", "zoom", "rotation"):
        tr, Kmat = _make_tracks(motion, n_frames, n_pts, px, seed=0, baseline_scale=APPROACH_BASELINE)
        r = depth_recoverability_pregate(tr, Kmat, px_noise=px)
        results[motion] = r
    if not (results["approach"]["verdict"] == "ADMIT" and results["zoom"]["verdict"] == "ABSTAIN"
            and results["rotation"]["verdict"] == "ABSTAIN"):
        print(f"FAIL gate1: verdicts = {[(m, results[m]['verdict']) for m in results]}"); fails += 1
    print("GATE 1 (canonical verdicts @ px_noise=1.5):")
    for m in ("approach", "zoom", "rotation"):
        r = results[m]
        print(f"  {m:9s}: verdict={r['verdict']:7s} parallax_score={r['parallax_score']:8.2f} "
              f"cert_valid={r['cert_valid']} n_pairs={r['n_pairs_used']}")

    # ---- GATE 2: noise sweep 0.5->8px, verdicts STABLE + report separation margins per level; find collapse ----
    noise_levels = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    sweep = {}
    for px_n in noise_levels:
        row = {}
        for motion in ("approach", "zoom", "rotation"):
            tr, Kmat = _make_tracks(motion, n_frames, n_pts, px_n, seed=1, baseline_scale=APPROACH_BASELINE)
            r = depth_recoverability_pregate(tr, Kmat, px_noise=px_n)
            row[motion] = (r["verdict"], r["parallax_score"])
        sweep[px_n] = row
    print("\nGATE 2 (noise sweep, parallax_score, this baseline):")
    stable = True
    collapse_at = None
    for px_n in noise_levels:
        row = sweep[px_n]
        print(f"  px_noise={px_n:4.1f}: approach={row['approach'][1]:8.2f}({row['approach'][0]}) "
              f"zoom={row['zoom'][1]:7.2f}({row['zoom'][0]}) rotation={row['rotation'][1]:7.2f}({row['rotation'][0]})")
        ok_row = (row["approach"][0] == "ADMIT" and row["zoom"][0] == "ABSTAIN" and row["rotation"][0] == "ABSTAIN")
        if not ok_row:
            stable = False
            if collapse_at is None:
                collapse_at = px_n
    if not stable:
        print(f"FAIL gate2: verdicts not stable across noise sweep; first collapse at px_noise={collapse_at}")
        fails += 1
    else:
        print(f"  -> STABLE across the full 0.5-8px sweep for this baseline (|t|~{APPROACH_BASELINE * np.linalg.norm(_APPROACH_T_DIR):.2f}).")

    # ---- GATE 3: tiny-baseline approach is a CONTINUUM (monotone-ish decay), not an erratic flip ----
    baseline_scales = [2.5, 0.75, 0.25, 0.075, 0.025, 0.0075, 0.0025]
    cont = []
    for bs in baseline_scales:
        tr, Kmat = _make_tracks("approach", n_frames, n_pts, px_noise=1.5, seed=2, baseline_scale=bs)
        r = depth_recoverability_pregate(tr, Kmat, px_noise=1.5)
        cont.append((bs, r["parallax_score"], r["verdict"]))
    print("\nGATE 3 (tiny-baseline approach continuum, px_noise=1.5):")
    for bs, score, v in cont:
        print(f"  baseline_scale={bs:7.4f} (|t|={bs * np.linalg.norm(_APPROACH_T_DIR):6.3f}): "
              f"parallax_score={score:8.2f} -> {v}")
    scores_only = [c[1] for c in cont]
    decreasing_overall = scores_only[0] > scores_only[-1] + 3.0
    diffs = np.diff(scores_only)
    n_upticks = int((diffs > 0.5 * abs(np.median(np.abs(diffs)) + 1e-9)).sum())
    erratic = n_upticks > 2
    if not decreasing_overall or erratic:
        print(f"FAIL gate3: continuum not monotone-decreasing enough (decreasing_overall={decreasing_overall}, "
              f"n_upticks={n_upticks})")
        fails += 1

    # ---- GATE 4: null-safety — never a false ADMIT on degenerate/malformed input ----
    Kmat = _K()
    null_cases = []
    r = depth_recoverability_pregate(np.zeros((1, 20, 2)), Kmat, px_noise=1.5)
    null_cases.append(("1 frame", r))
    r = depth_recoverability_pregate(np.zeros((3, 4, 2)), Kmat, px_noise=1.5)
    null_cases.append(("4 points < min_corr", r))
    tr_bad, _ = _make_tracks("approach", n_frames, n_pts, 1.5, seed=0)
    tr_bad[:, :] = np.nan
    r = depth_recoverability_pregate(tr_bad, Kmat, px_noise=1.5)
    null_cases.append(("all-nan tracks", r))
    tr_ok, _ = _make_tracks("approach", n_frames, n_pts, 1.5, seed=0)
    r = depth_recoverability_pregate(tr_ok, np.zeros((3, 3)), px_noise=1.5)
    null_cases.append(("singular K", r))
    static = np.tile(_project(_make_scene(n_pts, seed=0), Kmat)[None, :, :], (n_frames, 1, 1))
    r = depth_recoverability_pregate(static, Kmat, px_noise=1.5)
    null_cases.append(("static/zero-motion tracks", r))
    print("\nGATE 4 (null-safety, never false ADMIT):")
    null_ok = True
    for name, r in null_cases:
        bad = (r["verdict"] == "ADMIT")
        null_ok = null_ok and not bad
        print(f"  {name:32s}: verdict={r['verdict']:7s} cert_valid={r['cert_valid']} {'<-- FALSE ADMIT!' if bad else ''}")
    if not null_ok:
        print("FAIL gate4: a degenerate/malformed case produced a false ADMIT"); fails += 1

    # ---- HONEST DIAGNOSTIC (not a pass/fail gate): pure-axial dolly is a genuinely harder case (documented scope) ----
    print("\nDIAGNOSTIC (honest, not gated): pure axial dolly vs the zoom/rotation ceiling")
    for px_n in (0.5, 1.5, 3.0):
        tr, Kmat = _make_tracks("axial_dolly", n_frames, n_pts, px_n, seed=3, baseline_scale=1.0)
        r = depth_recoverability_pregate(tr, Kmat, px_noise=px_n)
        print(f"  px_noise={px_n}: axial_dolly parallax_score={r['parallax_score']:.2f} -> {r['verdict']} "
              "(a pure forward-only dolly is closer to the zoom degenerate case at this depth range/baseline; "
              "see module docstring scope note)")

    n = 4
    print(f"\nvideo_admission_identifiability_pregate selftest: {n - fails}/{n} passed")
    print(f"  THRESH_PARALLAX={THRESH_PARALLAX} (calibrated: zoom/rotation ceiling ~1.4-2.6 vs approach floor "
          f">=3.7 at worst noise, this baseline — see GATE 2 numbers above)")
    return fails


if __name__ == "__main__":
    raise SystemExit(selftest())
