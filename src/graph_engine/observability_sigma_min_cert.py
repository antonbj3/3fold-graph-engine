"""observability_sigma_min_cert — a reusable LINEAR-IDENTIFIABILITY cert (G's σ_min/observability axis, productized from
-). Given a stacked measurement Jacobian J (m×p) for a linear/linearized parameter vector θ, the SMALLEST singular
value σ_min of J is the identifiability FLOOR: σ_min ≈ 0 ⇔ some parameter COMBINATION is unobservable (confounded), and the
corresponding right singular vector NAMES that combination (the null direction). This is the object behind the pool's
'σ_min-margin predicts per-view error' certs (B Belysning G2, G  gravity↔bias, kinematic-singularity).

TWO honest disciplines baked in from -:
  1. CONFIRM the mechanism by the NULL DIRECTION, not just the σ_min magnitude — a perfect null-direction match to a
     PREDICTED confound is a stronger, magnitude-independent certificate (|cos|=1.000 for db_a=R^T dg).
  2. When real data doesn't span the singular EXTREME (easy trajectory / low excitation range), use a FROZEN-EXCITATION
     COUNTERFACTUAL to expose the clean confounded limit the data only approaches (frozen-R σ_min=6e-17 vs real lift).

★The σ_min→error SENSITIVITY exponent (error ~ σ_min^-k) is an ESTIMATOR-REGIME property, NOT substrate-universal physics
: k≈1 for a pure CRB/unbounded estimand, k<1 for a bounded/regularized one. Do not read a cross-substrate 'k
transfers' as a physical constant. Pure numpy.
"""
from __future__ import annotations
import numpy as np

__all__ = ["observability_sigma_min", "null_direction_matches", "confounded_limit", "selftest"]


def observability_sigma_min(J, normalize=True):
    """Identifiability floor of a stacked measurement Jacobian J (m×p) for parameter vector θ.
    Returns {sigma_min, sigma_max, cond, null_direction (p-vector, the least-observable θ-combination), singular_values}.
    normalize=True divides J by sqrt(#measurement-rows) for a per-measurement scale (compare σ_min across window sizes).
    σ_min ≈ 0 ⇒ the null_direction combination is unobservable/confounded; report the DIRECTION, it names the confound."""
    J = np.asarray(J, float)
    if J.ndim != 2 or J.shape[0] < J.shape[1]:
        raise ValueError(f"J must be (m>=p) 2-D; got {J.shape}")
    scale = np.sqrt(J.shape[0]) if normalize else 1.0
    sv = np.linalg.svd(J / scale, compute_uv=False)
    Vt = np.linalg.svd(J / scale, compute_uv=True)[2]
    return dict(sigma_min=float(sv[-1]), sigma_max=float(sv[0]),
                cond=float(sv[0] / (sv[-1] + 1e-300)),
                null_direction=Vt[-1].copy(), singular_values=sv.copy())


def null_direction_matches(null_dir, predicted_dir):
    """|cos| between the measured unobservable direction and a PREDICTED confound direction (both p-vectors). ~1.0 = the
    mechanism is confirmed exactly (magnitude-independent certificate). Returns |cos| in [0,1]."""
    a = np.asarray(null_dir, float); b = np.asarray(predicted_dir, float)
    return float(abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def confounded_limit(J_real, J_frozen, normalize=True):
    """Frozen-excitation counterfactual: compare σ_min(real Jacobian) to σ_min(Jacobian with the excitation variable
    frozen to a constant). Exposes the clean confounded limit (σ_min→0) that a low-excitation real dataset only approaches.
    Returns {sigma_min_real, sigma_min_frozen, lift_ratio}. lift_ratio ≫ 1 ⇒ real excitation lifts σ_min off the confounded
    floor (the identifiability lever is present in the data)."""
    sr = observability_sigma_min(J_real, normalize)["sigma_min"]
    sf = observability_sigma_min(J_frozen, normalize)["sigma_min"]
    return dict(sigma_min_real=sr, sigma_min_frozen=sf, lift_ratio=float(sr / (sf + 1e-300)))


def _R_from_axis_angle(axis, ang):
    ax = np.asarray(axis, float); ax = ax / (np.linalg.norm(ax) + 1e-300)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def selftest() -> int:
    """Validate on the gravity↔accel-bias observability object: specific-force Jacobian J_t=[-R_t^T | I] for
    θ=(g, b_a). No rotation ⇒ σ_min=0 + null-direction = (δg, R^T δg); rotation ⇒ σ_min>0 (observable)."""
    ok = True
    print("observability_sigma_min_cert selftest (gravity↔accel-bias observability, G17-G19):")

    def build_J(Rs):
        m = len(Rs); J = np.zeros((3 * m, 6))
        for i, R in enumerate(Rs):
            J[3 * i:3 * i + 3, 0:3] = -R.T
            J[3 * i:3 * i + 3, 3:6] = np.eye(3)
        return J

    R0 = _R_from_axis_angle([0.3, 0.4, 0.5], 0.7)
    # (1) NO rotation: constant R -> sigma_min = 0 (confounded), null = (dg, R0^T dg)
    J_static = build_J([R0] * 40)
    r_static = observability_sigma_min(J_static)
    c1 = bool(r_static["sigma_min"] < 1e-9)
    nd = r_static["null_direction"]; dg, dba = nd[:3], nd[3:]
    cos = null_direction_matches(dba, R0.T @ dg)
    c1b = bool(cos > 0.99)
    ok &= c1 and c1b
    print(f"  [{'✓' if c1 and c1b else '✗'}] no-rotation: σ_min={r_static['sigma_min']:.2e} (~0 confounded); "
          f"null-direction = predicted db_a=R^T dg |cos|={cos:.3f}")

    # (2) rotation: varying R -> sigma_min > 0 (observable)
    rng = np.random.default_rng(0)
    Rs_rot = [_R_from_axis_angle(rng.standard_normal(3), a) for a in np.linspace(0, 2.0, 40)]
    r_rot = observability_sigma_min(build_J(Rs_rot))
    c2 = bool(r_rot["sigma_min"] > 0.1)
    ok &= c2
    print(f"  [{'✓' if c2 else '✗'}] rotation: σ_min={r_rot['sigma_min']:.3f} (>0 observable), cond={r_rot['cond']:.1f}")

    # (3) frozen-excitation counterfactual: real (rotating) lifts σ_min far off the frozen (static) confounded limit
    cl = confounded_limit(build_J(Rs_rot), J_static)
    c3 = bool(cl["lift_ratio"] > 1e6)
    ok &= c3
    print(f"  [{'✓' if c3 else '✗'}] frozen-excitation limit: real σ_min {cl['sigma_min_real']:.3f} vs frozen "
          f"{cl['sigma_min_frozen']:.1e} (lift {cl['lift_ratio']:.0e})")

    print(f"\n  SELFTEST observability_sigma_min_cert: {'✓ ALL PASS — reusable linear-identifiability floor (σ_min + null-direction + frozen-counterfactual)' if ok else '✗ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
