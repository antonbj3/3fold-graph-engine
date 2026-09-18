"""mechanism_signature on textbook models with known answers."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from graph_engine.mechanism_signature import signature, distance  # noqa: E402

MODELS = {   # F(x, mu), x-range, mu-bracket, field
    "electrostatic pull-in (MEMS)":   (lambda u, m: m - u * (1 - u) ** 2,              (0.02, 0.60), (-1, 1),   "microsystems"),     # fold at u=1/3, mu=4/27
    "thermal runaway (Semenov)":      (lambda t, p: p * np.exp(t) - t,                 (0.05, 2.50), (1e-9, 5), "combustion"),       # fold at theta=1, psi=1/e
    "snap-through (shallow truss)":   (lambda u, P: P - u * (u - 1) * (u - 2),         (0.02, 0.95), (-3, 3),   "structures"),       # limit load at u=1-1/sqrt(3)
    "symmetric column (pitchfork)":   (lambda x, m: x * (m - 1 - x * x),               (-0.8, 0.8),  (-3, 5),   "structures"),
    "cusp":                           (lambda x, m: m - x ** 3,                        (-0.8, 0.8),  (-2, 2),   "generic"),
    "linear spring":                  (lambda x, m: m - 2 * x,                         (0.02, 0.9),  (-5, 5),   "structures"),
    "saturating response":            (lambda x, m: m - x / (1 - x),                   (0.02, 0.9),  (-1, 50),  "chemistry"),
}
SIG = {k: signature(F, xr, mb) for k, (F, xr, mb, _) in MODELS.items()}
FOLDS = ["electrostatic pull-in (MEMS)", "thermal runaway (Semenov)", "snap-through (shallow truss)"]


def test_known_limit_points_and_exponents():
    assert np.isclose(SIG[FOLDS[0]].x_c, 1 / 3, atol=1e-3) and np.isclose(SIG[FOLDS[0]].mu_c, 4 / 27, atol=1e-4)
    assert np.isclose(SIG[FOLDS[1]].x_c, 1.0, atol=2e-3) and np.isclose(SIG[FOLDS[1]].mu_c, np.exp(-1), atol=1e-4)
    assert np.isclose(SIG[FOLDS[2]].x_c, 1 - 1 / np.sqrt(3), atol=1e-3)
    for k in FOLDS:
        s = SIG[k]; assert s.n_limit == 1 and not s.odd and abs(s.order - 2) < 0.05 and abs(s.beta - 0.5) < 0.02 and abs(s.gamma - 0.5) < 0.03


def test_three_fields_one_mechanism_and_the_others_are_not_it():
    for a in FOLDS:
        for b in FOLDS: assert distance(SIG[a], SIG[b]) < 0.1
        for other in ["symmetric column (pitchfork)", "cusp", "linear spring", "saturating response"]:
            assert distance(SIG[a], SIG[other]) >= 0.9
    assert SIG["symmetric column (pitchfork)"].odd and SIG["linear spring"].n_limit == 0 and SIG["saturating response"].n_limit == 0 and SIG["cusp"].n_limit == 0
    assert len({MODELS[k][3] for k in FOLDS}) == 3                           # three different subject fields


def test_oddness_is_a_property_of_F_not_of_its_branch_and_bad_brackets_fail_loudly():
    import pytest
    s = signature(lambda x, m: (m - x * x) * x + (m - x * x) ** 2, (-0.8, 0.8), (0.0, 1.0))      # branch m = x² is symmetric, F is not odd
    assert not s.odd
    with pytest.raises(ValueError):
        signature(lambda x, m: 1.0 + x * x + m * m, (0.1, 0.9), (0.0, 1.0))                       # never zero


def test_vector_state_signature_agrees_with_scalar_and_reads_a_coupled_fold():
    from graph_engine.mechanism_signature import signature_vector
    # the scalar pull-in model, traced as a 1-vector
    s1 = signature_vector(lambda u, m: np.array([m - u[0] * (1 - u[0]) ** 2]), np.array([0.02]), 0.01, ds=0.004, mu_stop=(-1, 1))
    assert s1.n_limit >= 1 and abs(s1.mu_c - 4 / 27) < 2e-3 and abs(s1.order - 2) < 0.1 and abs(s1.gamma - 0.5) < 0.05
    assert s1.converged
    # GENUINELY coupled two-state fold: u' = mu - u^2 + 0.3 w, w' = u - w. The equilibrium has w = u (both states move),
    # so mu = u^2 - 0.3u; dmu/du = 2u - 0.3 = 0 at u_c = 0.15, mu_c = 0.15^2 - 0.3*0.15 = -0.0225, and the singular
    # Jacobian [[-2u, 0.3], [1, -1]] has null vector (1, 1): the fold lives in BOTH states, not in a scalar subspace.
    # (The previous model's second equation -w(k+u) = 0 was satisfied by w = 0 identically, so the trace never left w = 0.)
    def coupled(v, mu):
        u, w = v
        return np.array([mu - u * u + 0.3 * w, u - w])
    u_c, mu_c = 0.15, 0.15 ** 2 - 0.3 * 0.15
    s2 = signature_vector(coupled, np.array([1.0, 1.0]), 0.7, ds=0.01, mu_stop=(-1, 1), direction=-1)
    assert s2.n_limit >= 1 and s2.converged and abs(s2.mu_c - mu_c) < 1e-5 and abs(mu_c + 0.0225) < 1e-15
    assert abs(s2.x_c - np.hypot(u_c, u_c)) < 2e-2                       # |x| at the fold: both components are 0.15
    assert abs(s2.order - 2) < 0.1 and abs(s2.gamma - 0.5) < 0.05 and not s2.odd


def test_a_sharp_fold_is_not_reported_as_no_fold_the_step_is_refined_until_it_repeats():
    """Same fold, steeper scale: at ds = 0.02 the continuation steps over it and reads n_limit = 0 — indistinguishable
    from a fold-free model unless the step is refined. The adaptive tracer halves ds until the count AND the exponents
    repeat, and says so (converged)."""
    from graph_engine.mechanism_signature import signature_vector
    a = 20.0                                              # mu = a u^2 - 0.3u: fold at u_c = 0.15/a, mu_c = -0.0225/a
    def sharp(v, mu):
        u, w = v
        return np.array([mu - a * u * u + 0.3 * w, u - w])
    u_c = 0.15 / a; mu_c = -0.0225 / a; u0 = 4 * u_c; mu0 = a * u0 * u0 - 0.3 * u0
    coarse = signature_vector(sharp, np.array([u0, u0]), mu0, ds=0.02, n_steps=4000, mu_stop=(-1, mu0), direction=-1, adaptive=False)
    assert coarse.n_limit == 0                            # the fold is MISSED at ds = 0.02 (the bug: silently a clean zero)
    s = signature_vector(sharp, np.array([u0, u0]), mu0, ds=0.02, n_steps=4000, mu_stop=(-1, mu0), direction=-1)
    assert s.n_limit == 1 and s.converged
    assert abs(s.mu_c - mu_c) < 1e-6 and abs(s.order - 2) < 0.1 and abs(s.gamma - 0.5) < 0.05
    # a fold-free model stays n_limit = 0 under the same refinement, and says converged: the two cases are distinguishable
    flat = signature_vector(lambda v, mu: np.array([mu - 2 * v[0] + 0.3 * v[1], v[0] - v[1]]), np.array([0.0, 0.0]), 0.0,
                            ds=0.02, n_steps=200, mu_stop=(-1, 1))
    assert flat.n_limit == 0 and flat.converged
    assert distance(s, flat) == 4.0 and distance(s, s) == 0.0           # distance() unchanged by the new field


def test_a_branch_stopped_by_a_failed_corrector_step_is_flagged_not_reported_as_fold_free():
    """F undefined past u = 0.5: the corrector step fails (fsolve ier != 1 / non-finite Jacobian). The branch stops there,
    and 'no limit point in what was traced' is returned with converged = False — not as a clean fold-free answer."""
    import warnings
    from graph_engine.mechanism_signature import signature_vector
    def half_defined(v, mu):
        u, w = v
        return np.array([mu - np.sqrt(0.5 - u) + 0.3 * w, u - w])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); old = np.seterr(all="ignore")
        s = signature_vector(half_defined, np.array([0.0, 0.0]), np.sqrt(0.5), ds=0.02, n_steps=500, mu_stop=(-2, 2), direction=-1)
        np.seterr(**old)
    assert s.n_limit == 0 and not s.converged


def test_eleven_named_phenomena_group_by_normal_form_blind_to_names():
    import json, subprocess, sys as _s
    from pathlib import Path as _P
    r = _P(__file__).resolve().parent.parent / "examples" / "engine_experiments"
    subprocess.run([_s.executable, str(r / "e17_same_thing_different_names.py")], check=True, capture_output=True)
    out = json.load(open(r / "e17_results.json"))
    assert out["every_group_is_one_normal_form"]
    fold = next(g for g in out["groups_by_computed_signature"] if g["normal_forms"] == ["fold"])
    assert len(fold["members"]) == 5 and len(fold["fields"]) == 5
