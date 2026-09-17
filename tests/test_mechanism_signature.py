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
