"""closed_loop: the world's sign structure, the truth-based score, lineage weighting of copied claims, and one run."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from graph_engine.closed_loop import World, run, wrong_measure, _posteriors  # noqa: E402
from graph_engine.regime_posterior import RegimePosterior  # noqa: E402


def test_world_signs_change_at_the_declared_points_only():
    w = World(seed=3)
    for p in range(w.n_pairs):
        xs = np.linspace(0, 1, 401); s = np.array([w.sign(p, x) for x in xs])
        assert int((np.diff(s) != 0).sum()) == len(w.transitions[p])


def test_wrong_measure_is_zero_when_the_posterior_knows_the_truth():
    w = World(seed=1, n_pairs=3)
    posts = []
    for p in range(3):
        rp = RegimePosterior(0.0, 1.0)
        for x in np.linspace(0.01, 0.99, 40):
            rp.add_probe(float(x), w.sign(p, float(x)), 0.999)
        posts.append(rp)
    assert wrong_measure(w, posts) < 0.03


def test_copies_do_not_add_support_under_lineage_weighting():
    w = World(seed=0)
    lin, ind = _posteriors(w, False), _posteriors(w, True)
    n_lin = sum(c[3] for rp in lin for c in rp.claims); n_ind = sum(c[3] for rp in ind for c in rp.claims)
    assert n_ind > n_lin                                    # copies were counted as independent in `ind`
    assert n_lin == len({(p, round(a, 6), round(b, 6), s, r) for p, a, b, s, r in w.claims})   # one per distinct root


def test_engine_run_spends_the_budget_and_records_a_curve():
    w = World(seed=2, n_pairs=4)
    r = run(w, "engine", budget=12, seed=2)
    assert r["cost"][-1] >= 12 and len(r["wrong"]) == len(r["cost"]) and r["final_wrong"] <= 4.0
