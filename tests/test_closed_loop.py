"""closed_loop: the world's sign structure, the truth-based score, lineage weighting of copied claims, and one run."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from graph_engine.closed_loop import (World, run, wrong_measure, _posteriors, _estimate_roots,  # noqa: E402
                                      DEFAULT_PRIOR)
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


def test_guard_spends_its_share_of_the_budget_on_the_model_check_probe():
    """+guard buys model_check_probe with `guard` × budget and no more than one cost unit off it."""
    w = World(seed=5, n_pairs=4)
    for share in (0.1, 0.25, 0.5):
        r = run(w, "engine+guard", budget=20, seed=5, guard=share)
        assert abs(r["spent_guard"] - share * 20) <= 1.0, (share, r["spent_guard"])
    assert run(w, "engine", budget=20, seed=5)["spent_guard"] == 0.0


def test_online_reliability_ranks_a_planted_bad_source_below_a_planted_good_one():
    """Source 0 is right 0.98 of the time, source 4 is a coin; after 20 probes the estimate must order them."""
    w = World(seed=7, n_pairs=16, n_sources=5, claims_per_pair=6, copy_fraction=0.0,
              source_reliability=(0.98, 0.9, 0.9, 0.85, 0.5))
    posts, roots = _posteriors(w, independent_copies=False, with_roots=True)
    rng = np.random.default_rng(0)
    probe_log = [[] for _ in range(w.n_pairs)]
    for i in range(20):                                   # 20 probes with the exact instrument, spread over the pairs
        p = i % w.n_pairs
        x = float(rng.random())
        s = w.probe(p, x, 0.99, rng)
        posts[p].add_probe(x, s, 0.99)
        probe_log[p].append((x, s, 0.99))
    rs = _estimate_roots(w, posts, roots, probe_log, bins=3)
    assert rs[4] < rs[0], rs
    assert rs[4] <= 0.8 <= rs[0]


def test_replay_learns_the_worlds_priors_from_its_own_history():
    """The running empirical-Bayes estimate after 20 worlds at e21's settings, against the World's true parameters.

    p_two and the mean source reliability land on the truth. p_flip does NOT: it moves from the default 0.30 to ~0.45
    against a true 0.60 and stops there — a transition no probe brackets keeps the prior's mass in the posterior, so the
    family masses under-count transitions and the EM fixed point sits below the truth. The test pins that bias rather
    than asserting a convergence that was not measured.
    """
    state, hist = dict(DEFAULT_PRIOR), {k: [v] for k, v in DEFAULT_PRIOR.items()}
    for s in range(20):
        r = run(World(seed=100 + s), "engine+replay", budget=40, seed=s, replay_prior=state)
        for k in DEFAULT_PRIOR:
            hist[k].append(r["world_stats"][k]); state[k] = float(np.mean(hist[k]))
    assert abs(state["p_two"] - World().p_two) < 0.05, state
    assert abs(state["reliability"] - float(np.mean(World().source_reliability[:5]))) < 0.15, state
    assert state["p_flip"] > DEFAULT_PRIOR["p_flip"] + 0.10, state            # it does move toward the truth
    assert abs(state["p_flip"] - World().p_transition) < 0.25, state          # but keeps a downward bias of ~0.15


def test_guard2_spends_its_share_on_pairs_of_probes_bought_together():
    """+guard2 buys model_check_pair: both probes are executed back to back on the same pair and both are charged to
    the guard share, which is never exceeded by more than one cost unit. A pair costs twice an instrument, so the
    guard spend is even; the last pair that does not fit is not bought (share 0.1 on budget 20 buys one pair)."""
    w = World(seed=5, n_pairs=4)
    for share in (0.1, 0.25, 0.5):
        r = run(w, "engine+guard2", budget=20, seed=5, guard=share)
        assert 0 < r["spent_guard"] <= share * 20 + 1.0, (share, r["spent_guard"])
        assert r["spent_guard"] % 2 == 0, (share, r["spent_guard"])
    assert run(w, "engine+guard2+replay+majority", budget=20, seed=5)["spent_guard"] > 0
    for bad in ("engine+guard+guard2", "engine+guard3"):
        try:
            run(w, bad, budget=4, seed=0); assert False, bad
        except ValueError:
            pass


def test_majority_flag_reads_claims_as_majority_reports():
    w = World(seed=5, n_pairs=4)
    r = run(w, "engine+majority", budget=8, seed=5)
    assert r["believed_wrong"] > run(w, "engine", budget=8, seed=5)["believed_wrong"]   # less sure inside wide boxes
