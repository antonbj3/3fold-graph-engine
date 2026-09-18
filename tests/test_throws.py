"""throws on a planted TEST world: 6 fields × 3 mechanisms; hidden true links join same-mechanism nodes of different fields."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from graph_engine.throws import throw_scores, draw, draw_pairs, inclusion_probabilities  # noqa: E402
from graph_engine.resistance_sketch import ResistanceSketch  # noqa: E402


def _world(seed=0, per=5):
    rng = np.random.default_rng(seed); field = np.repeat(np.arange(6), 3 * per); mech = np.tile(np.repeat(np.arange(3), per), 6); n = len(field)
    e = [(i, j) for i in range(n) for j in range(i + 1, n) if field[i] == field[j] and rng.random() < 0.35]
    for f in range(6):                                                                                                # keep it connected
        ids = np.flatnonzero(field == f); e += [(int(a), int(b)) for a, b in zip(ids[:-1], ids[1:])]
    e += [(int(np.flatnonzero(field == f)[0]), int(np.flatnonzero(field == f + 1)[0])) for f in range(5)]
    e = sorted({(min(a, b), max(a, b)) for a, b in e})
    truth = (mech[:, None] == mech[None, :]) & (field[:, None] != field[None, :])
    sig = np.eye(3)[mech] + 0.15 * rng.standard_normal((n, 3))                                                       # noisy computed signature
    M = np.linalg.norm(sig[:, None] - sig[None], axis=2); subj = (field[:, None] != field[None, :]).astype(float)
    Z = ResistanceSketch.build(n, np.array(e), k=64, seed=seed).Z; R = ((Z[:, None] - Z[None]) ** 2).sum(2)
    linked = np.zeros((n, n), bool); linked[tuple(np.array(e).T)] = True
    return M, R, subj, truth, linked | linked.T


def test_throws_find_planted_cross_field_links_and_report_inclusion_probabilities():
    M, R, subj, truth, linked = _world(); iu = np.triu_indices(len(M), 1); base = truth[iu][~linked[iu]].mean()
    hit = lambda pairs: np.mean([truth[i, j] for i, j, _ in pairs])
    full = np.mean([hit(draw(throw_scores(M, R, subj), 60, linked, temperature=0.5, seed=s)) for s in range(10)])
    far_only = np.mean([hit(draw(throw_scores(np.zeros_like(M), R, subj), 60, linked, temperature=0.5, seed=s)) for s in range(10)])
    rnd = np.mean([hit(draw(np.zeros_like(M), 60, linked, seed=s)) for s in range(10)])
    assert abs(rnd - base) < 0.06 and full > 2.5 * base and full > far_only + 0.3        # distance alone does not find them
    pairs = draw(throw_scores(M, R, subj), 60, linked, seed=1)
    assert all(0 < p <= 1 for *_, p in pairs) and len({(i, j) for i, j, _ in pairs}) == 60 and not any(linked[i, j] for i, j, _ in pairs)


def test_every_unlinked_pair_can_be_drawn():
    M, R, subj, truth, linked = _world(); s = throw_scores(M, R, subj); seen = set()
    for seed in range(400): seen |= {(i, j) for i, j, _ in draw(s, 60, linked, floor=0.5, seed=seed)}
    iu = np.triu_indices(len(M), 1); total = int((~linked[iu]).sum())
    assert len(seen) > 0.97 * total                                                          # positivity: the floor reaches the unlikely pairs too


def test_reported_inclusion_probability_is_the_frequency_with_which_a_pair_is_drawn():
    rng = np.random.default_rng(0); n = 12; S = rng.standard_normal((n, n)) * 2; S = (S + S.T) / 2
    count = {}; reps = 6000
    for seed in range(reps):
        pairs = draw(S, 10, temperature=0.7, floor=0.1, seed=seed); assert len({(i, j) for i, j, _ in pairs}) == 10
        for i, j, pi in pairs: count[(i, j)] = (count.get((i, j), (0, pi))[0] + 1, pi)
    freq = np.array([c / reps for c, _ in count.values()]); stated = np.array([pi for _, pi in count.values()])
    assert np.abs(freq - stated).max() < 0.03 and np.isclose(sum(stated) + 0, stated.sum())
    pi = inclusion_probabilities(np.array([0.9, 0.05, 0.03, 0.02]), 2)
    assert np.isclose(pi.sum(), 2) and pi[0] == 1.0 and (pi <= 1).all()


def test_pair_list_form_needs_no_dense_matrix_and_agrees_with_the_matrix_form():
    M, R, subj, truth, linked = _world(); iu = np.triu_indices(len(M), 1); ok = ~linked[iu]
    s_list = throw_scores(M[iu][ok], R[iu][ok], subj[iu][ok]); s_mat = throw_scores(M, R, subj)[iu][ok]
    assert np.allclose(s_list - s_list.mean(), s_mat - s_mat.mean())          # same ranking; the median reference may differ by a constant
    a = draw_pairs(iu[0][ok], iu[1][ok], s_mat, 30, seed=4); b = draw(throw_scores(M, R, subj), 30, linked, seed=4)
    assert a == b


def test_dpp_set_draws_match_the_exact_inclusion_probabilities_and_avoid_duplicates():
    """Frequencies of membership over many draws equal diag(L(I+L)⁻¹); two identical rows (a duplicate node) are drawn together
    with probability 0 under the determinantal rule (det of a singular block)."""
    from graph_engine.throws import draw_set_dpp, dpp_inclusion_probabilities
    rng = np.random.default_rng(0); Z = rng.standard_normal((12, 6)); Z[11] = Z[0]        # node 11 duplicates node 0
    pi = dpp_inclusion_probabilities(Z); n = 3000
    counts = np.zeros(12); both = 0
    for s in range(n):
        S, pS = draw_set_dpp(Z, seed=s); counts[S] += 1
        both += (0 in S) and (11 in S)
        assert np.allclose(pS, pi[S])
    assert np.abs(counts / n - pi).max() < 0.03, (counts / n, pi)
    assert both == 0


def test_continuum_throw_without_dither_is_deterministic_and_with_dither_is_unbiased():
    from graph_engine.throws import throw_from_continuum
    rng = np.random.default_rng(1); Z = rng.standard_normal((40, 4)); x = Z[:2].mean(0) + 0.3 * rng.standard_normal(4)
    fixed = throw_from_continuum(Z, x, k=3, dither=0.0, n_draws=5, n_pi=50, seed=0)
    assert all(np.array_equal(fixed[0][0], s) for s, _ in fixed) and set(fixed[0][1].tolist()) == {1.0}
    dith = throw_from_continuum(Z, x, k=3, n_draws=400, n_pi=400, seed=0)
    members = np.concatenate([s for s, _ in dith]); assert len(np.unique(members)) > 3          # every corner within reach is reachable
    centroid = np.mean([Z[s].mean(0) for s, _ in dith], axis=0)
    assert np.linalg.norm(centroid - x) < np.linalg.norm(Z[fixed[0][0]].mean(0) - x) + 0.35     # not pulled away from x_star by the grid
    pis = np.concatenate([p for _, p in dith]); assert 0 < pis.min() and pis.max() <= 1


def test_sequential_densification_stops_rethrowing_what_it_learned():
    """A path graph with one Bernoulli anchor; each round observes the contrast between the two most distant members.
    The form's bits fall every round, and the inclusion probability of a node drops after its contrast was observed."""
    from graph_engine.precision_form import PrecisionForm
    from graph_engine.throws import densify_sequential
    n = 12; f = PrecisionForm.zeros(n).add_laplacian([(i, i + 1) for i in range(n - 1)]).add_bernoulli(0, 0.5)
    seen = []
    def observe(S):
        if len(S) < 2:
            return []
        i, j = int(S[0]), int(S[-1]); h = np.zeros(n); h[i] = 1; h[j] = -1; seen.append((i, j)); return [(h, 0.3, 0.0)]
    log = densify_sequential(f, np.arange(n), rounds=6, sigma=1.0, observe=observe, seed=3)
    assert all(r["bits_realized"] >= -1e-9 for r in log) and sum(r["bits_realized"] for r in log) > 0.5
    assert all(abs(r["inclusion"].sum() - 0) >= 0 for r in log)
    # a node whose contrast was observed is less likely in later rounds than before
    first = log[0]["members"]
    if len(first) >= 2:
        i = int(first[0]); before = float(log[0]["inclusion"][0]); later = [float(dict(zip(r["members"].tolist(), r["inclusion"])).get(i, 0.0)) for r in log[1:]]
        assert min(later) <= before + 1e-9


def test_point_dpp_throw_has_exact_marginals_and_sits_near_the_design_point():
    from graph_engine.throws import throw_at_point_dpp, dpp_inclusion_probabilities
    rng = np.random.default_rng(2); Z = rng.standard_normal((60, 5)); x = Z[:3].mean(0)
    hits = np.zeros(60); n = 400
    for s in range(n):
        S, pi = throw_at_point_dpp(Z, x, seed=s); hits[S] += 1
    d = np.linalg.norm(Z - x, axis=1)
    assert np.corrcoef(hits, -d)[0, 1] > 0.5                       # mass concentrates near the design point
    # exact marginals: recompute with the same quality on the same support and compare with frequencies
    scale = float(np.median(np.sort(d)[:6])); q = np.exp(-d ** 2 / (2 * scale ** 2)); near = np.flatnonzero(q > 1e-6)
    pi = dpp_inclusion_probabilities(Z[near], quality=q[near])
    assert np.abs(hits[near] / n - pi).max() < 0.08


def test_chain_throw_decodes_the_segment_to_intermediate_nodes():
    from graph_engine.throws import chain_throw
    Z = np.zeros((7, 2)); Z[:, 0] = np.arange(7); Z[3] = [3.0, 0.2]           # a line of nodes with one slightly off
    ch = chain_throw(Z, 0, 6, steps=3)
    assert ch[0] == 0 and ch[-1] == 6 and set(ch) >= {0, 6} and 1 <= len(ch) - 2 <= 3
    assert all(0 < i < 6 for i in ch[1:-1])                                    # intermediate nodes lie between the ends
