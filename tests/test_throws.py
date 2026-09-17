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
