"""source_reliability: recovers planted per-origin accuracies from agreement alone, and improves fused signs."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from graph_engine.source_reliability import estimate, root_votes  # noqa: E402
from graph_engine.claim_federation import Federation  # noqa: E402


def _world(seed=0, n_q=300, rel=(0.9, 0.85, 0.8, 0.55, 0.5)):
    rng = np.random.default_rng(seed); truth = rng.choice([-1, 1], n_q); votes = {}
    for q in range(n_q):
        votes[("s", f"o{q}")] = {f"src{k}": int(truth[q] if rng.random() < r else -truth[q]) for k, r in enumerate(rel) if rng.random() < 0.7}
    return truth, votes


def test_recovers_planted_reliabilities_without_labels():
    truth, votes = _world(); est = estimate(votes)
    got = [est[f"src{k}"]["r"] for k in range(5)]
    assert np.allclose(got, [0.9, 0.85, 0.8, 0.55, 0.5], atol=0.07), got


def test_per_origin_weights_beat_one_fixed_reliability():
    truth, votes = _world(seed=1); est = estimate(votes)
    def build(**kw):
        f = Federation(**kw); claims, sources = [], [{"id": f"src{k}"} for k in range(5)]
        for (s, o), d in votes.items():
            for src, v in d.items():
                claims.append({"id": f"{o}-{src}", "subject": s, "object": o, "sign": v, "validity": {}, "evidence": [src]})
        f.add_graph({"graph_id": "T", "claims": claims, "sources": sources}); return f
    fixed, weighted = build(), build(reliability_by_root={s: e["r"] for s, e in est.items()})
    acc = lambda f: np.mean([(f.belief(("s", f"o{q}"))[0] > 0.5) == (truth[q] > 0) for q in range(len(truth))])
    assert acc(weighted) > acc(fixed) + 0.03
    assert root_votes(fixed)[("s", "o0")] == votes[("s", "o0")]


def test_mirror_symmetry_needs_an_anchor():
    """Review found EM converging to the mirrored solution (r → 1 − r, truth → −truth) when most origins are bad,
    scoring the one reliable origin 0.07 and negating its vote. Agreement alone cannot break that symmetry: without an
    anchor the estimate is documented to be wrong here (the majority-good branch); with 20 known truths it is right."""
    truth, votes = _world(seed=0, n_q=400, rel=(0.9, 0.2, 0.2, 0.15, 0.15))
    blind = estimate(votes)
    assert blind["src0"]["r"] <= 0.6 and all(blind[f"src{k}"]["r"] > 0.7 for k in range(1, 5)), blind   # the mirror, as documented
    known = {("s", f"o{q}"): int(truth[q]) for q in range(20)}
    est = estimate(votes, known=known)
    got = [est[f"src{k}"]["r"] for k in range(5)]
    assert np.allclose(got, [0.9, 0.2, 0.2, 0.15, 0.15], atol=0.08), got
