"""RegimeMarkov.best_probe: the optional pairwise joint-marginal screen (method="pairwise") returns the direct
result bit for bit; the reference is a verbatim copy of the per-cell loop as it was before (Graph c9dde8d)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_engine.regime_markov import RegimeMarkov  # noqa: E402


def ref_best_probe(rm, reliability):
    cells, w, mid, _ = rm._geometry()
    pp, _, _ = rm._solve(need_counts=False)
    now = float(rm._u(pp) @ w)
    best = (float(mid[0]), -1.0)
    for c in range(len(cells)):
        after = 0.0
        for sg in (1, -1):
            p = pp[c] if sg > 0 else 1 - pp[c]
            pout = p * reliability + (1 - p) * (1 - reliability)
            if pout <= 0:
                continue
            qpp, _, _ = rm._solve(extra=(float(mid[c]), sg, float(reliability), 1.0), need_counts=False)
            after += pout * float(rm._u(qpp) @ w)
        if now - after > best[1]:
            best = (float(mid[c]), now - after)
    return best


def random_markov(seed, empty=False):
    rng = np.random.default_rng(seed)
    rm = RegimeMarkov(0.0, 1.0, reliability=0.75, n_grid=int(rng.choice([1, 3, 6, 12, 24])),
                      potential=str(rng.choice(["entropy", "error"])),
                      claim_model=str(rng.choice(["majority", "pointwise"])),
                      max_switches_in_block=int(rng.choice([1, 2, 3])))
    if empty:
        return rm
    for _ in range(int(rng.integers(0, 4))):
        a, b = sorted(rng.uniform(0, 1, 2))
        rm.add_claim(float(a), float(b), int(rng.choice([-1, 1])), 1.0, float(rng.choice([0.6, 0.75, 0.9])))
    for _ in range(int(rng.integers(0, 8))):
        rm.add_probe(float(rng.uniform(0, 1)), int(rng.choice([-1, 1])), float(rng.choice([0.7, 0.95])),
                     float(rng.choice([1.0, 0.25])))
    return rm


@pytest.mark.parametrize("seed", range(20))
def test_default_and_pairwise_equal_the_original_loop_bit_for_bit(seed):
    rm = random_markov(seed, empty=(seed % 5 == 4))
    for r in (0.5, 0.5 + 1e-9, 0.3, 0.7, 0.95, 0.999999):
        ref = ref_best_probe(rm, r)
        assert repr(rm.best_probe(r)) == repr(ref)
        assert repr(rm.best_probe(r, method="pairwise")) == repr(ref), (seed, r)


@pytest.mark.parametrize("seed", range(8))
def test_pairwise_joint_predicts_every_hypothetical_posterior(seed):
    rm = random_markov(seed)
    J, pj = rm.pairwise_plus()
    pp, _, _ = rm._solve(need_counts=False)
    cells, w, mid, _ = rm._geometry()
    assert np.allclose(J, J.T, atol=1e-15) and np.max(np.abs(np.clip(pj, 0, 1) - pp)) < 1e-12
    for c in range(len(cells)):
        for sg, r in ((1, 0.9), (-1, 0.8)):
            qpp, _, _ = rm._solve(extra=(float(mid[c]), sg, r, 1.0), need_counts=False)
            lp, lm = (r, 1 - r) if sg > 0 else (1 - r, r)
            pred = (lp * J[:, c] + lm * (pp - J[:, c])) / (lp * pp[c] + lm * (1 - pp[c]))
            assert np.max(np.abs(pred - qpp)) < 1e-12


def test_empty_markov_resolves_mirror_ties_by_the_direct_rule():
    rm = RegimeMarkov(0.0, 1.0, n_grid=8)
    assert rm.best_probe(0.9, method="pairwise") == ref_best_probe(rm, 0.9)
    assert rm.last_probe_info["resolved"] >= 2


def test_unusable_reliability_falls_back_to_direct():
    rm = random_markov(2)
    with np.errstate(all="ignore"):
        a = rm.best_probe(float("nan"), method="pairwise")
        assert rm.last_probe_info["path"] == "direct"
        assert repr(a) == repr(rm.best_probe(float("nan")))


@pytest.mark.parametrize("r", [0.0, 1.0])
def test_degenerate_reliability_fails_the_same_way_on_both_paths(r):
    rm = random_markov(2)
    rm.add_probe(0.5, 1, 0.9)
    for m in ("direct", "pairwise"):
        with pytest.raises(ValueError):
            rm.best_probe(r, method=m)


def test_unknown_method_rejected_and_no_mutation():
    rm = random_markov(3)
    probes = list(rm.probes)
    with pytest.raises(ValueError):
        rm.best_probe(0.9, method="moments")
    rm.best_probe(0.9, method="pairwise")
    assert rm.probes == probes
