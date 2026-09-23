"""RegimePosterior probe choosers: the optional second-moment screen (method="moments") returns the direct
result bit for bit. The reference functions below are verbatim copies of the per-candidate loops as they
were before the screen was added (Graph c9dde8d), so the default path is pinned as well."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_engine.regime_posterior import RegimePosterior  # noqa: E402


def _h(p):
    return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def ref_best_probe(rp, reliability):
    cells, w, F, post = rp._with_probes()
    pp = post @ F
    now = float(rp._u(pp) @ w)
    xs = list(cells.mean(1)) + [rp.lo, rp.hi]
    cs = list(range(len(cells))) + [0, len(cells) - 1]
    best = (xs[0], -1.0)
    for x, c in zip(xs, cs):
        f = F[:, c]
        after = 0.0
        for sg in (1, -1):
            like = (f if sg > 0 else 1 - f) * reliability + (1 - (f if sg > 0 else 1 - f)) * (1 - reliability)
            pout = float(post @ like)
            if pout <= 0:
                continue
            q = post * like / pout
            qq = q @ F
            after += pout * float(rp._u(qq) @ w)
        if now - after > best[1]:
            best = (float(x), now - after)
    return best


def ref_total_value_probe(rp, reliability, lam=1.0):
    cells, w, F, post = rp._with_probes()
    pp = post @ F; now_u = float(rp._u(pp) @ w)
    two = np.zeros(len(post)); two[rp._n_one:] = 1.0
    now_f = _h(float(post @ two)) if len(post) > rp._n_one else 0.0
    best = (float(cells[0].mean()), -1.0)
    for c in range(len(cells)):
        f = F[:, c]; after_u = after_f = 0.0
        for sg in (1, -1):
            like = (f if sg > 0 else 1 - f) * reliability + (1 - (f if sg > 0 else 1 - f)) * (1 - reliability)
            pout = float(post @ like)
            if pout <= 0:
                continue
            q = post * like / pout
            after_u += pout * float(rp._u(q @ F) @ w)
            after_f += pout * (_h(float(q @ two)) if len(post) > rp._n_one else 0.0)
        gain = (now_u - after_u) + lam * (now_f - after_f)
        if gain > best[1]:
            best = (float(cells[c].mean()), gain)
    return best


def _like(f, sg, r):
    ff = f if sg > 0 else 1 - f
    return ff * r + (1 - ff) * (1 - r)


def ref_model_check_single(rp, reliability):
    cells, w, F, post = rp._with_probes()
    two = np.zeros(len(post)); two[rp._n_one:] = 1.0
    now = _h(float(post @ two))
    out = []
    for c in range(len(cells)):
        f = F[:, c]; after = 0.0
        for sg in (1, -1):
            like = _like(f, sg, reliability)
            pout = float(post @ like)
            if pout <= 0:
                continue
            after += pout * _h(float((post * like / pout) @ two))
        out.append((now - after, c))
    return now, out


def ref_model_check_probe(rp, reliability):
    cells = rp._with_probes()[0]
    if rp._with_probes()[3].shape[0] == rp._n_one:
        return (0.5 * (rp.lo + rp.hi), 0.0)
    _, single = ref_model_check_single(rp, reliability)
    best = (float(cells[0].mean()), -1.0)
    for g, c in single:
        if g > best[1]:
            best = (float(cells[c].mean()), g)
    return best


def ref_model_check_pair(rp, reliability):
    cells, w, F, post = rp._with_probes()
    mid = 0.5 * (rp.lo + rp.hi)
    if len(post) == rp._n_one:
        return ((mid, mid), 0.0)
    two = np.zeros(len(post)); two[rp._n_one:] = 1.0
    now, single = ref_model_check_single(rp, reliability)
    cand = [c for _, c in sorted(single, reverse=True)[:12]]
    best = ((float(cells[cand[0]].mean()), float(cells[cand[0]].mean())), -1.0)
    for i, c1 in enumerate(cand):
        f1 = F[:, c1]
        for c2 in cand[i + 1:]:
            f2 = F[:, c2]; after = 0.0
            for s1 in (1, -1):
                l1 = _like(f1, s1, reliability)
                p1 = float(post @ l1)
                if p1 <= 0:
                    continue
                q1 = post * l1 / p1
                for s2 in (1, -1):
                    l2 = _like(f2, s2, reliability)
                    p2 = float(q1 @ l2)
                    if p2 <= 0:
                        continue
                    after += p1 * p2 * _h(float((q1 * l2 / p2) @ two))
            if now - after > best[1]:
                x1, x2 = float(cells[c1].mean()), float(cells[c2].mean())
                best = ((min(x1, x2), max(x1, x2)), now - after)
    return best


def random_regime(seed, symmetric=False):
    rng = np.random.default_rng(seed)
    rp = RegimePosterior(0.0, 1.0, reliability=0.75, p_flip=0.3,
                         n_grid=int(rng.choice([2, 4, 9, 20, 48])),
                         potential=str(rng.choice(["entropy", "error"])),
                         p_two=float(rng.choice([0.0, 0.05, 0.3])),
                         claim_model=str(rng.choice(["pointwise", "majority"])))
    if symmetric:
        return rp                                      # no evidence: mirror-symmetric, exact ties everywhere
    for _ in range(int(rng.integers(0, 4))):
        a, b = sorted(rng.uniform(0, 1, 2))
        rp.add_claim(float(a), float(b), int(rng.choice([-1, 1])), 1.0, float(rng.choice([0.6, 0.75, 0.9])))
    for _ in range(int(rng.integers(0, 8))):
        rp.add_probe(float(rng.uniform(0, 1)), int(rng.choice([-1, 1])), float(rng.choice([0.7, 0.95])),
                     float(rng.choice([1.0, 0.25])))
    return rp


RELS = (0.5, 0.5 + 1e-9, 0.3, 0.7, 0.95, 0.999999)


@pytest.mark.parametrize("seed", range(24))
def test_default_and_moments_equal_the_original_loops_bit_for_bit(seed):
    rp = random_regime(seed, symmetric=(seed % 6 == 5))
    for r in RELS:
        ref = [ref_best_probe(rp, r), ref_total_value_probe(rp, r), ref_total_value_probe(rp, r, 3.0),
               ref_model_check_probe(rp, r), ref_model_check_pair(rp, r)]
        for m in ("direct", "moments"):
            got = [rp.best_probe(r, method=m), rp.total_value_probe(r, method=m),
                   rp.total_value_probe(r, lam=3.0, method=m), rp.model_check_probe(r, method=m),
                   rp.model_check_pair(r, method=m)]
            assert repr(got) == repr(ref), (seed, r, m)


def test_symmetric_regime_resolves_exact_ties_by_the_direct_rule():
    rp = RegimePosterior(0.0, 1.0, n_grid=8)            # no evidence: gains mirror around the centre
    x, g = rp.best_probe(0.9, method="moments")
    assert (x, g) == ref_best_probe(rp, 0.9)
    assert rp.last_probe_info["path"] == "moments" and rp.last_probe_info["resolved"] >= 2


def test_screen_matches_the_exact_gains_closely():
    rp = random_regime(3)
    cells, w, F, post = rp._with_probes()
    pp = post @ F
    after, _, _ = rp._screen_after_u(pp, rp._second_moments(post, F), w, np.arange(len(cells)), 0.9)
    exact = [rp._after_u(post, F, w, F[:, c], 0.9) for c in range(len(cells))]
    assert np.max(np.abs(after - np.array(exact))) < 1e-12 * (rp.hi - rp.lo) + 1e-12


@pytest.mark.parametrize("r", [0.0, 1.0, float("nan")])
def test_reliability_without_a_usable_screen_falls_back_to_direct(r):
    rp = random_regime(4)
    with np.errstate(all="ignore"):
        a = rp.best_probe(r, method="moments")
        assert rp.last_probe_info["path"] == "direct"
        b = rp.best_probe(r)
    assert repr(a) == repr(b)


def test_unknown_method_is_rejected_and_state_is_not_mutated():
    rp = random_regime(5)
    probes = list(rp.probes)
    with pytest.raises(ValueError):
        rp.best_probe(0.9, method="covariance")
    rp.best_probe(0.9, method="moments"); rp.model_check_pair(0.9, method="moments")
    assert rp.probes == probes


def test_chunked_second_moments_equal_one_shot():
    rp = random_regime(7)
    cells, w, F, post = rp._with_probes()
    rp._SCREEN_CHUNK = 7
    S = rp._second_moments(post, F)
    assert np.allclose(S, F.T @ (post[:, None] * F), rtol=0, atol=1e-14)
