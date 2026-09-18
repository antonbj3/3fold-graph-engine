"""RepresentationProbe on the smallest local model: fit on two quantity pairs, read two OTHER quantity pairs.

Skipped unless a local Qwen2.5-0.5B-Instruct directory is present (PROBE_TEST_MODEL, else
~/projects/hunt_3fold/models/q) and transformers is importable — the model is not part of the repo.
64 sentences: 4 pairs × 2 asserted signs × 8 templates (the e7 forms, including the inverted and negated ones,
so half of the sentences contradict textbook physics and the surface word alone does not give the label).
"""
import os
from pathlib import Path

import numpy as np
import pytest

MODEL = os.environ.get("PROBE_TEST_MODEL", str(Path.home() / "projects/hunt_3fold/models/q"))
pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not Path(MODEL).is_dir(), reason=f"no local model at {MODEL}")

from graph_engine.representation_probe import RepresentationProbe  # noqa: E402

PAIRS = [("temperature", "electrical resistance of copper", 1), ("altitude", "air pressure", -1),
         ("wire length", "electrical resistance", 1), ("porosity", "fatigue strength", -1)]


def sentences(x, y, s):
    """The e7 sentence forms for an asserted sign s between x and y."""
    up, dn = ("an increase", "a decrease") if s > 0 else ("a decrease", "an increase")
    return [f"An increase in {x} leads to {up} in {y}.", f"The {y} {'rises' if s > 0 else 'falls'} as {x} grows.",
            f"Reducing {x} {'lowers' if s > 0 else 'raises'} the {y}.",
            f"Higher {x} is associated with {'higher' if s > 0 else 'lower'} {y}.",
            f"The {y} is {'directly' if s > 0 else 'inversely'} proportional to {x}.",
            f"When {x} drops, the {y} {'drops' if s > 0 else 'climbs'}.",
            f"It is not the case that the {y} {'decreases' if s > 0 else 'increases'} with {x}; it "
            f"{'increases' if s > 0 else 'decreases'}.",
            f"Samples with lower {x} showed {'lower' if s > 0 else 'higher'} {y}."]


def _items(pairs):
    texts, xy, lab = [], [], []
    for x, y, _phys in pairs:
        for s in (1, -1):
            for t in sentences(x, y, s):
                texts.append(t); xy.append((x, y)); lab.append(s > 0)
    return texts, xy, np.array(lab)


@pytest.fixture(scope="module")
def fitted():
    dev = "cuda" if os.environ.get("E7_DEVICE") == "cuda" else "cpu"
    probe = RepresentationProbe(MODEL, device=dev, batch_size=16, name="q0.5b")
    tr_t, tr_p, tr_y = _items(PAIRS[:2])
    probe.fit(tr_t, tr_p, tr_y)                       # layer chosen by inner split over the two training pairs
    return probe


def test_reads_held_out_quantity_pairs(fitted):
    te_t, te_p, te_y = _items(PAIRS[2:])
    assert len(te_t) == 32
    p = fitted.predict_proba(te_t, te_p)
    assert p.shape == (32,) and ((p >= 0) & (p <= 1)).all()
    acc = float(((p > 0.5) == te_y).mean())
    assert acc > 0.75, f"probe accuracy on held-out quantity pairs was {acc:.3f}"


def test_layer_was_chosen_on_training_data(fitted):
    assert fitted.layer is not None and 0 <= fitted.layer < fitted.n_layers
    assert fitted.layer_scores is not None and len(fitted.layer_scores) == fitted.n_layers


def test_lens_callable_shape_and_lineage(fitted):
    lens = fitted.as_lens()
    assert callable(lens)
    assert lens.lineage == f"probe:q0.5b:{fitted.layer}" and lens.lineage.startswith("probe:")
    assert lens.origin == "representation_probe"
    te_t, te_p, _ = _items(PAIRS[2:])
    out = lens(te_t[:8], te_p[:8])
    assert out.shape == (8,) and ((out >= 0) & (out <= 1)).all()
    assert np.allclose(out, fitted.predict_proba(te_t[:8], te_p[:8]))


def test_predict_sign_never_abstains(fitted):
    te_t, te_p, _ = _items(PAIRS[2:])
    s = fitted.predict_sign(te_t[:8], te_p[:8])
    assert set(np.unique(s)) <= {-1, 1} and (s != 0).all()


# ---------------------------------------------------------------- difference-of-means direction (no model needed)
from graph_engine.representation_probe import difference_of_means_direction, remove_direction  # noqa: E402


def _planted(n=200, d=6, seed=0):
    """Two coordinates: 0 carries the asserted sign (the reading), 1 carries agreement-with-physics (the prior)."""
    rng = np.random.default_rng(seed)
    sign = rng.integers(0, 2, n) * 2 - 1
    agree = rng.integers(0, 2, n).astype(bool)
    H = 0.1 * rng.standard_normal((n, d))
    H[:, 0] += 3.0 * sign
    H[:, 1] += 2.0 * agree
    return H, sign, agree


def test_direction_points_along_the_planted_group_axis():
    H, sign, agree = _planted()
    d = difference_of_means_direction(H, agree)
    assert np.isclose(np.linalg.norm(d), 1.0)
    assert abs(d[1]) > 0.95 and d[1] > 0


def test_strata_cancel_a_correlated_nuisance():
    """With agreement 80 % aligned with the sign, the unstratified direction picks up the reading axis; stratifying
    by the asserted sign removes it."""
    H, sign, _ = _planted()
    rng = np.random.default_rng(1)
    agree = np.where(rng.random(len(sign)) < 0.8, sign > 0, sign < 0)
    H = H.copy()
    H[:, 1] += 2.0 * agree
    plain = difference_of_means_direction(H, agree)
    strat = difference_of_means_direction(H, agree, strata=sign)
    assert abs(plain[0]) > 0.5 and abs(strat[0]) < 0.1
    assert abs(strat[1]) > 0.95


def test_remove_direction_zeroes_the_component_and_keeps_the_rest():
    H, sign, agree = _planted()
    d = difference_of_means_direction(H, agree, strata=sign)
    Hp = remove_direction(H, d)
    assert np.abs(Hp @ d).max() < 1e-9
    assert np.corrcoef(Hp[:, 0], sign)[0, 1] > 0.99          # the reading axis survives
    assert abs(np.corrcoef(Hp[:, 1], agree.astype(float))[0, 1]) < 0.2


def test_remove_direction_target_and_alpha():
    H, sign, agree = _planted()
    d = difference_of_means_direction(H, agree, strata=sign)
    c = float((H @ d).mean())
    assert np.allclose(remove_direction(H, d, target=c) @ d, c)
    assert np.allclose(remove_direction(H, d, alpha=0.0), H)
    assert np.allclose(remove_direction(H, 2.5 * d), remove_direction(H, d))


def test_direction_rejects_an_empty_side():
    H, sign, agree = _planted()
    bad = np.where(sign > 0, True, agree)                     # one stratum is all-True
    with pytest.raises(ValueError):
        difference_of_means_direction(H, bad, strata=sign)
