"""record_guarantee: the admitted error rate stays under α on held-out records, where the product gate does not."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "examples" / "engine_experiments"))
import pytest
pytest.importorskip("sklearn")
from graph_engine.record_guarantee import least_certain_field_score, fit_threshold, admit  # noqa: E402
from graph_engine.typed_extraction import extract  # noqa: E402
from graph_engine.lens_pooling import balanced_lenses  # noqa: E402
import e12_documents_to_next_action as e12  # noqa: E402

FIELDS = [("atom", "a"), ("atom", "b"), ("atom", "c"), ("polarity", "ax")]; BAL = balanced_lenses({"option_order": 2, "phrasing_a": 2, "phrasing_b": 2})


def _records(n, q, seed):
    rng = np.random.default_rng(seed); P, ok = [], []
    for _ in range(n):
        tr = {f: bool(rng.integers(2)) for f in FIELDS}; ex = extract("", FIELDS, e12.make_judge(rng, tr, q), BAL)
        P.append([ex["p"][f] for f in FIELDS]); ok.append(all((ex["p"][f] > 0.5) == tr[f] for f in FIELDS))
    return np.array(P), np.array(ok)


def _judge(rng, tr, q, strength):
    """e12's judge with the logit strength as an argument: a weak judge puts p near 0.5, which the score can see."""
    flip = {f: rng.random() < q for f in tr}; noise = {}

    def judge(text, field, lens):
        key = (field, tuple(sorted(lens.items())))
        if key not in noise: noise[key] = e12.NOISE * rng.standard_normal()
        v = tr[field] ^ flip[field]
        return 1 / (1 + np.exp(-((strength if v else -strength) + (e12.ORDER_BIAS if lens["option_order"] == 0 else -e12.ORDER_BIAS) + noise[key])))
    return judge


def _records_mixed(n, seed):
    """Half the records get e12's strong judge with 2 % shared misreading, half a weak judge (logit 0.4) with 12 %.
    The population is then heterogeneous in how right it is AND the score sees it, so the threshold has somewhere to sit
    (with one homogeneous judge the error rate is ~0.38 in every score bin, which is why the α = 0.45 demonstration below
    admits everything and α = 0.1 admits nothing — neither exercises the selection)."""
    rng = np.random.default_rng(seed); P, ok = [], []
    for _ in range(n):
        tr = {f: bool(rng.integers(2)) for f in FIELDS}; weak = rng.random() < 0.5
        ex = extract("", FIELDS, _judge(rng, tr, 0.12 if weak else 0.02, 0.4 if weak else e12.STRENGTH), BAL)
        P.append([ex["p"][f] for f in FIELDS]); ok.append(all((ex["p"][f] > 0.5) == tr[f] for f in FIELDS))
    return np.array(P), np.array(ok)


def test_level_holds_where_the_threshold_is_interior():
    """The α = 0.45 / α = 0.1 pair below is degenerate (everything / nothing). Here the score is informative — base error
    ~0.40, error ~0.07 among the most certain third — so there is an α strictly between the best bin's error rate and the
    base rate at which the rule admits SOME records and refuses others. α is read off the data: the smallest α on a 0.05
    grid whose calibration coverage reaches 0.5. Measured: α = 0.20, held-out coverage 0.52 (0.45–0.62 over 20 splits),
    admitted error 0.12 (max 0.17), level held in 20 of 20 splits."""
    Pc0, ok0 = _records_mixed(600, 0); s0 = least_certain_field_score(Pc0)
    alpha = next(a for a in np.arange(0.05, 0.95, 0.05) if fit_threshold(s0, ok0, alpha=float(a))["admitted_in_calibration"] / len(s0) >= 0.5)
    cov, err, held = [], [], 0
    for seed in range(20):
        Pc, okc = _records_mixed(600, seed); Pt, okt = _records_mixed(600, 100 + seed)
        a = admit(least_certain_field_score(Pt), fit_threshold(least_certain_field_score(Pc), okc, alpha=float(alpha)))
        e = float(1 - okt[a].mean()) if a.sum() else 0.0
        cov.append(float(a.mean())); err.append(e); held += e <= alpha
    print(f"interior alpha={alpha:.2f} coverage mean={np.mean(cov):.3f} min={min(cov):.3f} max={max(cov):.3f} "
          f"admitted error mean={np.mean(err):.3f} max={max(err):.3f} level held {held}/20")
    assert 0.2 < np.mean(cov) < 0.8 and all(0.2 < c < 0.8 for c in cov)     # the threshold is interior in every split
    assert held >= 19                                                       # and the level holds where it can be broken


def test_admitted_error_rate_is_controlled_where_the_product_gate_is_not():
    """Measured: 4 fields, 10 % shared misreading → ~0.61 of records entirely right. The product gate at nominal 0.9 admits
    records that are right ~0.65 of the time (level broken in every split); the guarantee at α = 0.1 admits nothing, and at a
    reachable α = 0.45 holds in every split.
    This population is homogeneous: the error rate is ~0.35–0.39 in EVERY score bin, so the threshold is never interior —
    α = 0.45 (above the base rate 0.39) admits 0.998 of held-out records with error 0.35, α = 0.1 admits none. What is
    tested here is only the two ENDS (refusal when the target is unreachable, level when nothing is selected); the
    selection itself is tested in test_level_holds_where_the_threshold_is_interior."""
    broken_prod, refused, broken_conf, cov = 0, 0, 0, []
    for seed in range(20):
        Pc, okc = _records(600, 0.1, seed); Pt, okt = _records(600, 0.1, 100 + seed); s_c, s_t = least_certain_field_score(Pc), least_certain_field_score(Pt)
        prod = np.prod(np.maximum(Pt, 1 - Pt), 1) >= 0.9; broken_prod += prod.sum() > 0 and (1 - okt[prod].mean()) > 0.1
        refused += fit_threshold(s_c, okc, alpha=0.1)["admitted_in_calibration"] == 0
        a = admit(s_t, fit_threshold(s_c, okc, alpha=0.45)); cov.append(a.mean()); broken_conf += (1 - okt[a].mean()) > 0.45
    print(f"alpha=0.45 coverage mean={np.mean(cov):.3f} min={min(cov):.3f}; alpha=0.1 refused {refused}/20")
    assert broken_prod >= 15 and refused == 20 and broken_conf <= 1
    assert np.mean(cov) > 0.95   # not a coverage claim: α is above the base error rate, so the rule admits (almost) everything


def test_threshold_is_monotone_in_alpha_and_refuses_when_nothing_is_safe():
    Pc, okc = _records(500, 0.1, 7); s = least_certain_field_score(Pc)
    t1, t2 = fit_threshold(s, okc, 0.42)["threshold"], fit_threshold(s, okc, 0.6)["threshold"]
    assert t2 >= t1
    assert fit_threshold(s, okc, 0.1)["admitted_in_calibration"] == 0                  # an unreachable target admits nothing
    assert fit_threshold(s, np.zeros_like(okc), 0.1)["admitted_in_calibration"] == 0
